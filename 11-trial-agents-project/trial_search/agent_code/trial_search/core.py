"""trial_search core: the agent loop and the middleware that bounds it.

    orchestrate(question)
        │
        ├─ connect_tools()         SigV4-signed MCP session to the Gateway
        │                          -> semantic_search, expand_neighbors,
        │                             expand_table as LangChain tools
        │
        ├─ build_agent(tools)      create_agent + ToolStrategy(ModelDecision)
        │     │
        │     └─ RetrievalMiddleware        (one class, one state schema)
        │           BEFORE a tool runs      call budget, window clamp,
        │                                   token budget, dedupe — by
        │                                   REWRITING the tool's arguments
        │           AFTER it returns        passages -> state,
        │                                   counters -> state,
        │                                   a readable view -> the model
        │
        └─ assemble                 TrialSearchResponse from STATE

LOOP ENGINEERING — WHAT BOUNDS THIS LOOP

    recursion     each tool has its own call limit per turn. A refused
                  call returns a message, not an exception, so the model
                  can still finish with what it has.
    size          expand_neighbors' window is clamped to MAX_WINDOW.
    tokens        Case A and Case C share EXPANSION_TOKEN_BUDGET. The
                  middleware passes the REMAINING allowance to the Lambda
                  as max_tokens; the Lambda stops nearest-first at it.
    dedupe        every chunk already captured is passed as exclude_ids,
                  so widening a window from 2 to 5 fetches — and charges
                  — only the new ring.

    None of this is a prompt instruction. The model cannot overspend by
    ignoring text, because the arguments that reach the Lambda are the
    middleware's, not the model's.

WHY THE MODEL SEES FULL PASSAGE TEXT HERE

trial_graph shows the model only a compact summary so it cannot retype
bulk rows. This agent is different: its whole job is to read a passage
and judge whether it was cut off. A preview of the first 300 characters
hides exactly the part that decides that — the end. The model sees the
full text; the token budget is what bounds how much text that is.

THREE FACTS VERIFIED AT RUNTIME, NOT ASSUMED

    1. Middleware state must be declared with `state_schema` (a subclass
       of AgentState). An attribute with any other name is ignored and
       every key written through Command is silently dropped — confirmed
       by running a real create_agent loop both ways.
    2. Under ainvoke, a middleware that defines only wrap_tool_call raises
       NotImplementedError on the first tool call. This class defines both
       wrap_tool_call and awrap_tool_call over one shared implementation.
    3. langchain_mcp_adapters returns tool content as a LIST of blocks
       ([{"type": "text", "text": ...}]), plus an optional artifact dict
       with structured_content. _payload() reads either.

WHAT THIS DOES NOT DO

    - It does not decide whether to expand. The model decides; this file
      only bounds and records what it asks for.
    - It does not rank or rerank. Passages keep the order the tools
      return; the final response is ordered by document and position.
"""
from __future__ import annotations

import json
import logging
import operator
from contextlib import asynccontextmanager
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .config import CFG
from .prompt import SYSTEM_PROMPT
from .schemas import (ModelDecision, Passage, RetrievalStats, TrialSearchResponse,
                      collect_usage)

log = logging.getLogger("agent.trial_search.core")

SEARCH, NEIGHBORS, TABLE = "semantic_search", "expand_neighbors", "expand_table"


def _kind(tool_name: str) -> str | None:
    """Gateway tool names may carry a target prefix
    ("trial-search-tools___expand_table"). Match on the suffix."""
    return next((k for k in (SEARCH, NEIGHBORS, TABLE) if tool_name.endswith(k)), None)


# ── MCP connection ──────────────────────────────────────────────────────

@asynccontextmanager
async def connect_tools():
    """SigV4-signed MCP session to the Gateway. aws_service must be
    "bedrock-agentcore" — the signing name the Gateway expects."""
    from mcp import ClientSession
    from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
    from langchain_mcp_adapters.tools import load_mcp_tools

    async with aws_iam_streamablehttp_client(
            endpoint=CFG.gateway_url, aws_service="bedrock-agentcore",
            aws_region=CFG.aws_region) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield await load_mcp_tools(session)


# ── state ───────────────────────────────────────────────────────────────

class RetrievalState(AgentState):
    """Every key the middleware writes. The reducers sum or append, so a
    Command update of {"neighbor_calls": 1} adds one — it never overwrites."""
    search_calls: Annotated[int, operator.add]
    neighbor_calls: Annotated[int, operator.add]
    table_calls: Annotated[int, operator.add]
    expansion_tokens: Annotated[int, operator.add]
    captured_passages: Annotated[list[dict], operator.add]


# ── reading a tool result ───────────────────────────────────────────────

def _payload(result) -> dict | None:
    """The Lambda's JSON, whichever way the MCP adapter delivered it."""
    if not isinstance(result, ToolMessage):
        return None
    artifact = getattr(result, "artifact", None)
    if isinstance(artifact, dict) and isinstance(artifact.get("structured_content"), dict):
        return artifact["structured_content"]
    content = result.content
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content
                          if isinstance(b, dict) and b.get("type") == "text")
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _fence(text: str) -> str:
    """Passage text is authored by protocol sponsors, not by this project.
    Strip the fence tags so a document cannot close its own fence."""
    return str(text).replace("<untrusted_data>", "").replace("</untrusted_data>", "")


# ── the middleware ──────────────────────────────────────────────────────

class RetrievalMiddleware(AgentMiddleware):
    """Bounds and records the three retrieval tools. See module docstring."""

    state_schema = RetrievalState

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.limits = {SEARCH: cfg.max_searches_per_turn,
                       NEIGHBORS: cfg.max_neighbor_calls,
                       TABLE: cfg.max_table_calls}
        self.counter = {SEARCH: "search_calls", NEIGHBORS: "neighbor_calls",
                        TABLE: "table_calls"}

    # sync and async share one implementation
    def wrap_tool_call(self, request, handler):
        gated = self._gate(request)
        if isinstance(gated, ToolMessage):
            return gated
        return self._record(gated, handler(gated))

    async def awrap_tool_call(self, request, handler):
        gated = self._gate(request)
        if isinstance(gated, ToolMessage):
            return gated
        return self._record(gated, await handler(gated))

    # ── BEFORE ──────────────────────────────────────────────────────────
    def _gate(self, request):
        kind = _kind(request.tool_call["name"])
        if kind is None:
            return request
        state = request.state or {}
        call_id = request.tool_call["id"]

        # STEP 1 — recursion budget for this tool
        used_calls = state.get(self.counter[kind], 0)
        if used_calls >= self.limits[kind]:
            return ToolMessage(
                tool_call_id=call_id,
                content=f"REFUSED: {kind} call limit reached "
                        f"({used_calls}/{self.limits[kind]} this question). "
                        "Answer from the passages you have and state in `note` "
                        "what is missing.")
        if kind == SEARCH:
            return request

        # STEP 2 — shared token budget for expansions
        remaining = self.cfg.expansion_token_budget - state.get("expansion_tokens", 0)
        if remaining <= 0:
            return ToolMessage(
                tool_call_id=call_id,
                content=f"REFUSED: expansion token budget exhausted "
                        f"({self.cfg.expansion_token_budget} tokens). Answer from "
                        "what you have and state in `note` what is missing.")

        # STEP 3 — rewrite the arguments: the Lambda gets OUR numbers
        args = dict(request.tool_call["args"])
        args["max_tokens"] = remaining
        args["exclude_ids"] = sorted({p["chunk_id"] for p in state.get("captured_passages", [])})
        if kind == NEIGHBORS:
            args["window"] = max(1, min(int(args.get("window") or 2), self.cfg.max_window))
        return request.override(tool_call={**request.tool_call, "args": args})

    # ── AFTER ───────────────────────────────────────────────────────────
    def _record(self, request, result):
        kind = _kind(request.tool_call["name"])
        if kind is None or isinstance(result, Command):
            return result
        call_id = request.tool_call["id"]
        payload = _payload(result)

        # A failed call still counts, so an erroring tool cannot loop forever.
        if payload is None or payload.get("error"):
            detail = (payload or {}).get("detail") or _fence(getattr(result, "content", ""))
            return Command(update={
                self.counter[kind]: 1,
                "messages": [ToolMessage(tool_call_id=call_id,
                                         content=f"ERROR from {kind}: {detail}")]})

        passages = payload.get("passages", [])
        tokens = int(payload.get("tokens_used", 0)) if kind != SEARCH else 0
        state = request.state or {}
        stats = {k: state.get(k, 0) for k in
                 ("search_calls", "neighbor_calls", "table_calls", "expansion_tokens")}
        stats[self.counter[kind]] += 1
        stats["expansion_tokens"] += tokens

        return Command(update={
            self.counter[kind]: 1,
            "expansion_tokens": tokens,
            "captured_passages": passages,
            "messages": [ToolMessage(tool_call_id=call_id,
                                     content=self._view(kind, payload, passages, stats))]})

    # ── what the model reads ────────────────────────────────────────────
    def _view(self, kind, payload, passages, stats) -> str:
        c = self.cfg
        lines = [f"{kind}: {len(passages)} passage(s)"]
        if kind == NEIGHBORS:
            lines.append(f"window={payload.get('window')} around position "
                         f"{payload.get('seed_position')}, stopped by: "
                         f"{payload.get('stopped_by')}")
        if kind == TABLE:
            lines.append(f"table has {payload.get('n_fragments')} fragment(s), "
                         f"stopped by: {payload.get('stopped_by')}")
        if not passages:
            lines.append("Nothing new was returned." if kind != SEARCH else
                         "No passage matched. The corpus may not cover this — "
                         "do not invent an answer.")

        lines.append("<untrusted_data>")
        for p in passages:
            head = (f"[{p['chunk_id']}] doc={p['doc_id']} pos={p.get('position')} "
                    f"p{p.get('page')} type={p.get('content_type')}")
            if p.get("score") is not None:
                head += f" score={p['score']:.3f}"
            if p.get("content_type") == "table_summary":
                head += (f" n_fragments={p.get('n_fragments')} "
                         "-> expand_table(chunk_id) returns the exact rows")
            lines.append(head)
            lines.append(f"headings: {p.get('headings')}")
            lines.append(_fence(p.get("text", "")))
            lines.append("")
        lines.append("</untrusted_data>")

        lines.append(
            f"budget used: search {stats['search_calls']}/{c.max_searches_per_turn}, "
            f"neighbors {stats['neighbor_calls']}/{c.max_neighbor_calls}, "
            f"table {stats['table_calls']}/{c.max_table_calls}, "
            f"expansion tokens {stats['expansion_tokens']}/{c.expansion_token_budget}")
        return "\n".join(lines)


# ── assembly ────────────────────────────────────────────────────────────

def build_agent(tools: list, model=None):
    """`model` is injectable so the loop can be tested without Bedrock."""
    if model is None:
        from langchain_aws import ChatBedrockConverse
        model = ChatBedrockConverse(
            model=CFG.model_id,
            guardrail_config={"guardrailIdentifier": CFG.guardrail_id,
                              "guardrailVersion": CFG.guardrail_version})
    return create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT,
                        response_format=ToolStrategy(ModelDecision),
                        middleware=[RetrievalMiddleware()])


def assemble(result: dict) -> TrialSearchResponse:
    """TrialSearchResponse from the finished loop's STATE.

    STEP 1  dedupe by chunk_id — the first capture wins, so a search hit
            keeps its score even if a later expansion also returned it
    STEP 2  order by document, then reading position
    STEP 3  choose result_shape from what actually happened
    """
    decision: ModelDecision = result["structured_response"]
    stats = RetrievalStats(
        search_calls=result.get("search_calls", 0),
        neighbor_calls=result.get("neighbor_calls", 0),
        table_calls=result.get("table_calls", 0),
        expansion_tokens=result.get("expansion_tokens", 0),
        expansion_token_budget=CFG.expansion_token_budget)
    usage = collect_usage(result.get("messages"), CFG.model_id)

    unique: dict[str, dict] = {}
    for p in result.get("captured_passages", []):
        unique.setdefault(p["chunk_id"], p)
    passages = sorted((Passage(**p) for p in unique.values()),
                      key=lambda p: (p.doc_id, p.position if p.position is not None else 0))

    common = dict(stats=stats, usage=usage, entities=decision.entities,
                  result_note=decision.note)
    if stats.search_calls == 0:
        return TrialSearchResponse(result_shape="not_executed", **{
            **common, "result_note": "No search was executed."})
    if not decision.answerable:
        return TrialSearchResponse(result_shape="unanswerable", **common)
    if not passages:
        return TrialSearchResponse(result_shape="empty", **common)
    return TrialSearchResponse(result_shape="passages", passages=passages, **common)


async def orchestrate(question: str) -> TrialSearchResponse:
    async with connect_tools() as tools:
        agent = build_agent(tools)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    return assemble(result)
