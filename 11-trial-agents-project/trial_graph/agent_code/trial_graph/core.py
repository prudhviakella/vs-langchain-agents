"""trial_graph core: the agent loop and the middleware that bounds it.

    orchestrate(question)
        │
        ├─ connect_tools()      SigV4-signed MCP session to the Gateway
        │                       -> find_entity_by_name, validate_cypher,
        │                          execute_cypher as LangChain tools
        │
        ├─ build_agent(tools)   create_agent + ToolStrategy(ModelDecision)
        │     │
        │     └─ CypherMiddleware       (one class, one state schema)
        │           BEFORE execute      reject writes, clamp/inject LIMIT
        │           AFTER  execute      rows -> state, summary -> model,
        │                               failures counted against a budget
        │
        └─ assemble             TrialGraphResponse from STATE

WHERE EVERYTHING COMES FROM

    model           OpenAI chat model; key and model name from Secrets Manager
    system prompt   Bedrock Prompt Management, at the version pinned in
                    Parameter Store
    limits          Parameter Store (row_cap, max_repairs, graph_node_cap)
    guardrail       Bedrock guardrail via ApplyGuardrail — guardrail.py
    All loaded once, at container start, by config.settings().

LOOP ENGINEERING — WHAT BOUNDS THIS LOOP

    writes        rejected before the query reaches the Gateway
    row cap       a LIMIT is injected when absent, and lowered when the
                  model asks for more than row_cap
    repair        a failed query is counted; past the budget the tool is
                  refused and the model must answer or say it cannot
    dedupe        none needed — one query's result replaces the previous

    These are argument rewrites and refusals, not prompt instructions.
    The model cannot exceed them by ignoring text.

FIVE FACTS VERIFIED AT RUNTIME, NOT ASSUMED

    1. Middleware state must be declared with `state_schema` (a subclass
       of AgentState). `state_schema_extra` — used by the previous
       version of this file — is not a LangChain attribute: it is
       ignored, and every key written through Command is silently
       dropped. Confirmed by running a real create_agent loop both ways.
    2. Under ainvoke, a middleware defining only wrap_tool_call raises
       NotImplementedError on the first tool call. Both the sync and
       async forms are defined here over one shared implementation.
    3. Gateway tool names may carry a target prefix
       ("trial-graph-tools___execute_cypher"). The previous version
       compared names with == , so every guard was bypassed and
       unbounded writes would have reached Neo4j. Matching is on suffix.
    4. langchain_mcp_adapters returns tool content as a LIST of blocks
       ([{"type": "text", "text": ...}]), plus an optional artifact dict
       holding structured_content. _payload() reads either.
    5. ToolStrategy, not a bare response_format schema. A bare schema
       lets Bedrock constrain the whole response to JSON, making a tool
       call structurally impossible — the model would emit a confident
       decision having run no query at all.

WHY THE MODEL NEVER SEES THE ROWS

A model that retypes a result can retype it wrong, and a wrong value
feeds the next query. The rows go to state; the model sees counts, the
column list, and at most three sample rows.

WHAT THIS DOES NOT DO

    - It does not write to Neo4j. The Lambda uses a read-only path and
      this middleware rejects write keywords before dispatch; neither
      alone is relied on.
    - It does not rank or re-order rows. They keep Neo4j's own order.
"""
from __future__ import annotations

import json
import logging
import operator
import re
from contextlib import asynccontextmanager
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .config import settings
from .guardrail import GuardrailBlocked, GuardrailMiddleware
from .schemas import ModelDecision, TrialGraphResponse, collect_usage
from .tools import summarize

log = logging.getLogger("agent.trial_graph.core")

RESOLVE, VALIDATE, EXECUTE = "find_entity_by_name", "validate_cypher", "execute_cypher"

# Cypher keywords that mutate. Matched as whole words: a trial named
# "Creating a Registry" must not look like CREATE.
_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s*\{[^}]*\bCREATE\b)\b",
    re.IGNORECASE)
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)


def _kind(tool_name: str) -> str | None:
    """Gateway names may carry a target prefix; match on the suffix."""
    return next((k for k in (RESOLVE, VALIDATE, EXECUTE) if tool_name.endswith(k)), None)


@asynccontextmanager
async def connect_tools():
    """SigV4-signed MCP session. aws_service must be "bedrock-agentcore"
    — the signing name the Gateway expects."""
    from mcp import ClientSession
    from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
    from langchain_mcp_adapters.tools import load_mcp_tools

    s = settings()
    async with aws_iam_streamablehttp_client(
            endpoint=s.gateway_url, aws_service="bedrock-agentcore",
            aws_region=s.region) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield await load_mcp_tools(session)


class CypherState(AgentState):
    """Keys the middleware writes. Reducers sum or replace; a Command
    update of {"repair_count": 1} adds one rather than overwriting."""
    repair_count: Annotated[int, operator.add]
    execute_calls: Annotated[int, operator.add]
    captured: dict
    executed_cypher: str


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


class CypherMiddleware(AgentMiddleware):
    """Guards and records the Cypher tools. See module docstring."""

    state_schema = CypherState

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or settings()

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
        if kind not in (VALIDATE, EXECUTE):
            return request
        state = request.state or {}
        call_id = request.tool_call["id"]
        args = dict(request.tool_call["args"])
        query = str(args.get("query", ""))

        # STEP 1 — repair budget, execute only
        if kind == EXECUTE:
            used = state.get("repair_count", 0)
            if used >= self.cfg.max_repairs:
                return ToolMessage(
                    tool_call_id=call_id,
                    content=f"REFUSED: {used} queries have already failed "
                            f"(limit {self.cfg.max_repairs}). Stop rewriting. Set "
                            "answerable=false and explain in `note` — an honest "
                            "gap is the correct outcome here.")

        # STEP 2 — reject writes before dispatch
        if found := _WRITE.search(query):
            return ToolMessage(
                tool_call_id=call_id,
                content=f"REJECTED: {found.group(0).upper()} is a write operation "
                        "and this graph is read-only. Rewrite as MATCH ... RETURN. "
                        "This does not count against your repair budget.")

        # STEP 3 — cap the row count: inject a LIMIT, or lower one too high
        if kind == EXECUTE:
            match = _LIMIT.search(query)
            if match is None:
                query = f"{query.rstrip().rstrip(';')} LIMIT {self.cfg.row_cap}"
            elif int(match.group(1)) > self.cfg.row_cap:
                query = _LIMIT.sub(f"LIMIT {self.cfg.row_cap}", query)
            args["query"] = query
            return request.override(tool_call={**request.tool_call, "args": args})
        return request

    # ── AFTER ───────────────────────────────────────────────────────────
    def _record(self, request, result):
        kind = _kind(request.tool_call["name"])
        if kind is None or isinstance(result, Command):
            return result
        call_id = request.tool_call["id"]
        payload = _payload(result)

        # A tool whose payload cannot be read is a failure, not a silent pass.
        if payload is None:
            return Command(update={
                "repair_count": 1 if kind == EXECUTE else 0,
                "messages": [ToolMessage(
                    tool_call_id=call_id,
                    content=f"ERROR from {kind}: unreadable response.")]})

        if payload.get("error"):
            detail = payload.get("detail", "unknown error")
            hint = " Rewrite the query." if kind == EXECUTE else ""
            return Command(update={
                "repair_count": 1 if kind == EXECUTE else 0,
                "messages": [ToolMessage(
                    tool_call_id=call_id,
                    content=f"{payload.get('error_class', 'Error')}: {detail}{hint}")]})

        if kind == RESOLVE:
            return ToolMessage(tool_call_id=call_id,
                               content=self._resolved(payload))
        if kind == VALIDATE:
            return ToolMessage(
                tool_call_id=call_id,
                content="VALID — the query parses and every label and property "
                        "in it exists." if payload.get("valid")
                        else f"INVALID: {payload.get('error', 'unknown')}")

        # execute: rows to state, summary to the model
        return Command(update={
            "execute_calls": 1,
            "captured": payload,
            "executed_cypher": request.tool_call["args"].get("query", ""),
            "messages": [ToolMessage(tool_call_id=call_id,
                                     content=summarize(payload, self.cfg))]})

    def _resolved(self, payload: dict) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            return ("NO_MATCH: no entity matches that name. It is not in this "
                    "graph under that spelling. Do NOT retry with CONTAINS or a "
                    "wildcard — report the gap honestly in `note`.")
        # Keys are never truncated: they exist to be copied verbatim into the
        # next query. A silently shortened id matches nothing and reads as
        # "no connections exist".
        lines = [f"{len(candidates)} candidate(s), best first:"]
        lines += [f"  (:{c['label']} {{{c['property']}: {json.dumps(c['value'])}}})  "
                  f"name={c['name']}  score={c['score']:.2f}" for c in candidates]
        lines.append("Anchor your MATCH on the pattern shown, exactly — that property "
                     "is the indexed identity. Do not filter on the raw name, and do "
                     "not use the node's `key` property.")
        return "\n".join(lines)


def build_agent(tools: list, model=None, cfg=None):
    """The agent loop. `model` and `cfg` are injectable so it runs in tests
    without AWS or OpenAI.

    Middleware order is deliberate: the guardrail is first, so its INPUT
    check runs before anything else and its OUTPUT check sees every model
    turn; CypherMiddleware then guards the tool calls themselves.
    """
    s = cfg or settings()
    return create_agent(
        name="trial_graph",
        model=model or s.chat_model(), tools=tools, system_prompt=s.system_prompt,
        response_format=ToolStrategy(ModelDecision),
        middleware=[GuardrailMiddleware(s.guardrail_id, s.guardrail_version,
                                        decision_tool="ModelDecision"),
                    CypherMiddleware(s)])


def assemble(result: dict, cfg=None) -> TrialGraphResponse:
    """TrialGraphResponse from the finished loop's STATE.

    STEP 1  no execute call at all -> not_executed, whatever the model said
    STEP 2  model declared it unanswerable -> unanswerable
    STEP 3  otherwise the shape the Lambda reported
    """
    s = cfg or settings()
    decision: ModelDecision = result["structured_response"]
    captured = result.get("captured") or {}
    usage = collect_usage(result.get("messages"), s.openai_model)
    common = dict(usage=usage, entities=decision.entities,
                  cypher=result.get("executed_cypher", ""))

    if not result.get("execute_calls"):
        return TrialGraphResponse(result_shape="not_executed",
                                  result_note="No query was executed.", **common)
    if not decision.answerable:
        return TrialGraphResponse(result_shape="unanswerable",
                                  result_note=decision.note, **common)

    shape = captured.get("result_shape", "empty")
    note = decision.note
    if captured.get("truncated"):
        note = (note + " " if note else "") + f"Result capped at {s.row_cap} rows."
    return TrialGraphResponse(
        result_shape=shape if shape in ("graph", "table", "empty") else "empty",
        nodes=captured.get("nodes", []), relationships=captured.get("relationships", []),
        columns=captured.get("columns", []), rows=captured.get("rows", []),
        result_note=note, **common)


async def orchestrate(question: str) -> TrialGraphResponse:
    """One question, one bounded loop. A guardrail intervention on the
    question or on anything the model writes ends the turn as unanswerable,
    carrying the guardrail's own message."""
    try:
        async with connect_tools() as tools:
            result = await build_agent(tools).ainvoke(
                {"messages": [{"role": "user", "content": question}]})
    except GuardrailBlocked as blocked:
        return TrialGraphResponse(result_shape="unanswerable",
                                  result_note=f"Blocked by guardrail ({blocked.source}): "
                                              f"{blocked.message}")
    return assemble(result)
