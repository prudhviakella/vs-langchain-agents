"""trial_graph's core agent: LangGraph loop, middleware stack, and the
SigV4-signed MCP connection to the AgentCore Gateway.

    orchestrate(question)
        |
        v
    connect_tools()   SigV4-signed MCP session -> the Gateway -> Lambda
        |              (mcp-proxy-for-aws + langchain-mcp-adapters,
        |               NOT a direct in-process Neo4j call — see
        |               tools.py's own note on why this changed)
        v
    build_agent(tools)
        |
        |-- (schema lives directly in SYSTEM_PROMPT, not injected via
        |    middleware — this graph's schema is small and fixed, unlike
        |    NLC's dynamically-discovered agent list, so there is
        |    nothing per-turn to inject)
        |-- CypherGuardMiddleware      wraps execute_cypher: forces the
        |                              validate_cypher check, rejects
        |                              writes, injects LIMIT
        |-- ResultSummaryMiddleware    wraps execute_cypher: stashes the
        |                              full result in state, replaces
        |                              the model-visible message with a
        |                              compact summary — the actual
        |                              "bulk data never reaches the
        |                              model" mechanism, now living in
        |                              middleware instead of inside a
        |                              tool function
        |-- RepairBudgetMiddleware     caps failed-query retries; an
        |                              honest "unanswerable" beats an
        |                              endless repair loop
        |
        v
    model (ToolStrategy, not bare schema — see WHY below)
        |
        v
    TrialGraphResponse, assembled from STATE, never from what the
    model itself said

WHY ToolStrategy, NOT A BARE SCHEMA

Same lesson NLC and the Supervisor both learned the hard way in the
reference system: a bare response_format schema lets Bedrock's own
ProviderStrategy constrain the ENTIRE response to JSON, making a
toolUse block structurally impossible. That would mean the model could
emit a confident ModelDecision having called execute_cypher zero times.
ToolStrategy makes the decision schema another tool alongside the real
ones, so the model chooses between "call a tool" and "emit my decision"
rather than being blocked from the first. Applied here preemptively,
not because this agent has already reproduced the incident.

WHY THE MCP CONNECTION IS ESTABLISHED PER-INVOCATION, NOT ONCE AT IMPORT

A stdio or long-lived session could be cached across calls, but a
streamable-HTTP MCP session over SigV4 is billed and scoped like any
other signed AWS request — there is no meaningful "keep it open"
semantics the way there is for a database connection pool. Opening a
fresh session per question keeps the failure mode simple: a Gateway
outage fails this one question loudly, not every question until the
process restarts.
"""
from __future__ import annotations

import logging
import operator

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from mcp import ClientSession
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import CFG
from .prompt import SYSTEM_PROMPT
from .schemas import ModelDecision, TrialGraphResponse, collect_usage

log = logging.getLogger("agent.trial_graph.core")


async def connect_tools():
    """Async context manager: SigV4-signs a streamable-HTTP MCP session
    to the Gateway and loads its tools as LangChain BaseTool objects.

    aws_service="bedrock-agentcore" is the SigV4 signing name AgentCore
    Gateway expects — confirmed directly against mcp-proxy-for-aws's own
    documented usage, not "mcp" or any other plausible-looking guess.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _cm():
        async with aws_iam_streamablehttp_client(
            endpoint=CFG.gateway_url, aws_service="bedrock-agentcore",
            aws_region=CFG.aws_region,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                yield tools

    return _cm()


# ── Middleware ────────────────────────────────────────────────────────

class CypherGuardMiddleware(AgentMiddleware):
    """Wraps execute_cypher: rejects writes, injects a LIMIT if missing,
    and runs validate_cypher before letting the real query through.

    Adapted from NLC's own CypherGuardMiddleware — the read-only/LIMIT/
    EXPLAIN guarantee is domain-independent; only the write-keyword list
    is specific to Cypher itself, which is unchanged across both graphs.
    """
    _WRITE_KEYWORDS = ("CREATE", "MERGE", "DELETE", "SET", "REMOVE",
                      "DROP", "DETACH")

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "execute_cypher":
            return handler(request)

        query = request.tool_call["args"].get("query", "")
        upper = query.upper()
        for keyword in self._WRITE_KEYWORDS:
            if keyword in upper:
                return ToolMessage(
                    content=f"REJECTED: '{keyword}' is a write operation. "
                            "This tool is read-only. Rewrite as a MATCH/RETURN "
                            "query.",
                    tool_call_id=request.tool_call["id"])

        if " LIMIT " not in f" {upper} ":
            query = f"{query.rstrip(';')} LIMIT {CFG.row_cap}"
            request = request.override(
                tool_call={**request.tool_call,
                          "args": {**request.tool_call["args"], "query": query}})

        return handler(request)


class ResultSummaryMiddleware(AgentMiddleware):
    """Wraps execute_cypher: the actual "bulk data never reaches the
    model" mechanism. Stashes the full Lambda result into state under
    `captured`, and replaces the tool's own returned message with a
    compact summary — same guarantee NLC's execute_cypher tool made
    structurally by returning a Command directly; here it has to live
    in middleware instead, since execute_cypher is now a remote,
    MCP-loaded tool whose own implementation we do not control.
    """
    state_schema_extra = {"captured": (dict, {}), "executed_cypher": (str, "")}

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "execute_cypher":
            return handler(request)

        result = handler(request)
        if isinstance(result, Command):
            # A prior middleware (or a retry) already produced a Command —
            # pass it through unchanged rather than double-wrap it.
            return result

        import json
        try:
            captured = json.loads(result.content) if isinstance(result.content, str) \
                else result.content
        except (json.JSONDecodeError, TypeError):
            # The Lambda's own error path returns a plain string, not JSON —
            # let the model see it as-is so it can rewrite the query.
            return result

        if captured.get("error"):
            return ToolMessage(
                content=f"{captured.get('error_class', 'QueryError')}: "
                        f"{captured.get('detail', 'unknown error')}\n"
                        "Rewrite the query.",
                tool_call_id=request.tool_call["id"])

        query = request.tool_call["args"].get("query", "")
        return Command(update={
            "executed_cypher": query,
            "captured": captured,
            "messages": [ToolMessage(content=_summarize(captured),
                                     tool_call_id=request.tool_call["id"])],
        })


def _summarize(captured: dict) -> str:
    """The compact view — same guards as tools.py's own _summarize(),
    which this replaces at the point of actual use. See tools.py for
    the full reasoning behind each guard (ID truncation, column-cap
    consistency, the "0 relationships" confabulation warning) — kept
    there as the canonical version since it can be unit tested without
    any MCP/Gateway machinery in the loop; this file just needs to call
    it at the right point in the tool-call lifecycle.
    """
    from .tools import _summarize as _canonical_summarize
    return _canonical_summarize(captured, warnings=None)


class RepairBudgetMiddleware(AgentMiddleware):
    """Caps the number of failed-query repair attempts. An honest
    "unanswerable" after N failures beats an unbounded retry loop —
    same reasoning as NLC's own RepairBudgetMiddleware.
    """
    state_schema_extra = {"repair_count": (int, 0, operator.add)}

    def __init__(self, max_repairs: int = 3):
        super().__init__()
        self.max_repairs = max_repairs

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "execute_cypher":
            return handler(request)

        if request.state.get("repair_count", 0) >= self.max_repairs:
            return ToolMessage(
                content=f"REPAIR BUDGET EXHAUSTED ({self.max_repairs} failed "
                        "attempts). Stop trying to fix this query — set "
                        "answerable=false and explain what went wrong in "
                        "`note`. This is the honest outcome, not a failure "
                        "to hide.",
                tool_call_id=request.tool_call["id"])

        result = handler(request)
        is_failure = (isinstance(result, ToolMessage)
                     and ("REJECTED" in str(result.content)
                          or "Rewrite the query" in str(result.content)))
        if is_failure:
            return Command(update={"repair_count": 1, "messages": [result]})
        return result


# ── Assembly ──────────────────────────────────────────────────────────

def build_agent(tools: list):
    model = ChatBedrockConverse(
        model=CFG.model_id,
        guardrail_config={"guardrailIdentifier": CFG.guardrail_id,
                          "guardrailVersion": CFG.guardrail_version},
    )
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        # ToolStrategy, not a bare schema — see the module docstring.
        response_format=ToolStrategy(ModelDecision),
        middleware=[CypherGuardMiddleware(), ResultSummaryMiddleware(),
                   RepairBudgetMiddleware()],
    )


async def orchestrate(question: str) -> TrialGraphResponse:
    async with await connect_tools() as tools:
        agent = build_agent(tools)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    decision: ModelDecision = result["structured_response"]
    captured = result.get("captured", {})
    usage = collect_usage(result["messages"], CFG.model_id)

    if not decision.answerable:
        return TrialGraphResponse(result_shape="unanswerable",
                                  result_note=decision.note, usage=usage)

    if not result.get("executed_cypher"):
        # Same guarantee RequireAgentCallMiddleware enforces in the
        # Supervisor: a ModelDecision claiming an answer with no query
        # ever executed is a hollow decision, not a real one.
        return TrialGraphResponse(result_shape="not_executed",
                                  result_note="No query was executed.",
                                  usage=usage)

    shape = captured.get("result_shape", "empty")
    return TrialGraphResponse(
        result_shape=shape,
        nodes=captured.get("nodes", []),
        relationships=captured.get("relationships", []),
        columns=captured.get("columns", []),
        rows=captured.get("rows", []),
        cypher=result.get("executed_cypher", ""),
        entities=decision.entities,
        result_note=decision.note,
        usage=usage,
    )
