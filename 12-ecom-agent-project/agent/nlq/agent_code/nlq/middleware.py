"""Middleware for the NLQ agent — the policy the tools do not hold.

    outermost ─► BudgetMiddleware     tool calls · queries · failed queries · scope after a result
               ─► GuardMiddleware     guard() every sql argument; the tool runs the GUARDED sql
               ─► GroundingMiddleware once per question: schema into the system prompt

FROM ACT
    - Budgets cover ALL tools, not just execute_sql: ACT once burned a whole turn
      on uncounted schema lookups.
    - A failed query is answered with a message the model can act on; past the
      repair budget it is told to stop and say so — never loop to the recursion limit.
    - Scope discipline: once a complete result exists, only a small number of
      further queries — and a LIMIT 1 peek is not a "result" (ACT: a peek armed the
      guard and 18 real queries were refused).
    - Counters are deltas into state (reducers sum them), never instance attributes.

WHAT THIS DOES NOT DO
    It does not judge whether SQL answers the question; it bounds the loop and
    keeps every query read-only.
"""
from __future__ import annotations

import asyncio
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

from . import grounding as G
from . import progress as P
from .config import CFG
from .guard import GuardError, guard
from .state import NLQState


def _message(request, text: str) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=request.tool_call["id"])


def _with_update(result, update: dict):
    """Attach state deltas to whatever the tool returned."""
    if isinstance(result, Command):
        merged = dict(result.update or {})
        for k, v in update.items():
            merged[k] = v
        return Command(update=merged, goto=result.goto, graph=result.graph)
    return Command(update={"messages": [result], **update})


def _failed(result) -> bool:
    msgs = (result.update or {}).get("messages", []) if isinstance(result, Command) else [result]
    return any(str(getattr(m, "content", "")).startswith(("SQL_ERROR", "GUARD")) for m in msgs)


def _is_peek(sql: str) -> bool:
    return bool(re.search(r"\bLIMIT\s+1\s*;?\s*$", sql or "", re.I))


class GroundingMiddleware(AgentMiddleware):
    state_schema = NLQState

    async def abefore_agent(self, state, runtime):
        question = next((m.content for m in reversed(state["messages"]) if getattr(m, "type", "") == "human"), "")
        P.emit(P.GROUNDING, detail="finding the tables and columns this question needs")
        g = await asyncio.to_thread(G.retrieve, str(question))
        P.emit(P.GROUNDING, detail=f"schema ready: {', '.join(g.tables) or 'no tables matched'}",
               tables=g.tables, terms=g.terms, joins=g.joins, examples=g.examples)
        return {"grounding_context": g.context_block, "grounding_ids": g.record_ids}

    async def awrap_model_call(self, request, handler):
        block = request.state.get("grounding_context") or ""
        if block:
            base = request.system_message.content if request.system_message else ""
            request = request.override(system_message=SystemMessage(content=f"{base}\n\n{block}"))
        return await handler(request)


class GuardMiddleware(AgentMiddleware):
    state_schema = NLQState

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in ("execute_sql", "validate_sql"):
            return await handler(request)
        try:
            # row_cap + 1, the SAME bound the executor injects: the one extra row is how
            # truncation is detected. A LIMIT row_cap here would make the executor see
            # an outer LIMIT, add none of its own, and report a capped result as complete.
            checked = guard(request.tool_call["args"].get("sql", ""), CFG.row_cap + 1)
        except GuardError as exc:
            P.emit(P.ERROR, sql=request.tool_call["args"].get("sql", ""), detail=f"refused by the SQL guard: {exc}")
            return _message(request, f"GUARD: {exc}")
        request.tool_call["args"]["sql"] = checked.sql          # the tool runs what was checked
        return await handler(request)


class BudgetMiddleware(AgentMiddleware):
    state_schema = NLQState

    async def awrap_tool_call(self, request, handler):
        st, name = request.state, request.tool_call["name"]
        if (st.get("tool_calls") or 0) >= CFG.max_tool_calls:
            return _with_update(_message(request, f"BUDGET: {CFG.max_tool_calls} tool calls used. Stop "
                                         "calling tools and give your ModelDecision with what you have."),
                                {"tool_calls": 1})
        if name == "execute_sql":
            if (st.get("queries") or 0) >= CFG.max_queries:
                return _with_update(_message(request, f"BUDGET: {CFG.max_queries} queries run. Give your "
                                             "ModelDecision now from the results you have."), {"tool_calls": 1})
            if (st.get("repair_count") or 0) >= CFG.max_repairs:
                return _with_update(_message(request, f"BUDGET: {CFG.max_repairs} failed queries. Stop: set "
                                             "answerable=false and say in note what failed."), {"tool_calls": 1})
            if st.get("has_result") and (st.get("queries_since_result") or 0) >= CFG.queries_after_result:
                return _with_update(_message(request, "SCOPE: you already have a complete result. Finish with "
                                             "ModelDecision; list anything still missing in unmet_parts."),
                                    {"tool_calls": 1})
        result = await handler(request)
        update = {"tool_calls": 1}
        if name == "execute_sql":
            update["queries"] = 1
            if st.get("has_result"):
                update["queries_since_result"] = 1
            if _failed(result):
                update["repair_count"] = 1
            else:
                sql = request.tool_call["args"].get("sql", "")
                captured = (result.update or {}).get("captured") if isinstance(result, Command) else None
                if captured and captured.get("rows") and not _is_peek(sql):
                    update["has_result"] = True
        return _with_update(result, update)
