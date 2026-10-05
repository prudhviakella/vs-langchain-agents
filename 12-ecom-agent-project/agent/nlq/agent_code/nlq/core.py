"""NLQ core — one question in, a stream of progress events out, ending with the result.

    question + history
      ─► GroundingMiddleware   Pinecone: schema, joins, terms, rules, examples -> prompt
      ─► model (gpt-6-sol)     writes SQL; tools: lookup_schema / validate_sql / execute_sql
      ─► GuardMiddleware       read-only + LIMIT, before every query
      ─► BudgetMiddleware      calls / queries / failures / scope after a result
      ─► ModelDecision         judgements only (ToolStrategy: structured output as a tool call)
      ─► assemble()            NLQResponse from STATE — rows and SQL the tools captured

    events:  {"type":"progress","agent":"nlq","phase": grounding|lookup|validating|executing|result, …}
             {"type":"progress","agent":"nlq","phase":"done","result": NLQResponse}

FROM ACT
    ToolStrategy explicitly: a provider-native JSON mode can make tool calls
    impossible, producing a decision with no query ever run.
    repair_count is SUMMED across the stream's per-node deltas, never overwritten.

WHAT THIS DOES NOT DO
    It keeps no conversation: history arrives with each call (the backend owns it).
"""
from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from . import progress as P
from .assemble import assemble
from .config import CFG, api_key
from .middleware import BudgetMiddleware, GroundingMiddleware, GuardMiddleware
from .prompt import SYSTEM_PROMPT
from .schemas import ModelDecision, collect_usage
from .state import NLQState
from .tools import TOOLS

log = logging.getLogger("nlq.core")
_AGENT = None


def chat_model():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CFG.model, use_responses_api=True,
                      api_key=api_key("OPENAI_API_KEY", "OPENAI_SECRET_ID"), max_retries=4)


def build(model=None):
    return create_agent(
        model=model or chat_model(), tools=TOOLS, system_prompt=SYSTEM_PROMPT, state_schema=NLQState,
        response_format=ToolStrategy(schema=ModelDecision),
        middleware=[BudgetMiddleware(), GuardMiddleware(), GroundingMiddleware()], name="nlq")


def agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build()
    return _AGENT


async def answer_stream(question: str, history: list[dict] | None = None, graph=None):
    messages = [{"role": m.get("role", "user"), "content": m.get("text") or m.get("content") or ""}
                for m in (history or [])] + [{"role": "user", "content": question}]
    decision, audit, llm_messages = None, {"repair_count": 0}, []
    async for chunk in (graph or agent()).astream({"messages": messages}, stream_mode=["custom", "updates"],
                                                  version="v2", config={"recursion_limit": CFG.recursion_limit}):
        mode, data = chunk["type"], chunk["data"]
        if mode == "custom":
            yield data
            continue
        for _node, update in (data or {}).items():
            if not isinstance(update, dict):
                continue
            llm_messages.extend(update.get("messages") or [])
            if update.get("structured_response") is not None:
                decision = update["structured_response"]
            if update.get("repair_count"):
                audit["repair_count"] += update["repair_count"]        # SUM the deltas
            for key in ("executed_sql", "captured", "grounding_ids"):
                if update.get(key) is not None:
                    audit[key] = update[key]
    if decision is None:
        raise RuntimeError("the NLQ agent produced no decision")
    resp = assemble(decision, audit.get("captured"), audit.get("executed_sql", ""),
                    audit.get("grounding_ids") or [], audit["repair_count"])
    resp.usage = collect_usage(llm_messages, CFG.model)
    log.info("[nlq] shape=%s rows=%d repairs=%d sql=%s", resp.result_shape, resp.row_count,
             resp.repair_count, resp.sql[:200].replace("\n", " "))
    yield {"type": "progress", "agent": "nlq", "phase": P.DONE, "result": resp.model_dump()}
