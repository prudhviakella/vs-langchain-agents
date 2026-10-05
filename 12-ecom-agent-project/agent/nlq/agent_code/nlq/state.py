"""LangGraph state for one NLQ question.

    every COUNTER is Annotated[int, operator.add]: middleware returns +1 deltas and
    the reducer SUMS them — never "last write wins" (ACT's NLQ hit
    INVALID_CONCURRENT_GRAPH_UPDATE and lost counts until counters had reducers).

WHAT THIS DOES NOT DO
    Counters do not live on middleware objects: the agent is built ONCE and
    shared by every request, so an instance attribute would leak one
    question's budget into the next.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain.agents.middleware import AgentState


def _last(_old, new):
    return new


class NLQState(AgentState):
    grounding_context: Annotated[str, _last]
    grounding_ids: Annotated[list[str], _last]
    executed_sql: Annotated[str, _last]
    captured: Annotated[dict[str, Any], _last]
    tool_calls: Annotated[int, operator.add]
    queries: Annotated[int, operator.add]
    repair_count: Annotated[int, operator.add]
    queries_since_result: Annotated[int, operator.add]
    has_result: Annotated[bool, _last]
