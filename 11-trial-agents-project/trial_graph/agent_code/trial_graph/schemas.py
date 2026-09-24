"""Output contract for the Trial Graph agent.

Adapted from ACT Xerebro's NLC schemas.py — GraphNode, GraphRelationship,
TokenUsage, and collect_usage are unchanged, since none of them reference
anything fraud-graph-specific. Only ModelDecision and the final response
model are adapted to this domain.

Trial Graph is a pure DATA PRODUCER, same as NLC: it returns graph-shaped
data based on the Cypher it wrote, plus a result_shape hint the Supervisor
uses for routing. It does NOT write a summary — the Supervisor owns that.

Bulk data (nodes/relationships) is captured server-side by the execute
tool and assembled here programmatically; the LLM never re-transcribes it.
Same principle as NLC, same reason: a model that retypes a result can
retype it wrong, and a wrong retype compounds if it feeds a follow-up
query.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    element_id: str
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    element_id: str
    type: str
    start: str = Field(description="start node element_id")
    end: str = Field(description="end node element_id")
    properties: dict[str, Any] = Field(default_factory=dict)


class ModelDecision(BaseModel):
    """The ONLY thing the LLM emits as structured output — small, no bulk
    data, and NO cypher. The model does not report the query: the executed
    query is recorded by the execute_cypher tool itself and filled in
    programmatically. This makes it structurally impossible to "answer"
    without running the query — every field here is a judgment the model
    can only make AFTER seeing a real result from execute_cypher.
    """
    entities: list[str] = Field(
        default_factory=list,
        description="Business entities the answer surfaces (ids/names) — "
                    "trials, sponsors, drugs, diseases, sites. Feeds the "
                    "entity list the analyst sees alongside the answer.",
    )
    answerable: bool = Field(
        default=True,
        description="False if the question cannot be grounded in the graph "
                    "schema (e.g. asking about data this graph does not "
                    "model — adverse events, dosing schedules, anything not "
                    "in the Document/Section/Chunk/Trial/registry layers). "
                    "When False, return no query result and explain in note.",
    )
    note: str = Field(
        default="",
        description="Brief FACTUAL note only (not a summary): why "
                    "unanswerable, or an interpretation made (e.g. 'read "
                    "this trial as the most recently updated phase'). The "
                    "Supervisor writes the user-facing summary.",
    )


class TokenUsage(BaseModel):
    """Token metering for this agent's handling of one request, summed
    across every LLM call in the tool loop. Rides the response so the
    Supervisor can aggregate usage across specialists. Cost is
    deliberately NOT computed here: store tokens+model_id, derive cost
    from a price table elsewhere — prices change, models vary.
    """
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    model_id: str = ""


class TrialGraphResponse(BaseModel):
    """Final structured result returned to the Supervisor."""
    result_shape: Literal["graph", "table", "empty", "unanswerable", "not_executed"]

    # graph-shaped result -> Supervisor routes to a graph-rendering agent
    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)

    # tabular result -> Supervisor routes to a chart-rendering agent
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)

    # metadata (audit + routing + honest summarization by the Supervisor)
    cypher: str = ""
    entities: list[str] = Field(default_factory=list)
    grounding_record_ids: list[str] = Field(default_factory=list)
    result_note: str = Field(
        default="",
        description="Factual signal for the Supervisor: row/node count, "
                    "truncation flag, or why the question was unanswerable.",
    )
    usage: TokenUsage = Field(default_factory=TokenUsage)


def collect_usage(messages, model_id: str) -> TokenUsage:
    """Sum token usage across every LLM call in the loop.

    Uses langchain_core's own add_usage rather than adding fields by hand —
    it also sums input_token_details (cache reads) and output_token_details
    (reasoning tokens), which a manual sum silently drops. Cached input is
    roughly an order of magnitude cheaper, so ignoring it reports a cost
    that is simply wrong.
    """
    from langchain_core.messages.ai import add_usage

    total = None
    calls = 0
    for message in messages or []:
        meta = getattr(message, "usage_metadata", None)
        if meta is None and isinstance(message, dict):
            meta = message.get("usage_metadata")
        if not meta:
            continue
        total = add_usage(total, meta)
        calls += 1

    if not total:
        return TokenUsage(model_id=model_id)

    usage = TokenUsage(
        model_id=model_id,
        input_tokens=int(total.get("input_tokens", 0) or 0),
        output_tokens=int(total.get("output_tokens", 0) or 0),
        total_tokens=int(total.get("total_tokens", 0) or 0),
        llm_calls=calls,
    )
    if usage.total_tokens == 0 and (usage.input_tokens or usage.output_tokens):
        usage.total_tokens = usage.input_tokens + usage.output_tokens
    cached = (total.get("input_token_details") or {}).get("cache_read")
    if cached and hasattr(usage, "cached_input_tokens"):
        usage.cached_input_tokens = int(cached)
    return usage
