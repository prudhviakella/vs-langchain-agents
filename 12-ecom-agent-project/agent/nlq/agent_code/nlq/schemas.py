"""The NLQ contract — what the model may decide, and what the supervisor receives.

    model ──► ModelDecision   judgements only: NO sql field, NO rows field
    state ──► NLQResponse     columns + rows + the sql that RAN, assembled by code

From ACT: the model cannot "answer" without running a query, because there is
no field to put an answer in — the rows and the executed SQL come from what
execute_sql wrote to state. Nothing the model types can alter a number.

WHAT THIS DOES NOT DO
    No prose summary: NLQ is a data producer; the supervisor writes the answer.
    No try_graph_instead: there is no graph agent to route to here.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelDecision(BaseModel):
    """Your FINAL output, after execute_sql has returned a result (or after you
    decided the question cannot be answered from this schema)."""
    entities: list[str] = Field(default_factory=list, description=(
        "Business entities the result surfaces, as IDs with a label: 'customer 4817', "
        "'product 2297 (RAN-FIC-1296)', 'order 31877', 'supplier 10044'. Empty for pure aggregates."))
    answerable: bool = Field(default=True, description=(
        "False only when no table/column in the schema can answer the question "
        "(a metric that does not exist, data that is not collected). Explain in note."))
    note: str = Field(default="", description=(
        "A brief FACTUAL note, not a summary: an interpretation you made ('revenue = "
        "merchandise total, shipping excluded'), why the question is unanswerable, or a caveat."))
    unmet_parts: list[str] = Field(default_factory=list, description=(
        "Parts of a multi-part question this result does NOT cover, each as the "
        "sub-question left unanswered. Empty when the result covers the whole question."))


class TokenUsage(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    model_id: str = ""


class NLQResponse(BaseModel):
    """What the supervisor receives. Always tabular — chart_gen draws from it."""
    result_shape: Literal["table", "empty", "unanswerable", "not_executed"]
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    sql: str = ""
    entities: list[str] = Field(default_factory=list)
    unmet_parts: list[str] = Field(default_factory=list)
    grounding_record_ids: list[str] = Field(default_factory=list)
    result_note: str = ""
    repair_count: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)


def collect_usage(messages: list, model_id: str) -> TokenUsage:
    """Summed over every model call in the loop — billing-grade, rides the response."""
    u = TokenUsage(model_id=model_id)
    for m in messages:
        meta = getattr(m, "usage_metadata", None)
        if not meta:
            continue
        u.llm_calls += 1
        u.input_tokens += int(meta.get("input_tokens") or 0)
        u.output_tokens += int(meta.get("output_tokens") or 0)
        u.total_tokens += int(meta.get("total_tokens") or 0)
        u.cached_input_tokens += int((meta.get("input_token_details") or {}).get("cache_read") or 0)
    return u
