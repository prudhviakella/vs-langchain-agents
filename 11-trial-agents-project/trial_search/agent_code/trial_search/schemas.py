"""Output contract for the trial_search agent.

    model emits          ModelDecision        small judgment, no data
    tools write          captured_passages    in STATE, never retyped
    orchestrate builds   TrialSearchResponse  from STATE + decision

A passage has one shape whichever tool produced it. `origin` records
which: "search" (entry), "neighbor" (Case A), "table" (Case C). score is
set only for search hits — an expanded chunk was fetched by id, not
ranked, so it has no similarity score.

WHAT THIS DOES NOT DO

    ModelDecision does not carry passage text or chunk ids. The model
    cannot claim evidence it did not retrieve: every passage in the final
    response comes from a tool's own return value.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelDecision(BaseModel):
    """The only structured output the model writes, after retrieving."""
    entities: list[str] = Field(
        default_factory=list,
        description="doc_ids or trial identifiers the answer is grounded in.")
    answerable: bool = Field(
        default=True,
        description="False if the corpus has no relevant passage. When False, "
                    "explain in note.")
    note: str = Field(
        default="",
        description="Brief factual note: why unanswerable, or a caveat "
                    "(e.g. 'the eligibility list continues past the budget').")


class TokenUsage(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    model_id: str = ""


class Passage(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    origin: Literal["search", "neighbor", "table"]
    score: float | None = None
    content_type: str = ""
    headings: list[str] = Field(default_factory=list)
    page: int | None = None
    position: int | None = None
    table_id: str = ""
    n_fragments: int | None = None
    n_tokens: int | None = None


class RetrievalStats(BaseModel):
    """What the loop actually spent — for the Supervisor and for traces."""
    search_calls: int = 0
    neighbor_calls: int = 0
    table_calls: int = 0
    expansion_tokens: int = 0
    expansion_token_budget: int = 0


class TrialSearchResponse(BaseModel):
    """Returned to the Supervisor.

    result_shape:
        passages       at least one passage retrieved
        empty          searched, nothing matched
        unanswerable   the model judged the corpus does not cover it
        not_executed   no search ran at all — a hollow decision
    """
    result_shape: Literal["passages", "empty", "unanswerable", "not_executed"]
    passages: list[Passage] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    result_note: str = ""
    stats: RetrievalStats = Field(default_factory=RetrievalStats)
    usage: TokenUsage = Field(default_factory=TokenUsage)


def collect_usage(messages, model_id: str) -> TokenUsage:
    """Sum usage across every LLM call. add_usage also sums cache reads,
    which a manual field sum silently drops."""
    from langchain_core.messages.ai import add_usage

    total, calls = None, 0
    for message in messages or []:
        meta = getattr(message, "usage_metadata", None)
        if meta is None and isinstance(message, dict):
            meta = message.get("usage_metadata")
        if meta:
            total = add_usage(total, meta)
            calls += 1
    if not total:
        return TokenUsage(model_id=model_id)

    usage = TokenUsage(
        model_id=model_id, llm_calls=calls,
        input_tokens=int(total.get("input_tokens", 0) or 0),
        output_tokens=int(total.get("output_tokens", 0) or 0),
        total_tokens=int(total.get("total_tokens", 0) or 0))
    if not usage.total_tokens:
        usage.total_tokens = usage.input_tokens + usage.output_tokens
    cached = (total.get("input_token_details") or {}).get("cache_read")
    if cached:
        usage.cached_input_tokens = int(cached)
    return usage
