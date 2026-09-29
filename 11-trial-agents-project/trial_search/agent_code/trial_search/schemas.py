"""Output contract for the trial_search agent.

The important design principle is that the LLM does NOT own the retrieved
data.

The flow is:

    ┌──────────────────────────────┐
    │           MODEL              │
    │                              │
    │       ModelDecision          │
    │                              │
    │  - entities                  │
    │  - answerable                │
    │  - note                      │
    │                              │
    │  Small judgment only         │
    └──────────────┬───────────────┘
                   │
                   │
    ┌──────────────▼───────────────┐
    │            TOOLS             │
    │                              │
    │      captured_passages       │
    │                              │
    │  Actual evidence returned    │
    │  by retrieval tools          │
    └──────────────┬───────────────┘
                   │
                   ▼
          orchestrate()
                   │
                   ▼
    ┌──────────────────────────────┐
    │     TrialSearchResponse      │
    │                              │
    │  Built from actual STATE    │
    │  + ModelDecision             │
    └──────────────────────────────┘


WHY THIS SEPARATION MATTERS
----------------------------

The model is allowed to make a judgment such as:

    "The corpus does not contain enough information."

or:

    "The answer is grounded in trial ABC."

But the model does NOT get to manufacture the actual passages.

Every passage included in TrialSearchResponse must originate from a
retrieval tool and be captured in agent state.

This creates a strong grounding boundary:

    Model
      │
      │ decision
      ▼
    State
      │
      │ actual tool output
      ▼
    Final response


PASSAGE ORIGIN
--------------

A passage has the same structure regardless of which retrieval tool
produced it.

`origin` tells us how the passage entered the system:

    search
        Initial semantic-search result.

    neighbor
        Additional context retrieved around an existing chunk.

    table
        Actual table content retrieved from a table summary.

`score` is only populated for semantic-search results.

Why?

A semantic-search result was ranked by the vector search system and
therefore has a similarity score.

An expanded chunk was explicitly fetched by `chunk_id`.

It was not independently ranked, so there is no similarity score to
report.


WHAT THIS MODULE DOES NOT DO
----------------------------

ModelDecision does NOT contain:

    - passage text
    - chunk IDs
    - retrieved evidence

Therefore the model cannot simply claim:

    "I found this passage..."

and inject arbitrary text into the final response.

The final passages always come from the retrieval tools' actual
return values.
"""


# ===========================================================================
# Imports
# ===========================================================================

from __future__ import annotations

# Literal is used to restrict fields such as `origin` and `result_shape`
# to a fixed set of allowed values.
from typing import Literal

# Pydantic provides runtime validation and serialization for all response
# objects passed between the specialist agent and the Supervisor.
from pydantic import (
    BaseModel,
    Field,
)


# ===========================================================================
# MODEL DECISION
# ===========================================================================

class ModelDecision(BaseModel):
    """The only structured output the model writes after retrieving.

    This is intentionally small.

    The model decides:

        - which entities/documents support the answer
        - whether the corpus contains enough information
        - a short factual note/caveat

    It does NOT return the retrieved passages themselves.
    """

    # Documents or trial identifiers that support the model's decision.
    #
    # These are references to the evidence, not the evidence itself.
    entities: list[str] = Field(
        default_factory=list,
        description=(
            "doc_ids or trial identifiers the answer is grounded in."
        ),
    )

    # Indicates whether the retrieved corpus contains enough relevant
    # information to answer the user's question.
    #
    # False means the model should explain the limitation in `note`.
    answerable: bool = Field(
        default=True,
        description=(
            "False if the corpus has no relevant passage. "
            "When False, explain in note."
        ),
    )

    # Short factual explanation or caveat.
    #
    # Examples:
    #
    #     "The eligibility list continues past the retrieval budget."
    #
    #     "The corpus does not contain the requested information."
    note: str = Field(
        default="",
        description=(
            "Brief factual note: why unanswerable, or a caveat "
            "(e.g. 'the eligibility list continues past the budget')."
        ),
    )


# ===========================================================================
# TOKEN USAGE
# ===========================================================================

class TokenUsage(BaseModel):
    """Aggregated LLM usage for the entire specialist-agent execution.

    A single user question can result in multiple model calls because the
    agent may perform several retrieval/tool steps.

    These fields allow the Supervisor and observability layer to understand
    the total LLM cost of the execution.
    """

    # Total input/prompt tokens sent to the model.
    input_tokens: int = 0

    # Portion of input tokens served from the model/provider cache.
    cached_input_tokens: int = 0

    # Tokens generated by the model.
    output_tokens: int = 0

    # Provider-reported total token usage.
    total_tokens: int = 0

    # Number of LLM calls that contributed usage metadata.
    llm_calls: int = 0

    # Model identifier used for the calls.
    model_id: str = ""


# ===========================================================================
# RETRIEVED PASSAGE
# ===========================================================================

class Passage(BaseModel):
    """One piece of evidence retrieved from the clinical-trial corpus.

    This is the canonical passage shape used regardless of which retrieval
    tool produced it.
    """

    # Unique identifier of the chunk in the document corpus.
    chunk_id: str

    # Document/trial identifier containing the chunk.
    doc_id: str

    # Actual retrieved text.
    #
    # This is tool-owned evidence and is NOT generated by ModelDecision.
    text: str

    # How the passage was obtained.
    #
    # search:
    #     Initial semantic-search result.
    #
    # neighbor:
    #     Chunk retrieved around another chunk.
    #
    # table:
    #     Actual table content retrieved from a table summary.
    origin: Literal[
        "search",
        "neighbor",
        "table",
    ]

    # Vector similarity score returned by the semantic-search system.
    #
    # Expanded chunks do not have a similarity score because they were
    # explicitly fetched by ID rather than independently ranked.
    score: float | None = None

    # Cohere reranking score.
    #
    # This is only populated when the search pipeline performs reranking.
    #
    # IMPORTANT:
    # Pydantic only preserves fields declared in the model.
    #
    # The Lambda returns this value, so it must be declared here.
    # Otherwise Pydantic would discard it before the Supervisor receives
    # the final TrialSearchResponse.
    rerank_score: float | None = None

    # Type of retrieved content.
    #
    # Examples:
    #
    #     paragraph
    #     table_summary
    #     heading
    #     etc.
    content_type: str = ""

    # Document heading hierarchy associated with the chunk.
    #
    # This helps the Supervisor understand the section/context where the
    # passage originated.
    headings: list[str] = Field(
        default_factory=list
    )

    # Source document page number, when available.
    page: int | None = None

    # Position of the chunk within the document.
    #
    # This is useful for restoring document order after retrieval.
    position: int | None = None

    # Identifier of the source table, when the passage represents table
    # content or a table summary.
    table_id: str = ""

    # Number of fragments represented by this passage.
    n_fragments: int | None = None

    # Approximate token count associated with the passage.
    n_tokens: int | None = None


# ===========================================================================
# RETRIEVAL STATISTICS
# ===========================================================================

class RetrievalStats(BaseModel):
    """Actual retrieval resources consumed during the agent execution.

    These values are recorded by the middleware rather than being reported
    by the model.

    This distinction is important:

        MODEL:
            decides what retrieval action it wants.

        MIDDLEWARE:
            records and enforces what actually happened.
    """

    # Number of semantic-search calls actually executed.
    search_calls: int = 0

    # Number of neighbor-expansion calls actually executed.
    neighbor_calls: int = 0

    # Number of table-expansion calls actually executed.
    table_calls: int = 0

    # Total expansion tokens consumed by neighbor/table expansion.
    expansion_tokens: int = 0

    # Maximum token budget available for expansion.
    expansion_token_budget: int = 0


# ===========================================================================
# SEARCH QUERY METADATA
# ===========================================================================

class SearchQuery(BaseModel):
    """Metadata describing one semantic_search execution.

    This information is captured by the middleware from the actual tool
    invocation and result.

    The model does NOT construct this object itself.

    Therefore this represents what actually happened rather than what the
    model claims happened.
    """

    # Exact semantic query sent to the retrieval system.
    query: str

    # Optional document restriction.
    #
    # None means the search was not restricted to a specific document.
    doc_id: str | None = None

    # Optional content-type restriction.
    content_type: str | None = None

    # Requested number of top results.
    top_k: int | None = None

    # Number of candidates initially returned by the vector-search layer.
    #
    # This represents the recall pool before filtering/reranking.
    candidates: int = 0

    # Number of passages retained after the retrieval/reranking process.
    results: int = 0

    # Whether reranking actually occurred.
    #
    # None means the retrieval result did not specify this information.
    reranked: bool | None = None

    # Whether the search call completed successfully.
    #
    # Failed calls are still recorded so the Supervisor/observability
    # layer has an accurate picture of what happened.
    succeeded: bool = True


# ===========================================================================
# FINAL TRIAL SEARCH RESPONSE
# ===========================================================================

class TrialSearchResponse(BaseModel):
    """Final response returned by trial_search to the Supervisor.

    `result_shape` communicates the high-level outcome of retrieval.

    Possible states:

        passages
            At least one passage was retrieved.

        empty
            A search was performed, but nothing matched.

        unanswerable
            Evidence may have been retrieved, but the model determined
            that the corpus does not contain enough information to answer
            the question.

        not_executed
            No search happened at all. This generally represents a hollow
            or incomplete decision.
    """

    # High-level outcome of the retrieval operation.
    result_shape: Literal[
        "passages",
        "empty",
        "unanswerable",
        "not_executed",
    ]

    # Actual evidence retrieved from tools.
    #
    # These passages come from state/tool outputs rather than from the
    # model's structured decision.
    passages: list[Passage] = Field(
        default_factory=list
    )

    # Trial/document identifiers selected by the model as grounding
    # references.
    entities: list[str] = Field(
        default_factory=list
    )

    # Additional factual explanation or caveat.
    #
    # This normally originates from ModelDecision.note.
    result_note: str = ""

    # Actual retrieval resource consumption.
    stats: RetrievalStats = Field(
        default_factory=RetrievalStats
    )

    # Every semantic search executed during this question, in execution
    # order.
    #
    # This provides visibility into:
    #
    #     - query
    #     - scope
    #     - top_k
    #     - candidate count
    #     - final result count
    #     - reranking
    #     - success/failure
    searches: list[SearchQuery] = Field(
        default_factory=list,
        description=(
            "Every search that ran, in order: "
            "the query, its scope, what came back."
        ),
    )

    # Aggregated LLM token usage across the complete agent execution.
    usage: TokenUsage = Field(
        default_factory=TokenUsage
    )


# ===========================================================================
# TOKEN-USAGE AGGREGATION
# ===========================================================================

def collect_usage(
    messages,
    model_id: str,
) -> TokenUsage:
    """Sum usage across every LLM call.

    An agent may call the model multiple times during a single user request.

    For example:

        Question
           │
           ▼
        LLM call 1
           │
           ▼
        semantic_search
           │
           ▼
        LLM call 2
           │
           ▼
        expand_neighbors
           │
           ▼
        LLM call 3
           │
           ▼
        final decision

    Therefore usage must be aggregated across all messages rather than
    reading usage from only the final model message.

    `add_usage` is used instead of manually summing fields because it also
    correctly handles provider-specific nested usage information such as
    cached input tokens.
    """

    # Import only when this function is called.
    #
    # add_usage knows how to merge LangChain usage metadata structures.
    from langchain_core.messages.ai import add_usage


    # `total` accumulates token usage across all LLM messages.
    #
    # It starts as None because add_usage() supports building the
    # aggregate incrementally.
    total = None

    # Count how many messages actually contained usage metadata.
    calls = 0


    # Iterate through every message produced during the agent execution.
    for message in messages or []:

        # Normal LangChain messages expose usage_metadata as an attribute.
        meta = getattr(
            message,
            "usage_metadata",
            None,
        )

        # Some execution paths may represent messages as dictionaries.
        #
        # Support both forms so usage collection is robust.
        if (
            meta is None
            and isinstance(
                message,
                dict,
            )
        ):
            meta = message.get(
                "usage_metadata"
            )


        # Only messages containing usage metadata represent an LLM call
        # that can contribute to the aggregate.
        if meta:

            # Merge this message's usage into the running total.
            #
            # add_usage() handles nested fields such as:
            #
            #     input_token_details.cache_read
            total = add_usage(
                total,
                meta,
            )

            # Count the call.
            calls += 1


    # If no message contained usage information, return a valid empty
    # TokenUsage object with the model identifier.
    if not total:
        return TokenUsage(
            model_id=model_id
        )


    # Build the normalized TokenUsage response from the aggregated
    # provider/LangChain usage structure.
    usage = TokenUsage(
        model_id=model_id,

        # Number of LLM calls for which usage metadata was available.
        llm_calls=calls,

        # Total input tokens.
        input_tokens=int(
            total.get(
                "input_tokens",
                0,
            )
            or 0
        ),

        # Total output/generated tokens.
        output_tokens=int(
            total.get(
                "output_tokens",
                0,
            )
            or 0
        ),

        # Provider-reported total.
        total_tokens=int(
            total.get(
                "total_tokens",
                0,
            )
            or 0
        ),
    )


    # Some providers may not return total_tokens.
    #
    # In that case calculate it from input + output.
    if not usage.total_tokens:
        usage.total_tokens = (
            usage.input_tokens
            + usage.output_tokens
        )


    # Extract cached-input-token information from the nested usage
    # structure.
    #
    # Example:
    #
    #     input_token_details:
    #         cache_read: 1234
    cached = (
        total.get(
            "input_token_details"
        )
        or {}
    ).get(
        "cache_read"
    )


    # Store cached input tokens when the provider reported them.
    if cached:
        usage.cached_input_tokens = int(
            cached
        )


    # Return the complete normalized usage object.
    return usage