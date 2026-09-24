"""Output contract for the Supervisor.

Adapted from ACT Xerebro's own SupervisorDecision, simplified for a
two-agent system: no mode_assessment (the reference system's own
routing between several different "modes" of investigation doesn't
apply here — this Supervisor only ever routes between a graph question
and a search question), no chart_gen/graph_gen rendering fields (those
agents don't exist in this build yet).

Same structural guarantee as every other schema in this project: this
is the ONLY thing the LLM emits as structured output, it carries no
bulk data, and RequireAgentCallMiddleware (core.py) makes it
structurally impossible to answer without having actually called a
specialist — see that middleware's own docstring for the real
production incident this guards against in the reference system.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    """What the model decides after (not instead of) calling a specialist."""
    answerable: bool = Field(
        default=True,
        description="False only for a genuine dead end — neither "
                    "trial_graph nor trial_search could address the "
                    "question. Not for 'I haven't tried yet.'",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Trial/sponsor/drug/document identifiers the answer "
                    "is grounded in — empty when the answer is a "
                    "quantity or trend, not about particular entities.",
    )
    clarifying_question: str = Field(
        default="",
        description="Set only if ask_user_input was actually called this "
                    "turn (see core.py's own note on why the tool call, "
                    "not this field, is authoritative).",
    )
    note: str = Field(
        default="",
        description="Brief FACTUAL note: why unanswerable, or a caveat "
                    "worth surfacing. Not a summary — compose() writes "
                    "the analyst-facing prose separately.",
    )


class TokenUsage(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    model_id: str = ""


class AgentCall(BaseModel):
    """One record of a specialist call — audit trail, written by the
    tool that made the call, not by the model. Same "the model decides
    whether, the data decides what" principle as the reference
    Supervisor's own entity resolution.
    """
    agent_name: Literal["trial_graph", "trial_search"]
    question: str
    rationale: str
    result_shape: str
    succeeded: bool


class SupervisorResponse(BaseModel):
    """Final structured result — assembled from STATE (what the tools
    actually returned), never from what the model's own decision says.
    """
    calls: list[AgentCall] = Field(default_factory=list)
    decision: SupervisorDecision
    render_target: Literal["graph", "chart", "none"] = Field(
        default="none",
        description="Set by orchestrate()'s own render_decision node — "
                    "plain code inspecting each call's result_shape, not "
                    "a model judgment. 'graph' if any call returned a "
                    "graph-shaped result, 'chart' if any returned a "
                    "table (and no graph took priority), else 'none'. "
                    "This is the deterministic half of the pipeline: no "
                    "render tool exists for the model to forget to call.",
    )
    composed_answer: str = Field(
        default="",
        description="The analyst-facing prose, from a SEPARATE compose() "
                    "call — see core.py for why this can't be the same "
                    "call that produces the structured decision.",
    )
    usage: TokenUsage = Field(default_factory=TokenUsage)


def collect_usage(messages, model_id: str) -> TokenUsage:
    """Identical to trial_graph/trial_search's own collect_usage."""
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
