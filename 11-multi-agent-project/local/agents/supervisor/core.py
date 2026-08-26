"""The supervisor — plans, calls the specialists, renders, composes.

    question
      -> AvailableAgents      who exists, discovered this turn
      -> RequireAgentCall     no "answered" decision with zero calls
      -> AgentCallBudget      at most N specialist calls
      -> NoRepeatCall         no re-asking an agent that already answered
      -> model                ONE tool: call_agent (agents are DISCOVERED)
      -> artifacts land in STATE; the model sees only summaries
      -> Decision             what it found, NOT the data
      -> render()             charts, in CODE, after the loop
      -> compose()            the prose answer

Two boundaries hold this together, and both are structural rather than hoped-for.

**Data flows AROUND the model.** Specialists write results to state; the model
gets counts and three sample rows. It cannot retype a number it never saw.

**Rendering is code, not a tool.** If the result is a table, chart it. A model
that forgets a `render` tool call produces a silently chartless answer — so
there is no `render` tool to forget.

The composer is a separate call, deliberately. Structured output and
token-by-token streaming cannot both happen in one call, and the two jobs need
different data: the planner LOOPS, so anything it holds could be retyped into a
follow-up; the composer runs ONCE at the end and its whole job is to state the
finding. Give it only counts and it writes "the query returned 2 rows", which is
a status report, not an answer.
"""

from __future__ import annotations

import json
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from agents_lib import config, llm

from .context import RequestContext
from .middleware import (AgentCallBudget, AvailableAgents, NoRepeatCall,
                         RequireAgentCall)
from .prompt import COMPOSER_PROMPT, SYSTEM_PROMPT
from .state import SupervisorState
from .tools import build_tools


class Decision(BaseModel):
    """What the planner reports when it has finished.

    No `rows`, no `data`, no `answer` field carrying figures. If the affordance
    existed the model would eventually retype a value — and the whole point of
    keeping data in state is that it cannot.
    """
    finding: str = Field(description="What you established, in one or two "
                                     "sentences. Not the data itself.")
    answerable: bool = Field(description="False when no available agent can "
                                         "serve the question")
    note: str = Field(default="", description="Anything the reader should know: "
                                              "an assumption you made, a part "
                                              "that went unanswered")


def build_agent(registry):
    """Compile the planner.

    `response_format=ToolStrategy(schema=Decision)`, never the bare schema.

    Passing the schema alone lets create_agent choose a strategy, and for models
    advertising native structured output it picks the provider's constrained
    decoding. That forces the response to be schema-conforming JSON — and a
    response constrained to emit JSON cannot emit a tool call.

    The result is a model that answers on turn one having called nothing: every
    field filled with no data behind it, answerable true, a confident
    non-answer. It looks like a prompting problem and it is not.

    ToolStrategy makes the decision schema ANOTHER TOOL beside call_agent, so
    the model chooses between "call an agent" and "emit my decision" instead of
    being structurally forbidden from the first. Plan-execute-replan cannot work
    if the first model call may only produce a final answer.

    Middleware nests outside-in, and the order is load-bearing:
      AvailableAgents   first, so its block is present inside everything else —
                        including RequireAgentCall's corrective retry, which
                        would otherwise ask the model to pick an agent while
                        showing it none
      RequireAgentCall  next, wrapping the model call
      AgentCallBudget   then the tool wraps: budget outside repeat, so a refused
                        repeat does not consume a call
      NoRepeatCall      innermost
    """
    return create_agent(
        model=llm.chat_model(),
        tools=build_tools(),
        system_prompt=SYSTEM_PROMPT,
        state_schema=SupervisorState,
        context_schema=RequestContext,
        response_format=ToolStrategy(schema=Decision),
        middleware=[
            AvailableAgents(registry),
            RequireAgentCall(),
            AgentCallBudget(config.MAX_AGENT_CALLS),
            NoRepeatCall(),
        ],
        name="supervisor",
    )


async def render(captured: dict, calls: list[dict], ctx: RequestContext) -> list[dict]:
    """Charts, decided by SHAPE. Deterministic, and outside the loop.

    Every table above the minimum row count becomes a chart. There is no
    judgement here to get wrong and no tool call to forget.

    The renderer is still discovered — it can be down, and we need its endpoint —
    it is simply not part of the tool surface the model sees.
    """
    if "chart" not in ctx.registry.renderers:
        return []

    charts = []
    for record in calls:
        result = captured.get(record["call_id"])
        if not result or result.get("shape") != "table":
            continue
        # A two-row identifier lookup is not chartable data: noise to dismiss,
        # plus a wasted call and its latency on every such turn. The TABLE is
        # attached either way — only the chart is gated.
        if result.get("row_count", 0) < config.CHART_MIN_ROWS:
            ctx.note(f"chart skipped: {result['row_count']} row(s)")
            continue
        ctx.note("rendering a chart")
        try:
            chart = await ctx.invoke("chart", json.dumps(result),
                                     f"{record['call_id']}-chart")
            if chart.get("chart_type") != "none":
                charts.append(chart)
        except Exception as exc:
            # A failed chart must not fail the turn. The answer and the table
            # are the deliverable; the chart is an aid.
            ctx.note(f"chart failed: {str(exc)[:80]}")
    return charts


def compose(question: str, decision: Decision, calls: list[dict],
            captured: dict) -> str:
    """The prose answer. A second model call, at the end.

    WHAT THE COMPOSER SEES, and why it differs from the planner.

    The planner never sees data because it LOOPS: any value it held could be
    retyped into a follow-up question, and errors would compound.

    The composer runs ONCE and its job is to state the finding. Given only
    counts it can say "the query returned 2 rows" — a status report, not an
    answer. So it gets a BOUNDED, VERBATIM slice of the real result and is told
    to quote exactly and never compute.

    The safety properties still hold:
      bounded     a fixed top-N, never the whole result
      terminal    nothing downstream consumes this prose as data
      verifiable  the authoritative table renders next to it, so a discrepancy
                  is visible immediately
      inert       the composer cannot query, so a misreading cannot become a
                  wrong query
    """
    facts = [f"Question: {question}",
             f"Answerable: {decision.answerable}",
             f"Finding: {decision.finding}"]
    if decision.note:
        facts.append(f"Note: {decision.note}")

    for record in calls:
        result = captured.get(record["call_id"]) or {}
        facts.append(f"\n- {record['call_id']} ({record['agent']}) asked "
                     f"\"{record['question']}\" -> {record['shape']}")

        if result.get("shape") == "table":
            columns = result.get("columns", [])
            facts.append(f"  columns: {columns}")
            facts.append("  <untrusted_data>")
            for row in (result.get("rows") or [])[:config.COMPOSER_ROWS]:
                facts.append(f"    {dict(zip(columns, row))}")
            facts.append("  </untrusted_data>")
        elif result.get("shape") == "passages":
            facts.append("  <untrusted_data>")
            for passage in (result.get("passages") or [])[:config.COMPOSER_ROWS]:
                facts.append(f"    [p{passage.get('page')}] "
                             f"{str(passage.get('text', ''))[:400]}")
            facts.append("  </untrusted_data>")
        elif result.get("reason"):
            facts.append(f"  {result['reason']}")

    model = llm.chat_model()
    reply = model.invoke([{"role": "system", "content": COMPOSER_PROMPT},
                          {"role": "user", "content": "\n".join(facts)}])
    return reply.content if hasattr(reply, "content") else str(reply)
