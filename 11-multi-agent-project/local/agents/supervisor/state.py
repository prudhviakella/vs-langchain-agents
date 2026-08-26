"""Supervisor state — where the data lives instead of in the model.

THE RULE: bulk data never flows through the model.

The specialists return tables and passages. Those go into STATE, written by the
tool with `Command(update=...)`, and are attached to the answer verbatim. The
model receives a compact summary of each call and decides what to ask next.

That is why `Decision` has no `rows` field and no `data` field. If the affordance
existed the model would eventually retype a value — and an answer that quotes a
mistyped NCT number is worse than one that says nothing, because it reads as
correct.

REDUCERS ARE PER-FIELD, AND THE RIGHT ONE DEPENDS ON MEANING

LangGraph merges each tool's state update through a reducer. Two tool calls can
land in the same step — a model can fire two independent specialist calls in
parallel — so a plain dict key raises INVALID_CONCURRENT_GRAPH_UPDATE.

The fix is a reducer, but which reducer is a question about semantics:

    calls      operator.add    a list that accumulates
    captured   union           each entry is a DIFFERENT call's artifact.
                               Last-write-wins would silently drop one, so a
                               "show me both" question returns half an answer
                               with nothing to indicate the rest existed.
    clarify    last-wins       only one question can be put to the user, so
                               keeping the latest is correct.

Same bug class, three different correct answers.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentState


def _merge_captured(old: dict | None, new: dict | None) -> dict:
    """Union, keyed by call id. Entries never collide, so nothing is lost."""
    return {**(old or {}), **(new or {})}


def _last(old: Any, new: Any) -> Any:
    """Keep the most recent. Right for a field that can only hold one thing."""
    return new if new is not None else old


class SupervisorState(AgentState):
    """State for one turn."""

    # Every specialist call: what was asked, of whom, what shape came back, how
    # long it took. This IS the audit record — a failed call belongs here too,
    # so a turn is readable months later rather than vanishing because it did
    # not succeed.
    calls: NotRequired[Annotated[list[dict], operator.add]]

    # The artifacts, keyed by call id. Attached to the answer verbatim, never
    # summarised by the model and never retyped.
    captured: NotRequired[Annotated[dict[str, Any], _merge_captured]]

    # Set by the ask_user tool when the question refers to something it never
    # identifies and history does not supply it.
    #
    # A tool rather than a field on the structured output, deliberately: the
    # structured output is produced AFTER the tool loop, so a model that wanted
    # to ask would already have called an agent by the time it could say so. The
    # choice has to exist where the model picks a tool, or it is not a choice.
    clarify: NotRequired[Annotated[dict[str, Any], _last]]
