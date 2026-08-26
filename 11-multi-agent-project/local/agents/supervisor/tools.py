"""Supervisor tools.

ONE tool reaches the specialists:

    call_agent(agent_name, question, rationale, observation)

The agent name is data. Which agents exist comes from discovery and is rendered
into the prompt; there is no `@tool` per specialist to write and no second list
to drift.

What the tool does, and why:

  the ARTIFACT goes into STATE via `Command(update=...)`; the model gets a
  compact summary. Bulk data never flows through the model, so it can never
  retype a value it was shown.

  every database-origin value is CLEANED before it enters the model's context,
  and sample rows are fenced as untrusted data. The records came from documents
  and a database, not from us.

  a FAILED call still writes an audit record. A call that failed must be
  readable later, not vanish because it did not succeed.

RENDERERS ARE NOT HERE. Charting is not a routing decision: if the result is a
table, chart it. A model that forgets a `render` tool produces a silently
chartless answer — so there is no `render` tool to forget. It is an `if`, in
core.py, after the loop has finished.
"""

from __future__ import annotations

import json
import time
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agents_lib import config

from .context import RequestContext

MAX_VALUE_CHARS = 160


def _clean(value) -> str:
    """Neutralise one value from the data before it reaches the model.

    The text being summarised came out of PDFs and a database. A document title
    or a sponsor name is input from outside the system, so it is an injection
    surface rather than just a string — flattened to one line, fence markers
    removed, capped.
    """
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.replace("<untrusted_data>", "").replace("</untrusted_data>", "")
    text = " ".join(text.split())
    return text[:MAX_VALUE_CHARS] + "…" if len(text) > MAX_VALUE_CHARS else text


def _record(call_id: str, agent: str, question: str, raw: dict,
            elapsed: float) -> dict:
    """The audit row — what someone reads back later to see what happened."""
    return {"call_id": call_id, "agent": agent, "question": question,
            "shape": raw.get("shape", "error"),
            "row_count": raw.get("row_count", 0),
            "count": raw.get("count", 0),
            "elapsed_s": round(elapsed, 1),
            "note": _clean(raw.get("note", "") or raw.get("reason", ""))}


def _summarise(call_id: str, agent: str, raw: dict) -> str:
    """The compact view — all the planner ever sees of a result.

    Counts, column names and a few samples, so it can decide what to do next.
    Never the bulk.
    """
    shape = raw.get("shape", "error")

    if shape in ("empty", "error"):
        reason = _clean(raw.get("reason", "no reason given"))
        return (f"{call_id} ({agent}): {shape}. {reason}\n"
                "Do not retry the same question with the same agent. Ask a "
                "different agent, ask a narrower question, or report the gap "
                "honestly.")

    parts = [f"{call_id} ({agent}): {shape}"]

    if shape == "table":
        rows, columns = raw.get("rows") or [], raw.get("columns") or []
        parts.append(f"{len(rows)} rows x {len(columns)} columns")
        parts.append("columns: " + ", ".join(_clean(c) for c in columns[:12]))
        if rows:
            # Fenced, so the model can tell data from instructions. A row that
            # happens to read like a command is still just a row.
            parts.append("<untrusted_data>")
            for row in rows[:3]:
                parts.append("  sample: " + ", ".join(
                    f"{_clean(c)}={_clean(v)}"
                    for c, v in list(zip(columns, row))[:6]))
            parts.append("</untrusted_data>")
        if len(rows) > 3:
            parts.append(f"  ... and {len(rows) - 3} more rows (captured for "
                         "rendering — do NOT retype them)")

    elif shape == "passages":
        found = raw.get("passages") or []
        parts.append(f"{len(found)} passages")
        parts.append("<untrusted_data>")
        for passage in found[:3]:
            parts.append(f"  p{passage.get('page')}: "
                         f"{_clean(passage.get('text', ''))}")
        parts.append("</untrusted_data>")

    if raw.get("note"):
        parts.append(f"note: {_clean(raw['note'])}")
    return "\n".join(parts)


@tool
async def call_agent(agent_name: str, question: str,
                     rationale: str,
                     tool_call_id: Annotated[str, InjectedToolCallId],
                     runtime: ToolRuntime[RequestContext],
                     observation: str = "") -> Command:
    """Ask one of the available data agents a question, in plain English.

    Use the EXACT agent name from the AVAILABLE AGENTS section, chosen by
    matching the question to that agent's skills. Each agent is stateless and
    sees only what you send, so give it the full context it needs.

    rationale: REQUIRED. One or two sentences a non-technical reader could
    follow: what you are trying to establish with THIS call, why this agent
    rather than another, and what you expect back.

    It is a required ARGUMENT rather than something you say alongside the call,
    because a model bound to always call a tool frequently emits no text at all —
    so narration written as prose is often simply not produced. An argument
    always arrives.

    observation: what the PREVIOUS result told you and how it changed what you
    are doing now. Empty on your first call, filled on every one after.
    """
    ctx = runtime.context

    if agent_name not in ctx.registry.routable:
        available = ", ".join(ctx.registry.names()) or "none"
        return Command(update={"messages": [ToolMessage(
            content=f"UNKNOWN_AGENT '{agent_name}'. Available: {available}. "
                    "Use one of those exact names.",
            tool_call_id=tool_call_id)]})

    # Derived from tool_call_id, NOT from len(state["calls"]) + 1. Two calls
    # fired in the same step would both read the same length and compute the
    # same id — and a collision defeats the union reducer, because {**old,
    # **new} with the same key still overwrites. Unique by construction rather
    # than by hoping two calls never race.
    call_id = f"call_{tool_call_id[-8:]}"
    started = time.time()

    ctx.note(f"[{agent_name}] {rationale[:120]}")

    try:
        raw = await ctx.invoke(agent_name, question, call_id)
    except Exception as exc:
        # Converted to a ToolMessage rather than raised. Raising ends the whole
        # turn on something the model could have recovered from — and "the model
        # can replan" is only true if the error reaches it as a message.
        elapsed = time.time() - started
        return Command(update={
            "calls": [_record(call_id, agent_name, question,
                              {"reason": str(exc)}, elapsed)],
            "messages": [ToolMessage(
                content=f"{call_id} ({agent_name}): FAILED — {_clean(exc)}. "
                        "Do not retry the identical question with the same "
                        "agent. Try a different approach, or report honestly "
                        "that this could not be answered.",
                tool_call_id=tool_call_id)]})

    elapsed = time.time() - started
    return Command(update={
        "calls": [_record(call_id, agent_name, question, raw, elapsed)],
        # THIS CALL'S OWN entry only. Reading and merging state here would mean
        # two parallel calls each write back a merge of the same "before"
        # snapshot, and the second write silently discards the first call's
        # artifact — data the losing write never even contained. The reducer
        # does the union correctly BECAUSE each call emits only its delta.
        "captured": {call_id: raw},
        "messages": [ToolMessage(content=_summarise(call_id, agent_name, raw),
                                 tool_call_id=tool_call_id)]})


@tool
def ask_user(question: str, options: list[str],
             tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Ask the user for the ONE detail you need before you can act.

    Call this INSTEAD of call_agent when the question refers to something it
    never identifies — "that trial", "this drug", "them" — and nothing in the
    conversation says what it refers to.

    That is not a missing parameter. A missing date range has a sensible
    default; a missing SUBJECT does not. "Which trial?" has thousands of
    answers, and picking one is a guess, not a default — producing a confident
    answer about the wrong trial.

    Do NOT call this when the question names its subject, when history names it,
    or when the question is about a pattern rather than one thing ("which
    sponsors run several trials" needs no specific subject).
    """
    return Command(update={
        "clarify": {"question": question, "options": options[:4]},
        "messages": [ToolMessage(content="Asked the user for clarification.",
                                 tool_call_id=tool_call_id)]})


def build_tools() -> list:
    return [call_agent, ask_user]
