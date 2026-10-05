"""Live progress — what the supervisor turns into status lines and Reasoning.

    tool / middleware ──emit(phase, …)──► LangGraph custom stream ──► main.py ──► A2A status update
        {"type": "progress", "agent": "nlq", "phase": "executing", "sql": "SELECT …", "detail": "…"}

The supervisor makes `sql` a REASONING line, persisted with the turn: the
query that answered the question stays readable after the turn ends.

WHAT THIS DOES NOT DO
    Outside a running graph (a unit test, a script) emit() is a no-op — progress
    must never be the reason a call fails.
"""
from __future__ import annotations

GROUNDING, LOOKUP, VALIDATING, EXECUTING, RESULT, DONE = (
    "grounding", "lookup", "validating", "executing", "result", "done")


def emit(phase: str, **fields) -> None:
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except Exception:
        return
    event = {"type": "progress", "agent": "nlq", "phase": phase}
    event.update({k: v for k, v in fields.items() if v is not None})
    try:
        writer(event)
    except Exception:
        pass
