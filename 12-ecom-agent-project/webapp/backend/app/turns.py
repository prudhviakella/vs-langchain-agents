"""One chat turn: supervisor stream in, the frontend's SSE events out, the answer saved.

    POST /chat/message ─► start() ─► background task run()
       STEP 1  history: the conversation's last messages (text only)
       STEP 2  save the user message + an assistant placeholder (status=streaming)
       STEP 3  meta {conversation_id, interaction_id}
       STEP 4  supervisor envelopes ─► frontend events, as they arrive
                  status ─► status · reasoning ─► reasoning · token ─► token
       STEP 5  final ─► the answer: text, reasoning, entities, sources, artifacts
                  charts: chart_gen's figures, ONE chart artifact each; or the fallback
                  chart when the user asked to plot and none came back
       STEP 6  SAVE, then artifact × n · entities · sources · (awaiting_input) · done
                  — saved BEFORE done: the frontend reloads the conversation on done
       failure ─► the placeholder saved as failed ─► error

    subscribers: subscribe() replays every buffered event, then follows live —
    a reconnecting browser (GET /chat/stream/{id}) sees the whole turn.
    keepalive: ": ping" every S.keepalive_s while nothing else is sent, so CloudFront
    (which drops a response silent for 30 s) never sees a gap.

FROM ACT
    The turn is a background task, not the request: a closed tab does not lose the
    answer — it finishes and is saved. The SSE response only watches it.

WHAT THIS DOES NOT DO
    It keeps live turns in THIS process only. After a restart, a turn that was still
    running is reported as failed ("interrupted"), never as streaming forever.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from . import plots, supervisor
from .models import Message
from .settings import S
from .store import get_store

log = logging.getLogger("backend.turns")
_DONE = object()


def now_ns() -> int:
    return time.time_ns()


class Turn:
    def __init__(self, user_id: str, conversation_id: str, question: str):
        self.interaction_id = uuid.uuid4().hex
        self.user_id, self.conversation_id, self.question = user_id, conversation_id, question
        self.user_msg_id, self.answer_id = uuid.uuid4().hex, uuid.uuid4().hex
        self.events: list[tuple[str, dict]] = []
        self.listeners: set[asyncio.Queue] = set()
        self.finished = False
        self.task: asyncio.Task | None = None
        self.started = time.time()

    def emit(self, event: str, data: dict) -> None:
        self.events.append((event, data))
        for q in list(self.listeners):
            q.put_nowait((event, data))

    def close(self) -> None:
        self.finished = True
        for q in list(self.listeners):
            q.put_nowait(_DONE)


TURNS: dict[str, Turn] = {}


def is_live(interaction_id: str) -> bool:
    t = TURNS.get(interaction_id)
    return bool(t and not t.finished)


def _history(conversation_id: str) -> list[dict]:
    msgs = [m for m in get_store().list_messages(conversation_id) if m.status in (None, "complete") and m.text]
    return [{"role": m.role, "text": m.text} for m in msgs[-S.history_messages:]]


def _charts(final: dict) -> list[dict]:
    """chart_gen's decision(s) -> ONE chart artifact per figure (the frontend renders one figure each)."""
    out = []
    for decision in (final.get("artifacts") or {}).get("charts") or []:
        for fig in decision.get("figures") or []:
            out.append({"kind": "chart", "chart_type": decision.get("chart_type"),
                        "insight": decision.get("insight", ""), "spec": fig})
    return out


def _answer(turn: Turn, final: dict, text: str) -> Message:
    table = (final.get("artifacts") or {}).get("table")
    charts = _charts(final)
    if not charts and table and plots.wants_plot(turn.question):
        fallback = plots.chart_from_table(table)
        if fallback:
            charts = [fallback]
    artifacts = {}
    if table:
        artifacts["table"] = {"columns": table["columns"], "rows": table["rows"],
                              "row_count": table.get("row_count", len(table["rows"])),
                              "truncated": bool(table.get("truncated")), "sql": table.get("sql", "")}
    if charts:
        artifacts["charts"] = charts
    clarify = final.get("kind") == "clarification"
    return Message(
        message_id=turn.answer_id, conversation_id=turn.conversation_id, interaction_id=turn.interaction_id,
        role="assistant", msg_kind="clarification" if clarify else "answer",
        text=text or final.get("text") or final.get("clarifying_question") or "",
        reasoning=final.get("reasoning") or None, artifacts=artifacts or None,
        artifact_summary={"tables": 1 if table else 0, "charts": len(charts), "graphs": 0, "exports": 0},
        entities=final.get("entities") or [], options=final.get("options") or None,
        sources=final.get("sources") or [], status="complete", created_at_ns=0,
        trace_id=final.get("trace_id"), agents_used=final.get("agents_used") or [],
        usage_by_agent=final.get("usage_by_agent") or [], question=turn.question, user_id=turn.user_id)


async def run(turn: Turn) -> None:
    st = get_store()
    t0 = now_ns()
    # STEP 1-2
    history = await asyncio.to_thread(_history, turn.conversation_id)
    user = Message(message_id=turn.user_msg_id, conversation_id=turn.conversation_id,
                   interaction_id=turn.interaction_id, role="user", text=turn.question, status="complete",
                   created_at_ns=t0)
    placeholder = Message(message_id=turn.answer_id, conversation_id=turn.conversation_id,
                          interaction_id=turn.interaction_id, role="assistant", status="streaming",
                          created_at_ns=t0 + 1, question=turn.question, user_id=turn.user_id)
    await asyncio.to_thread(st.put_message, user)
    await asyncio.to_thread(st.put_message, placeholder)
    await asyncio.to_thread(_touch, turn.conversation_id, t0, True)
    # STEP 3
    turn.emit("meta", {"conversation_id": turn.conversation_id, "interaction_id": turn.interaction_id,
                       "message_id": turn.answer_id})
    final, tokens = None, []
    try:
        # STEP 4
        async for env in supervisor.stream_turn({"question": turn.question, "history": history,
                                                 "conversation_id": turn.conversation_id}):
            kind = env.get("type")
            if kind == "status":
                turn.emit("status", {k: env.get(k) for k in ("agent", "phase", "detail") if env.get(k)})
            elif kind == "reasoning":
                turn.emit("reasoning", {"text": env.get("text", "")})
            elif kind == "token":
                tokens.append(env.get("text", ""))
                turn.emit("token", {"text": env.get("text", "")})
            elif kind == "final":
                final = env
        if final is None:
            raise supervisor.SupervisorError("the supervisor ended without an answer")
        # STEP 5
        answer = _answer(turn, final, "".join(tokens))
        answer.created_at_ns, answer.latency_ms = t0 + 1, int((now_ns() - t0) / 1e6)
        if answer.msg_kind == "clarification" and not tokens:
            turn.emit("clarification", {"text": answer.text})
        # STEP 6 — save first, then tell the browser
        await asyncio.to_thread(st.put_message, answer)
        await asyncio.to_thread(_touch, turn.conversation_id, now_ns())
        if answer.artifacts and answer.artifacts.get("table"):
            t = answer.artifacts["table"]
            turn.emit("artifact", {"kind": "table", "data": {"columns": t["columns"], "rows": t["rows"],
                                                             "row_count": t["row_count"], "truncated": t["truncated"]}})
        for c in (answer.artifacts or {}).get("charts") or []:
            turn.emit("artifact", {"kind": "chart", "data": c})
        turn.emit("entities", {"entities": answer.entities})
        turn.emit("sources", {"sources": answer.sources})
        if answer.msg_kind == "clarification":
            turn.emit("awaiting_input", {"options": answer.options or [], "message_id": answer.message_id})
        turn.emit("done", {"message_id": answer.message_id, "interaction_id": turn.interaction_id})
    except asyncio.CancelledError:
        placeholder.status, placeholder.text = "stopped", "".join(tokens) or "Stopped."
        await asyncio.to_thread(st.put_message, placeholder)
        turn.emit("done", {"message_id": placeholder.message_id, "interaction_id": turn.interaction_id, "stopped": True})
        raise
    except Exception as exc:
        log.exception("[turn %s] failed", turn.interaction_id)
        placeholder.status = "failed"
        placeholder.text = "".join(tokens) or "Sorry — this answer could not be completed."
        placeholder.reasoning = f"Failed: {exc}"
        await asyncio.to_thread(st.put_message, placeholder)
        turn.emit("error", {"detail": str(exc), "message_id": placeholder.message_id})
    finally:
        turn.close()


def _touch(conversation_id: str, at_ns: int, new_turn: bool = False) -> None:
    """Move the conversation to the top of the sidebar; count the turn once, at its start."""
    st = get_store()
    conv = st.get_conversation(conversation_id)
    if conv:
        conv["last_activity_ns"] = conv["updated_at_ns"] = at_ns
        if new_turn:
            conv["interaction_count"] = conv.get("interaction_count", 0) + 1
        st.put_conversation(conv)


def start(user_id: str, conversation_id: str, question: str) -> Turn:
    turn = Turn(user_id, conversation_id, question)
    TURNS[turn.interaction_id] = turn
    turn.task = asyncio.get_running_loop().create_task(run(turn))
    return turn


def stop(interaction_id: str) -> bool:
    turn = TURNS.get(interaction_id)
    if turn and turn.task and not turn.finished:
        turn.task.cancel()
        return True
    return False


async def subscribe(turn: Turn):
    """SSE text: every event so far, then live ones, with keepalive comments."""
    q: asyncio.Queue = asyncio.Queue()
    for ev in list(turn.events):
        q.put_nowait(ev)
    if turn.finished:
        q.put_nowait(_DONE)
    else:
        turn.listeners.add(q)
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=S.keepalive_s)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if item is _DONE:
                return
            event, data = item
            yield f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
    finally:
        turn.listeners.discard(q)
