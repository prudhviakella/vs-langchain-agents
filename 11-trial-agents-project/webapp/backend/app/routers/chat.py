"""Chat: one request per turn, answered as a Server-Sent Events stream.

    POST /api/chat {text, conversation_id?}
        STEP 1  the conversation — the caller's, or a new one titled by the question
        STEP 2  history — the last 20 messages (10 interactions) BEFORE this one
        STEP 3  the question is saved
        STEP 4  the supervisor is asked on a worker thread
                    event: conversation   {conversation_id, user_message}     at once
                    event: progress       {elapsed_s}                         every 5 s
                    event: answer         {message}   the saved answer with details
                 or event: error          {message}   the saved failure

WHY ONE REQUEST, NOT POST-THEN-GET

The reference posted the question, then opened a second GET for the stream.
Behind a load balancer with two tasks, that GET can reach the OTHER task,
which knows nothing of the turn. One streaming POST cannot be split.

WHY PROGRESS EVENTS

A turn can run a minute or more. The load balancer closes a connection idle
for longer than its timeout; an event every 5 seconds keeps it busy and lets
the UI show that work is under way.

WHAT HAPPENS IF THE BROWSER LEAVES

The worker thread finishes and the answer is saved anyway; reopening the
conversation shows it. Only the live stream is lost.

WHAT THIS DOES NOT DO

    It does not stream the supervisor's intermediate steps — the supervisor
    returns one final message. The details arrive with the answer.
"""
import asyncio
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .. import answers, settings, supervisor
from ..models import ChatRequest
from ..session import CurrentUser
from ..store import StoreDep
from ..store.base import new_id, now_iso
from .conversations import owned

router = APIRouter(prefix="/api", tags=["chat"])
PROGRESS_EVERY_S = 5
_RUNNING: set = set()


async def _run_turn(store, question: dict, history: list[dict], username: str) -> dict:
    """Ask the supervisor on a worker thread; save and return the answer —
    or the failure, which is saved too so the conversation shows it."""
    started = time.monotonic()
    answer = {"message_id": new_id(), "conversation_id": question["conversation_id"],
              "role": "assistant"}
    try:
        response = await asyncio.to_thread(supervisor.ask, question["text"],
                                           question["conversation_id"], history, username)
        latency_ms = int((time.monotonic() - started) * 1000)
        answer.update(status="complete", text=response.get("composed_answer", ""),
                      details=answers.details(response, latency_ms))
    except Exception as exc:
        answer.update(status="error", text=f"The agents could not answer: {exc}",
                      details={"latency_ms": int((time.monotonic() - started) * 1000)})
    answer["created_at"] = now_iso()
    answer["interaction"] = answers.interaction(answer, question["text"], username)
    store.add_message(answer)
    return answer


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


# Messages read for history: 20 to send, plus room for failed answers, which
# are left out. One failed answer in the window still leaves 20 to send.
HISTORY_READ = settings.HISTORY_MESSAGES + 10


def history_for(messages: list[dict]) -> list[dict]:
    """The last 10 interactions as the supervisor expects them. A failed
    answer is left out: it is an error string, not something the platform said."""
    turns = [{"role": m["role"], "text": m["text"]} for m in messages
             if m["role"] == "user" or m.get("status") == "complete"]
    return turns[-settings.HISTORY_MESSAGES:]


@router.post("/chat", summary="Ask a question; the answer streams back as SSE")
async def chat(body: ChatRequest, username: CurrentUser, store: StoreDep) -> StreamingResponse:
    # STEP 1 the conversation
    if body.conversation_id:
        conv = owned(store, body.conversation_id, username)
    else:
        conv = store.create_conversation(username, body.text.strip()[:80])
    cid = conv["conversation_id"]

    # STEP 2 history, read before the new question is added — only the newest
    # messages, and only their text (not every answer's details)
    history = history_for(store.recent_messages(cid, HISTORY_READ))

    # STEP 3 the question
    question = {"message_id": new_id(), "conversation_id": cid, "role": "user",
                "text": body.text.strip(), "created_at": now_iso(), "status": "complete"}
    store.add_message(question)

    # STEP 4 the turn runs as its OWN task: ask the supervisor, save the answer.
    # The stream below only relays it. A disconnecting browser cancels the
    # stream generator at its next yield — never this task — so the answer is
    # saved either way. _RUNNING holds a reference until it finishes, so the
    # task cannot be garbage-collected mid-turn.
    turn = asyncio.ensure_future(_run_turn(store, question, history, username))
    _RUNNING.add(turn)
    turn.add_done_callback(_RUNNING.discard)

    async def stream():
        yield _event("conversation", {"conversation_id": cid, "user_message": question,
                                      "history_messages": len(history)})
        started = time.monotonic()
        while not turn.done():
            await asyncio.wait({turn}, timeout=PROGRESS_EVERY_S)
            if not turn.done():
                yield _event("progress", {"elapsed_s": int(time.monotonic() - started)})
        answer = turn.result()
        name = "answer" if answer["status"] == "complete" else "error"
        yield _event(name, {"message": {k: v for k, v in answer.items() if k != "interaction"}})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
