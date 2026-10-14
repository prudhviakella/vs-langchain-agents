"""Chat — the streamed turn.

    POST /chat/message {text, conversation_id?, mode?}  ─► SSE: meta · status · reasoning · token · artifact ·
                                                              entities · sources · awaiting_input · done | error
         no conversation_id ─► a new conversation titled by the question
    GET  /chat/stream/{interaction_id}   resume: every event so far, then live (404 when not running)
    POST /chat/{interaction_id}/stop     cancel a running turn (saved as stopped)

The response only WATCHES the turn (turns.subscribe); the turn runs as its own task.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import turns
from ..deps import current_user
from ..models import Analyst
from ..store import get_store
from .conversations import new_conversation

router = APIRouter()
_SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    mode: str | None = None


def _sse(turn: turns.Turn) -> StreamingResponse:
    return StreamingResponse(turns.subscribe(turn), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/chat/message")
async def message(body: ChatIn, me: Analyst = Depends(current_user)):
    if body.conversation_id:
        conv = get_store().get_conversation(body.conversation_id)
        if not conv or conv["owner_id"] != me.user_id:
            raise HTTPException(404, "conversation not found")
    else:
        conv = new_conversation(me, body.text[:60])
    return _sse(turns.start(me.user_id, conv["conversation_id"], body.text.strip()))


@router.get("/chat/stream/{interaction_id}")
async def resume(interaction_id: str, me: Analyst = Depends(current_user)):
    turn = turns.TURNS.get(interaction_id)
    if not turn or turn.user_id != me.user_id:
        raise HTTPException(404, "no running turn")
    return _sse(turn)


@router.post("/chat/{interaction_id}/stop", status_code=204)
async def stop(interaction_id: str, me: Analyst = Depends(current_user)):
    turn = turns.TURNS.get(interaction_id)
    if turn and turn.user_id == me.user_id:
        turns.stop(interaction_id)
