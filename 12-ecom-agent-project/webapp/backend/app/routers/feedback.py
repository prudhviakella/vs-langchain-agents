"""POST /feedback {conversation_id, message_id, vote?, comment?} — a thumb and/or a comment; resubmitting replaces."""
from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..deps import current_user
from ..models import Analyst
from ..store import get_store

router = APIRouter()


class FeedbackIn(BaseModel):
    conversation_id: str
    message_id: str
    vote: Literal["up", "down"] | None = None
    comment: str | None = Field(default=None, max_length=2000)


@router.post("/feedback", status_code=204)
def feedback(body: FeedbackIn, me: Analyst = Depends(current_user)):
    st = get_store()
    conv = st.get_conversation(body.conversation_id)
    m = st.get_message(body.conversation_id, body.message_id) if conv and conv["owner_id"] == me.user_id else None
    if not m or m.role != "assistant":
        raise HTTPException(404, "answer not found")
    st.put_feedback(body.conversation_id, body.message_id, me.user_id, body.vote, body.comment, time.time_ns(),
                    {"question": m.question or "", "answer": (m.text or "")[:300], "user_name": me.name,
                     "conversation_title": conv.get("title", "")})
