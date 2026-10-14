"""Conversations — the sidebar, one conversation, its messages, its latest turn.

    GET    /conversations                       ConversationPage (mine, newest activity first)
    GET    /conversations/search?q=             titles containing q
    POST   /conversations {title}               ConversationDetail
    GET    /conversations/{id}                  ConversationDetail
    PATCH  /conversations/{id} {title}          rename
    DELETE /conversations/{id}                  conversation + messages + feedback
    GET    /conversations/{id}/messages         Page[Message], my feedback joined in
    GET    /conversations/{id}/latest-interaction   is a turn still streaming? (resume on reload)
"""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .. import turns
from ..deps import current_user, owned_conversation
from ..models import (Analyst, ConversationDetail, ConversationPage, ConversationSummary, LatestInteraction,
                      Message, Page)
from ..store import get_store

router = APIRouter()


class NewConversation(BaseModel):
    title: str = Field(default="New conversation", max_length=120)


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


def new_conversation(owner: Analyst, title: str) -> dict:
    now = time.time_ns()
    conv = {"conversation_id": uuid.uuid4().hex, "title": title.strip() or "New conversation",
            "owner_id": owner.user_id, "owner_name": owner.name, "status": "Active",
            "last_activity_ns": now, "created_at_ns": now, "updated_at_ns": now, "interaction_count": 0}
    get_store().put_conversation(conv)
    return conv


def _summary(c: dict) -> ConversationSummary:
    return ConversationSummary(**{k: c[k] for k in ConversationSummary.model_fields if k in c})


@router.get("/conversations", response_model=ConversationPage)
def list_conversations(me: Analyst = Depends(current_user)):
    return ConversationPage(conversations=[_summary(c) for c in get_store().list_conversations(me.user_id)])


@router.get("/conversations/search", response_model=ConversationPage)
def search(q: str = "", me: Analyst = Depends(current_user)):
    needle = q.strip().lower()
    hits = [c for c in get_store().list_conversations(me.user_id, limit=500) if needle in c["title"].lower()]
    return ConversationPage(conversations=[_summary(c) for c in hits])


@router.post("/conversations", response_model=ConversationDetail)
def create(body: NewConversation, me: Analyst = Depends(current_user)):
    return ConversationDetail(**new_conversation(me, body.title))


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_one(conv: dict = Depends(owned_conversation)):
    return ConversationDetail(**conv)


@router.patch("/conversations/{conversation_id}", response_model=ConversationDetail)
def rename(body: Rename, conv: dict = Depends(owned_conversation)):
    conv["title"], conv["updated_at_ns"] = body.title.strip(), time.time_ns()
    get_store().put_conversation(conv)
    return ConversationDetail(**conv)


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete(conv: dict = Depends(owned_conversation)):
    get_store().delete_conversation(conv["conversation_id"])


def _with_live_status(m: Message) -> Message:
    """A placeholder still 'streaming' whose turn is not running here was interrupted."""
    if m.role == "assistant" and m.status == "streaming" and not turns.is_live(m.interaction_id):
        m.status, m.text = "failed", m.text or "This answer was interrupted — please ask again."
    return m


@router.get("/conversations/{conversation_id}/messages", response_model=Page[Message])
def messages(conv: dict = Depends(owned_conversation), me: Analyst = Depends(current_user)):
    st = get_store()
    fb = st.feedback(conv["conversation_id"], me.user_id)
    items = []
    for m in st.list_messages(conv["conversation_id"]):
        m = _with_live_status(m)
        m.feedback = fb.get(m.message_id)
        items.append(m)
    return Page[Message](items=items)


@router.get("/conversations/{conversation_id}/latest-interaction", response_model=LatestInteraction)
def latest(conv: dict = Depends(owned_conversation)):
    answers = [m for m in get_store().list_messages(conv["conversation_id"]) if m.role == "assistant"]
    if not answers:
        return LatestInteraction()
    m = _with_live_status(answers[-1])
    live = turns.is_live(m.interaction_id)
    return LatestInteraction(interaction_id=m.interaction_id, message_id=m.message_id,
                             status="streaming" if live else m.status, is_active=live,
                             awaiting_input=m.msg_kind == "clarification", answer=None if live else m)
