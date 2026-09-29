"""Conversations: list, search, create, rename, delete, read messages.

    GET    /api/conversations?q=<text>    mine, newest activity first; q filters
                                          title and questions (case-insensitive)
    POST   /api/conversations             new, empty
    PATCH  /api/conversations/{id}        rename
    DELETE /api/conversations/{id}        with its messages and feedback
    GET    /api/conversations/{id}/messages   questions and answers, each answer
                                          with its details and my feedback

Every route checks ownership: another user's conversation answers 404, the
same as one that does not exist, so ids cannot be probed.

WHAT THIS DOES NOT DO

    Deleting a conversation does not delete memories the agent recorded
    from it: episodic and semantic memory belong to the user and are meant
    to outlive a single conversation.
"""
from fastapi import APIRouter, HTTPException

from ..models import Conversation, ConversationCreate, ConversationRename, Message
from ..session import CurrentUser
from ..store import StoreDep

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def owned(store, conversation_id: str, username: str) -> dict:
    conv = store.get_conversation(conversation_id)
    if not conv or conv["username"] != username:
        raise HTTPException(404, "conversation not found")
    return conv


def _out(conv: dict) -> Conversation:
    return Conversation(**{k: conv[k] for k in Conversation.model_fields if k in conv})


@router.get("", response_model=list[Conversation], summary="My conversations")
def list_conversations(username: CurrentUser, store: StoreDep, q: str = "") -> list[Conversation]:
    convs = store.list_conversations(username)
    needle = q.strip().lower()
    if needle:
        convs = [c for c in convs if needle in c["title"].lower()
                 or needle in c.get("search_text", "")]
    return [_out(c) for c in convs]


@router.post("", response_model=Conversation, status_code=201, summary="New conversation")
def create_conversation(body: ConversationCreate, username: CurrentUser,
                        store: StoreDep) -> Conversation:
    return _out(store.create_conversation(username, body.title.strip() or "New conversation"))


@router.patch("/{conversation_id}", response_model=Conversation, summary="Rename")
def rename_conversation(conversation_id: str, body: ConversationRename,
                        username: CurrentUser, store: StoreDep) -> Conversation:
    owned(store, conversation_id, username)
    store.rename_conversation(conversation_id, body.title.strip())
    return _out(store.get_conversation(conversation_id))


@router.delete("/{conversation_id}", status_code=204, summary="Delete, with its messages")
def delete_conversation(conversation_id: str, username: CurrentUser, store: StoreDep) -> None:
    owned(store, conversation_id, username)
    store.delete_conversation(conversation_id)


@router.get("/{conversation_id}/messages", response_model=list[Message],
            summary="Questions and answers, oldest first")
def list_messages(conversation_id: str, username: CurrentUser,
                  store: StoreDep) -> list[Message]:
    owned(store, conversation_id, username)
    mine = {f["message_id"]: f for f in store.feedback_for_conversation(conversation_id)
            if f["username"] == username}
    return [Message(**{k: v for k, v in m.items() if k in Message.model_fields},
                    feedback=mine.get(m["message_id"]))
            for m in store.list_messages(conversation_id)]
