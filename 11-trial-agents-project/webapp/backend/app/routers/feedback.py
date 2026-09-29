"""Feedback on answers, and the user's own feedback history.

    POST /api/feedback {conversation_id, message_id, rating: up|down, comment}
        only on an ANSWER in one of MY conversations; resubmitting replaces
        my earlier rating on that answer
    GET  /api/feedback          everything I have rated, newest first, with
                                the question and a snippet of the answer

The question and answer snippet are copied onto the feedback when it is
given, so "my feedback" still reads correctly after the conversation is
renamed — and AgentOps can list feedback without reading every conversation.
"""
from fastapi import APIRouter, HTTPException

from ..models import FeedbackIn, FeedbackOut
from ..session import CurrentUser
from ..store import StoreDep
from ..store.base import now_iso
from .conversations import owned

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut, summary="Rate an answer")
def give_feedback(body: FeedbackIn, username: CurrentUser, store: StoreDep) -> FeedbackOut:
    owned(store, body.conversation_id, username)
    messages = store.list_messages(body.conversation_id)
    index = next((i for i, m in enumerate(messages) if m["message_id"] == body.message_id), None)
    if index is None or messages[index]["role"] != "assistant":
        raise HTTPException(404, "answer not found")
    question = next((m["text"] for m in reversed(messages[:index]) if m["role"] == "user"), "")
    record = {"conversation_id": body.conversation_id, "message_id": body.message_id,
              "username": username, "rating": body.rating, "comment": body.comment.strip(),
              "question": question[:500], "answer_snippet": messages[index]["text"][:300],
              "created_at": now_iso()}
    store.put_feedback(record)
    return FeedbackOut(**record)


@router.get("", response_model=list[FeedbackOut], summary="Feedback I have given")
def my_feedback(username: CurrentUser, store: StoreDep) -> list[FeedbackOut]:
    return [FeedbackOut(**f) for f in store.feedback_by_user(username)]
