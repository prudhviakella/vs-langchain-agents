"""API models — field for field the frontend's TypeScript types (frontend/src/api/types.ts).

    Analyst · ConversationSummary / ConversationDetail / ConversationPage · Message · Page[T]
    · MessageFeedback · LatestInteraction

Message.artifacts = {table: {columns, rows}, charts: [{kind: "chart", spec: {data, layout}, …}]}
— ONE figure per chart artifact (chart_gen returns up to 3; the backend expands them).
"""
from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Analyst(BaseModel):
    user_id: str
    name: str
    first_name: str | None = None
    last_name: str | None = None
    short_name: str
    email: str | None = None
    role: Literal["Analyst", "Admin"] = "Analyst"
    status: Literal["Active", "Disabled"] = "Active"
    consent_given: bool | None = True


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    status: Literal["Active", "Archived"] = "Active"
    owner_id: str
    owner_name: str | None = None
    is_shared: bool = False
    share_id: str | None = None
    last_activity_ns: int


class ConversationDetail(ConversationSummary):
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    interaction_count: int = 0
    created_at_ns: int
    updated_at_ns: int


class ConversationPage(BaseModel):
    conversations: list[ConversationSummary]
    next_cursor: str | None = None
    has_more: bool = False


class MessageFeedback(BaseModel):
    message_id: str
    likes: int = 0
    dislikes: int = 0
    my_vote: Literal["up", "down"] | None = None
    my_comment: str | None = None
    comments: list[dict] = Field(default_factory=list)


class Message(BaseModel):
    message_id: str
    conversation_id: str
    interaction_id: str
    role: Literal["user", "assistant"]
    msg_kind: Literal["answer", "reasoning", "clarification", "note"] = "answer"
    text: str = ""
    reasoning: str | None = None
    artifacts: dict[str, Any] | None = None
    artifact_summary: dict[str, Any] | None = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    options: list[str] | None = None
    status: str | None = None                     # streaming · complete · failed · stopped
    sources: list[dict[str, Any]] | None = None
    created_at_ns: int
    feedback: MessageFeedback | None = None
    # stored with the turn, not read by the screen
    trace_id: str | None = None
    agents_used: list[str] = Field(default_factory=list)
    usage_by_agent: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int | None = None
    question: str | None = None
    truncation_note: str | None = None
    user_id: str | None = None                    # who asked — Usage Analytics' "by user"


class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None


class LatestInteraction(BaseModel):
    interaction_id: str | None = None
    message_id: str | None = None
    status: Literal["streaming", "complete", "failed", "stopped"] | None = None
    is_active: bool = False
    awaiting_input: bool = False
    answer: Message | None = None
