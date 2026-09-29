"""Request and response contracts — what the frontend reads.

    User            username, first_name, last_name, is_admin
    Conversation    id, title, created_at, updated_at, message_count
    Message         one question or one answer; an answer carries AnswerDetails
    AnswerDetails   agents (with the query each ran), tools, citations,
                    artifact, memory recalled, decision, usage, latency, trace
    Feedback        a rating (+ comment) on one answer, by its asker
"""
from typing import Any, Literal

from pydantic import BaseModel, Field

USERNAME = r"^[a-z0-9][a-z0-9._-]{2,39}$"


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40)
    password: str = Field(..., min_length=8, max_length=200)
    first_name: str | None = Field(None, max_length=60)
    last_name: str | None = Field(None, max_length=60)


class User(BaseModel):
    username: str
    first_name: str
    last_name: str
    is_admin: bool = False


class Conversation(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationCreate(BaseModel):
    title: str = Field("New conversation", max_length=120)


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class AgentStep(BaseModel):
    agent: str
    question: str
    rationale: str
    result_shape: str
    succeeded: bool
    query: str | None = None           # the Cypher trial_graph executed
    stats: dict | None = None          # trial_search: searches, expansions
    searches: list[dict] | None = None # trial_search: each search as it ran


class ToolStep(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    succeeded: bool
    result: str = ""
    items: int = 0


class Citation(BaseModel):
    n: int
    doc_id: str
    page: int | None = None
    headings: list[str] = Field(default_factory=list)
    snippet: str
    rerank_score: float | None = None
    origin: str = "search"


class AnswerDetails(BaseModel):
    agents: list[AgentStep] = Field(default_factory=list)
    tools: list[ToolStep] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    artifact: dict[str, Any] | None = None          # {"kind": "table"|"graph", ...}
    memory: list[dict] = Field(default_factory=list)
    decision: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: int = 0
    history_turns: int = 0
    trace_id: str = ""
    trace_url: str = ""


class Message(BaseModel):
    message_id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    text: str
    created_at: str
    status: Literal["complete", "error"] = "complete"
    details: AnswerDetails | None = None
    feedback: "FeedbackOut | None" = None


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None


class FeedbackIn(BaseModel):
    conversation_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: str = Field("", max_length=2000)


class FeedbackOut(BaseModel):
    conversation_id: str
    message_id: str
    username: str
    rating: Literal["up", "down"]
    comment: str = ""
    question: str = ""
    answer_snippet: str = ""
    created_at: str


class Interaction(BaseModel):
    """One answered turn, as AgentOps lists it."""
    conversation_id: str
    message_id: str
    username: str
    question: str
    status: str
    created_at: str
    agents: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    trace_id: str = ""
    trace_url: str = ""


class UsageSummary(BaseModel):
    days: int
    scope: Literal["mine", "everyone"]
    interactions: int
    errors: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_p50_ms: int
    latency_p95_ms: int
    agents: dict[str, int]
    tools: dict[str, int]
    feedback_up: int
    feedback_down: int


Message.model_rebuild()
