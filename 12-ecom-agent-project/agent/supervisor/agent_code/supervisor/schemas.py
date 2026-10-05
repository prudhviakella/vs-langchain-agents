"""The supervisor's routing decision — what the model may decide before any data is fetched."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    action: Literal["query", "answer", "clarify"] = Field(description=(
        "query: the question needs the store's data. answer: no data needed (greeting, what you can do, "
        "out of scope). clarify: the question is too ambiguous to query (say what is missing)."))
    questions: list[str] = Field(default_factory=list, max_length=3, description=(
        "For query: 1-3 STANDALONE questions for the data agent — resolve 'it', 'that month', 'the second one' "
        "from the conversation. One question unless the user asked several distinct things."))
    rationale: str = Field(description="One or two sentences: why this plan. Shown to the user as reasoning.")
    reply: str = Field(default="", description="For answer: the reply itself. Empty otherwise.")
    clarifying_question: str = Field(default="", description="For clarify: the question to ask the user.")
    options: list[str] = Field(default_factory=list, max_length=4, description="For clarify: 2-4 short choices.")
