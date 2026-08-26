"""The four objects A2A is built from.

    AgentCard   what an agent can do, and where to send work
    Message     what you say to it, and what it says back
    Task        one unit of work, with an id and a state
    Artifact    the output

Written out by hand rather than imported from an SDK, because every field here
exists for a reason and the reasons are the lesson. The real spec has more
fields; none of the extra ones change how it works.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Task states
#
# A task is not a request and a response. It is an object with a lifecycle,
# because real work takes longer than a sensible HTTP timeout.
#
#     submitted -> working -> completed
#                     |
#                     +-> input-required -> working -> completed
#                     +-> failed
#                     +-> canceled
#
# `input-required` is the state most people miss. The agent can stop, ask a
# question, and resume on the same task id — minutes or days later.
# ─────────────────────────────────────────────────────────────────────────────
TaskState = Literal["submitted", "working", "input-required",
                    "completed", "failed", "canceled"]

TERMINAL: set[str] = {"completed", "failed", "canceled"}


@dataclass
class Part:
    """One piece of a message. Text here; the spec also has file and data parts."""
    kind: str = "text"
    text: str = ""


@dataclass
class Message:
    """One turn of conversation.

    `role` is always "user" for the caller and "agent" for the remote — even when
    the caller is another program. That trips people up: a supervisor calling a
    specialist sends role="user", because in A2A "user" means "the side asking".
    """
    role: Literal["user", "agent"]
    parts: list[Part]
    messageId: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:8]}")

    @property
    def text(self) -> str:
        """All text parts joined — the common case."""
        return " ".join(p.text for p in self.parts if p.kind == "text")

    @staticmethod
    def user(text: str) -> "Message":
        return Message(role="user", parts=[Part(text=text)])

    @staticmethod
    def agent(text: str) -> "Message":
        return Message(role="agent", parts=[Part(text=text)])


@dataclass
class Artifact:
    """The output of a task.

    Separate from Message on purpose. Messages are the conversation; artifacts
    are the deliverable. A task can produce several — a chart and the table
    behind it — and a client can render them differently from the chat.
    """
    name: str
    parts: list[Part]
    artifactId: str = field(default_factory=lambda: f"art-{uuid.uuid4().hex[:8]}")


@dataclass
class Task:
    """One unit of work.

    The id is what makes everything else possible: resuming after input, polling
    from another process, reattaching a dropped stream. Without a durable id, a
    task is just a slow HTTP call.
    """
    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    contextId: str = field(default_factory=lambda: f"ctx-{uuid.uuid4().hex[:8]}")
    state: TaskState = "submitted"
    history: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    createdAt: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Card
#
# One GET tells a stranger everything needed to use this agent. No registry, no
# SDK, no documentation. That is the point of the format.
#
# Served at a fixed path, so discovery needs no prior arrangement:
#     /.well-known/agent-card.json
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    """One thing the agent can do. Written for another agent to read, not a human."""
    id: str
    name: str
    description: str
    examples: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    # `streaming` decides whether a client may call message/stream. Calling it on
    # an agent that cannot stream is a protocol error, not a slow response — so it
    # is declared here rather than discovered by trying.
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {"streaming": True, "pushNotifications": False})
    defaultInputModes: list[str] = field(default_factory=lambda: ["text/plain"])
    defaultOutputModes: list[str] = field(default_factory=lambda: ["text/plain"])
    skills: list[Skill] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
