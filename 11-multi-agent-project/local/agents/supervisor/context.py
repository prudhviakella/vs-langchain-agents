"""Per-request context — what varies per turn but is not agent state.

`registry` is the agent set discovered this turn. Context rather than config,
because an agent can go down between two turns; not state, because the model
routes by NAME and never needs an endpoint.

`invoke` and `note` are the callbacks that reach the outside world. They live
here so the tool has no import of the A2A client and no knowledge of transport —
which is what lets the same tool work locally and on AgentCore with nothing
changed but what is passed in.

Anything the model must never see belongs here rather than in state. In a
deployment with per-user auth, the caller's token would live in this object with
`repr=False`, so it cannot be printed, logged, or emitted into an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .discovery import Registry


@dataclass
class RequestContext:
    session_id: str = "session"
    registry: Registry = field(default_factory=Registry)

    # Set by core.orchestrate for the turn in flight.
    invoke: Callable[[str, str, str], Awaitable[dict[str, Any]]] = None  # type: ignore
    note: Callable[[str], None] = lambda _text: None
