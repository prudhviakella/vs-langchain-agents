"""A minimal A2A implementation, written to be read.

    types.py    the four objects: AgentCard, Message, Task, Artifact
    queue.py    the event queue — why a task can outlive its connection
    server.py   JSON-RPC + SSE, about 150 lines
    client.py   discover, send, stream, resubscribe

Not a production library. Everything here maps one-to-one onto the real spec,
with the parts that do not change how it works left out.
"""

from .client import A2AClient
from .queue import EventQueue, QueueRegistry
from .server import A2AServer
from .types import AgentCard, Artifact, Message, Part, Skill, Task

__all__ = ["A2AClient", "A2AServer", "AgentCard", "Artifact", "EventQueue",
           "Message", "Part", "QueueRegistry", "Skill", "Task"]
