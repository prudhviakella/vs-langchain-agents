"""Agent 1 — the smallest thing that is still A2A.

    python agents/echo_agent.py        serves on :9101

It reverses your text. That is all. The agent is trivial on purpose — the
protocol is what you are here to see, and a clever agent would compete for
attention.

What this teaches:
    the agent card, and that one GET is the whole of discovery
    message/send: ask, wait, get a finished task
    an artifact: the output, separate from the conversation
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2a_mini import A2AServer, AgentCard, Artifact, EventQueue, Message, Part, Skill, Task

CARD = AgentCard(
    name="echo",
    description="Reverses text. Exists to demonstrate the protocol.",
    url="http://127.0.0.1:9101",
    # No streaming. A client that reads this card knows not to call
    # message/stream — declared rather than discovered by getting an error.
    capabilities={"streaming": False, "pushNotifications": False},
    skills=[Skill(id="reverse", name="Reverse text",
                  description="Returns the input text backwards",
                  examples=["hello world"])],
)


async def execute(task: Task, message: Message, queue: EventQueue) -> None:
    """Do the work.

    Signature is the same for every agent: the task, the message that arrived,
    and the queue to report progress on. This one finishes instantly, so it
    reports nothing and just attaches its output.
    """
    reversed_text = message.text[::-1]
    server.add_artifact(task, Artifact(name="reversed",
                                       parts=[Part(text=reversed_text)]))
    task.history.append(Message.agent(reversed_text))


server = A2AServer(CARD, execute)

if __name__ == "__main__":
    server.run(port=9101)
