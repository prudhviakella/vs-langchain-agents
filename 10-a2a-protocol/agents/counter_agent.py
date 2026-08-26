"""Agent 2 — work that takes time, and therefore needs a queue.

    python agents/counter_agent.py     serves on :9102

Counts to N slowly, reporting each step. Ask it to count to 20 and you have
twenty seconds to kill your client, reconnect, and watch it carry on.

What this teaches:
    why a task has states, not just a return value
    the queue: the agent keeps working whether or not anyone is listening
    SSE: events arrive as they happen
    tasks/resubscribe: reattach and catch up with no gap

The lesson is the third line. Run the client, press Ctrl-C halfway, then
resubscribe with the task id. The count did not pause and did not restart.
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2a_mini import A2AServer, AgentCard, Artifact, EventQueue, Message, Part, Skill, Task

CARD = AgentCard(
    name="counter",
    description="Counts to a number, one step per second, reporting progress.",
    url="http://127.0.0.1:9102",
    capabilities={"streaming": True, "pushNotifications": False},
    skills=[Skill(id="count", name="Count slowly",
                  description="Counts to N, emitting a progress event each second",
                  examples=["count to 10"])],
)


async def execute(task: Task, message: Message, queue: EventQueue) -> None:
    """Count, putting a progress event on the queue at each step.

    Note what this function does NOT do: it does not check whether a client is
    connected, and it has no reference to an HTTP response. It puts events on a
    queue. Whether anyone is draining that queue is not its concern.

    That is the separation the whole protocol rests on. Wire this directly to a
    response object instead and a dropped connection kills the work.
    """
    match = re.search(r"\d+", message.text)
    target = min(int(match.group()) if match else 5, 60)

    for n in range(1, target + 1):
        # put() never blocks. The count keeps its rhythm with zero subscribers,
        # one, or five.
        queue.put({"kind": "progress", "taskId": task.id,
                   "step": n, "of": target, "note": f"counted {n}"})
        await asyncio.sleep(1)

    server.add_artifact(task, Artifact(
        name="count", parts=[Part(text=f"counted to {target}")]))


server = A2AServer(CARD, execute)

if __name__ == "__main__":
    server.run(port=9102)
