"""Agent 3 — an agent that stops and asks a question.

    python agents/planner_agent.py     serves on :9103

Plans a trip. If you do not say how many days, it stops and asks — then resumes
on the same task when you answer.

What this teaches:
    input-required: the state everyone forgets
    a task that survives an unbounded wait
    resuming by task id, not by starting again

This is the state that separates a task from a function call. A function that
needs more input has to fail and be called again. A task pauses, keeps
everything it had, and continues when the answer arrives — an hour later, from
a different process.
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2a_mini import A2AServer, AgentCard, Artifact, EventQueue, Message, Part, Skill, Task

CARD = AgentCard(
    name="planner",
    description="Plans a trip. Asks for the duration if you did not give one.",
    url="http://127.0.0.1:9103",
    capabilities={"streaming": True, "pushNotifications": False},
    skills=[Skill(id="plan-trip", name="Plan a trip",
                  description="Produces a day-by-day outline for a destination",
                  examples=["plan a trip to Kyoto", "plan 3 days in Lisbon"])],
)


def _days(text: str) -> int | None:
    match = re.search(r"(\d+)\s*day", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


async def execute(task: Task, message: Message, queue: EventQueue) -> None:
    """Plan the trip, or stop and ask how long it is.

    The whole task history is available, so on resume this reads the original
    request and the answer together. The client sent only "3 days" — everything
    else is still here, on the server, attached to the task id.
    """
    # Only what the caller said. Joining the whole history would fold in this
    # agent's own question — "How many days is the trip?" — and the destination
    # would come out as "Kyoto How many days is". A task's history holds both
    # sides, so anything reading it has to say which side it wants.
    asked = " ".join(m.text for m in task.history if m.role == "user")
    days = _days(asked)

    if days is None:
        # Stop and ask. Note what is NOT happening: no error, no partial result,
        # no losing the original question. The task simply waits.
        queue.put({"kind": "progress", "taskId": task.id,
                   "note": "no duration given, asking"})
        task.history.append(Message.agent("How many days is the trip?"))
        server.set_state(task, "input-required", note="How many days is the trip?")
        # Returning here leaves the task alive in `input-required`. It is not
        # finished and it has not failed; it is waiting, indefinitely.
        return

    destination = re.sub(r"\d+\s*days?", "", asked, flags=re.IGNORECASE)
    destination = re.sub(r"\b(plan|a|trip|to|for|the)\b", "", destination,
                         flags=re.IGNORECASE).strip() or "somewhere"

    lines = []
    for day in range(1, min(days, 14) + 1):
        queue.put({"kind": "progress", "taskId": task.id,
                   "step": day, "of": days, "note": f"planning day {day}"})
        await asyncio.sleep(0.4)
        lines.append(f"Day {day}: explore {destination}")

    plan = "\n".join(lines)
    server.add_artifact(task, Artifact(name="itinerary", parts=[Part(text=plan)]))
    task.history.append(Message.agent(plan))


server = A2AServer(CARD, execute)

if __name__ == "__main__":
    server.run(port=9103)
