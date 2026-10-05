"""The supervisor's A2A server — what AgentCore Runtime runs (port 9000, /). The backend calls it.

    backend ──A2A message/stream {question, history, conversation_id}──► SupervisorExecutor
        status · reasoning · token  ─► status-update (working), one SSE frame each
        final                       ─► artifact "turn_result" + complete
        an exception                ─► task failed with {"error": …}

contextId = the conversation id: the supervisor passes it on to every specialist
call, so a conversation keeps reaching the same warm microVMs.
"""
from __future__ import annotations

import json
import logging
import sys

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TaskState, TextPart
from a2a.utils import new_task
from bedrock_agentcore.runtime import serve_a2a

from supervisor.core import orchestrate

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("supervisor.main")


def _part(text: str) -> Part:
    return Part(root=TextPart(text=text))


def parse_request(text: str, context_id: str) -> dict:
    try:
        body = json.loads(text)
        if isinstance(body, dict) and body.get("question"):
            return {"question": str(body["question"]), "history": list(body.get("history") or []),
                    "conversation_id": str(body.get("conversation_id") or context_id or "")}
    except (json.JSONDecodeError, TypeError):
        pass
    return {"question": text, "history": [], "conversation_id": context_id or ""}


class SupervisorExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        req = parse_request(context.get_user_input(), getattr(context, "context_id", "") or "")
        try:
            async for ev in orchestrate(req["question"], req["history"], req["conversation_id"]):
                if ev.get("type") == "final":
                    await updater.add_artifact([_part(json.dumps(ev, default=str))], name="turn_result")
                    await updater.complete()
                else:
                    await updater.update_status(TaskState.working, message=updater.new_agent_message(
                        [_part(json.dumps(ev, default=str))]), final=False)
        except Exception as exc:
            log.exception("[supervisor] turn failed")
            await updater.failed(message=updater.new_agent_message([_part(json.dumps({"error": str(exc)}))]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("turns run to completion")


def agent_card(url: str = "http://localhost:9000/") -> AgentCard:
    return AgentCard(name="supervisor", version="1.0.0", url=url,
                     description="Answers questions about the online store's data: plans, asks the NLQ agent, "
                                 "charts with chart_gen, and composes the answer — streamed.",
                     capabilities=AgentCapabilities(streaming=True),
                     default_input_modes=["text"], default_output_modes=["text"],
                     skills=[AgentSkill(id="store-analytics", name="Store analytics", tags=["analytics", "sql", "charts"],
                                        description="Revenue, customers, products, delivery, payments, stock.")])


if __name__ == "__main__":
    serve_a2a(SupervisorExecutor(), agent_card())
