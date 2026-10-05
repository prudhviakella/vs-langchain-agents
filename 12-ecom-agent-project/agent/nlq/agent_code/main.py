"""NLQ's A2A server — what AgentCore Runtime runs (port 9000, /).

    supervisor ──A2A message/stream──► NLQExecutor.execute
                                          │ answer_stream(question, history)
                                          ├─ progress event ─► status-update (working)   one SSE frame each
                                          └─ result         ─► artifact "nlq_result" + complete
    request text: {"question": "...", "history": [{"role","text"}]}  or a plain question

The supervisor relays every progress frame into its own stream while NLQ is
still working — that is how the Reasoning tab shows "running this SQL" live.

WHAT THIS DOES NOT DO
    No conversation state: history arrives in each request (the backend owns it).
    An exception fails the TASK with {"error": …}; it never kills the server.
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

from nlq.core import answer_stream

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("nlq.main")


def _part(text: str) -> Part:
    return Part(root=TextPart(text=text))


def parse_request(text: str) -> tuple[str, list[dict]]:
    try:
        body = json.loads(text)
        if isinstance(body, dict) and body.get("question"):
            return str(body["question"]), list(body.get("history") or [])
    except (json.JSONDecodeError, TypeError):
        pass
    return text, []


class NLQExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        question, history = parse_request(context.get_user_input())
        try:
            async for ev in answer_stream(question, history):
                if ev.get("phase") == "done":
                    await updater.add_artifact([_part(json.dumps(ev["result"], default=str))], name="nlq_result")
                    await updater.complete()
                else:
                    await updater.update_status(TaskState.working, message=updater.new_agent_message(
                        [_part(json.dumps(ev, default=str))]), final=False)
        except Exception as exc:
            log.exception("[nlq] question failed")
            await updater.failed(message=updater.new_agent_message([_part(json.dumps({"error": str(exc)}))]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("NLQ questions run to completion")


def agent_card(url: str = "http://localhost:9000/") -> AgentCard:
    return AgentCard(
        name="nlq", version="1.0.0", url=url,
        description="Answers questions about the online store's data (orders, customers, products, payments, "
                    "shipments, inventory, suppliers, promotions, reviews) with read-only SQL on PostgreSQL. "
                    "Returns a table (columns + rows) and the SQL that ran.",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"], default_output_modes=["text"],
        skills=[AgentSkill(id="nlq", name="Store data questions", tags=["sql", "ecommerce", "analytics"],
                           description="Revenue, orders, customers, products, delivery, payments, stock — "
                                       "any question answerable from the ecom schema.",
                           examples=["What is revenue by month?", "Which carrier delivers on time most often?"])])


if __name__ == "__main__":
    serve_a2a(NLQExecutor(), agent_card())
