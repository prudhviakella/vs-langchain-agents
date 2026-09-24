"""Supervisor's A2A server entrypoint — same structure as trial_graph
and trial_search's own main.py; see trial_graph/agent_code/main.py's
module docstring for the full reasoning behind instrumentation order,
port 9000, and why /ping needs no separate implementation.
"""
# STEP 1 — instrumentation, before anything else.
from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role, TextPart
from bedrock_agentcore.runtime import serve_a2a

# STEP 2 — safe now that instrumentation is active.
from supervisor.core import orchestrate
from supervisor.schemas import SupervisorResponse

log = logging.getLogger("agent.supervisor.main")


class SupervisorExecutor(AgentExecutor):
    """Bridges the A2A request/event-queue interface to orchestrate()."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = context.get_user_input()
        log.info("received question (context_id=%s)", context.context_id)

        try:
            response: SupervisorResponse = await orchestrate(question)
        except Exception as exc:
            log.exception("orchestrate() failed")
            await event_queue.enqueue_event(_text_message(
                f"supervisor failed: {type(exc).__name__}: {exc}",
                context_id=context.context_id))
            raise

        await event_queue.enqueue_event(_text_message(
            response.model_dump_json(), context_id=context.context_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        log.info("cancel requested (context_id=%s) — nothing to clean up",
                 context.context_id)


def _text_message(text: str, context_id: str) -> Message:
    return Message(
        message_id=f"supervisor-{context_id}",
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        context_id=context_id,
    )


if __name__ == "__main__":
    serve_a2a(SupervisorExecutor())
