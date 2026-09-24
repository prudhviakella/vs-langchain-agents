"""trial_graph's A2A server entrypoint — this is what AgentCore Runtime
actually runs.

    AgentCore Runtime container starts
        |
        v
    THIS FILE runs
        |
        |-- STEP 1: instrument BEFORE any LangChain/LangGraph import —
        |   see the CRITICAL warning below. This must stay the very
        |   first thing this file does.
        |
        |-- STEP 2: import core (which imports langchain/langgraph) —
        |   only safe now that instrumentation is already active
        |
        `-- STEP 3: serve_a2a(TrialGraphExecutor(), port=9000)
                |
                v
        AgentCore Runtime proxies every invocation to port 9000
        (the A2A contract's fixed port — confirmed directly from the
        bedrock_agentcore SDK's own serve_a2a source, not assumed;
        binding elsewhere fails deployed invocations with HTTP 424)

CRITICAL: INSTRUMENTATION IMPORT ORDER

OpenInference's LangChainInstrumentor must run before langchain or
langgraph are imported anywhere in the process — including transitively,
through core.py. A linter or IDE that reorders these imports
"cleans them up" into something that silently stops producing spans,
with no error anywhere. This is the same load-bearing ordering the
reference ACT Xerebro agents document in their own main.py files —
confirmed there via a real production incident (chart_gen's spans
missing from the unified trace despite byte-identical Dockerfiles).

WHY PORT 9000, NOT 8080

Verified directly against bedrock_agentcore.runtime.a2a's own source,
not assumed from a generic "Custom Agent" example: A2A_CONTRACT_PORT=9000
is hardcoded into the SDK, and the module's own docstring warns that
binding elsewhere produces HTTP 424 (RuntimeClientError) on every
deployed invocation, because the Runtime only proxies to 9000. Several
publicly documented AgentCore examples use port 8080 — those are for a
different, generic HTTP agent contract, not the A2A protocol this agent
actually uses. serve_a2a() below is called with no explicit port, which
means it uses this same 9000 default rather than something drifting.

WHAT serve_a2a HANDLES AUTOMATICALLY, SO THIS FILE DOES NOT HAVE TO

Confirmed directly from bedrock_agentcore.runtime.a2a's real source: a
GET /ping route is registered automatically, returning PingStatus.HEALTHY
by default. AgentCore Runtime's required health-check endpoint is
already satisfied by serve_a2a itself — this file does not build one.
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
from trial_graph.core import orchestrate
from trial_graph.schemas import TrialGraphResponse

log = logging.getLogger("agent.trial_graph.main")


class TrialGraphExecutor(AgentExecutor):
    """Bridges the A2A request/event-queue interface to orchestrate()."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = context.get_user_input()
        log.info("received question (context_id=%s)", context.context_id)

        try:
            response: TrialGraphResponse = await orchestrate(question)
        except Exception as exc:
            # An unhandled exception here becomes TASK_STATE_ERROR
            # automatically (the AgentExecutor contract handles this),
            # but a caller-visible message is still worth sending —
            # silence on failure is worse than a plain error string.
            log.exception("orchestrate() failed")
            await event_queue.enqueue_event(_text_message(
                f"trial_graph failed: {type(exc).__name__}: {exc}",
                context_id=context.context_id))
            raise

        await event_queue.enqueue_event(_text_message(
            response.model_dump_json(), context_id=context.context_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # No long-running, cancellable sub-work here — orchestrate() is
        # one bounded LangGraph invocation, not a resumable job. Nothing
        # to actually cancel mid-flight beyond letting the framework's
        # own asyncio.CancelledError propagate.
        log.info("cancel requested (context_id=%s) — nothing to clean up",
                 context.context_id)


def _text_message(text: str, context_id: str) -> Message:
    return Message(
        message_id=f"trial_graph-{context_id}",
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        context_id=context_id,
    )


if __name__ == "__main__":
    # No explicit port — defaults to the A2A contract's fixed 9000.
    serve_a2a(TrialGraphExecutor())
