"""supervisor's A2A server entrypoint — what AgentCore Runtime runs.

    Dockerfile CMD: opentelemetry-instrument python main.py
        │  (aws-opentelemetry-distro: exporters; on AgentCore Runtime the
        │   OTEL environment is preset by the runtime)
        v
    STEP 1  LangChain instrumentation — BEFORE any langchain import
    STEP 2  imports
    STEP 3  load settings, then serve_a2a on the A2A contract's port 9000
        │
        v
    per request: execute()
        headers -> extract_parent_context -> attach   (the caller's trace)
        span "supervisor.request"                      (this agent's root span)
        orchestrate(...)                              (LangGraph -> model ->
                                                       tools spans nest here)
        detach in finally

IMPORT ORDER IS LOAD-BEARING

The instrumentor patches LangChain's callback machinery; anything imported
before it runs is never instrumented. A linter that reorders these imports
switches the traces off and nothing looks broken. The package is
openINFERENCE, not opentelemetry-instrumentation-langchain, which has a known
incompatibility with current LangGraph and emits no spans without an error.

TELEMETRY NEVER STOPS THE AGENT

The instrumentor is guarded: if it fails, the agent serves without spans.
An earlier version called it unguarded — a telemetry import error would have
kept the container from starting.

WHY /ping IS NOT HERE: serve_a2a registers it.
WHY PORT 9000: the A2A contract port; the Runtime proxies nowhere else.
"""
# STEP 1 — instrumentation, before anything else.
try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument()
except Exception as _otel_exc:                                   # pragma: no cover
    print(f"[otel] langchain instrumentation unavailable: {_otel_exc}", flush=True)

# STEP 2 — imports, safe now that instrumentation is active.
import atexit                                                    # noqa: E402
import logging                                                   # noqa: E402

from a2a.server.agent_execution import AgentExecutor, RequestContext  # noqa: E402
from a2a.server.events import EventQueue                         # noqa: E402
from a2a.types import Message, Part, Role, TextPart              # noqa: E402
from bedrock_agentcore.runtime import serve_a2a                  # noqa: E402
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext  # noqa: E402

from supervisor.core import orchestrate                           # noqa: E402
from supervisor.schemas import SupervisorResponse                       # noqa: E402
from supervisor.tracing import (agent_span, attach_context, detach_context,   # noqa: E402
                               extract_parent_context, set_span_attrs, shutdown)

# force=True: uvicorn installs its own handlers first, and a plain
# basicConfig() is then a silent no-op — every log.info() discarded.
logging.basicConfig(level=logging.INFO, force=True,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("agent.supervisor.main")
atexit.register(shutdown)


class SupervisorExecutor(AgentExecutor):
    """Bridges the A2A request/event-queue interface to orchestrate()."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = context.get_user_input()
        # Join the caller's trace: the traceparent header if the caller sent
        # one, else the A2A message metadata — where the supervisor puts it,
        # because signed trace HEADERS broke SigV4 (see agent_client.py).
        # None when neither exists; the request then starts its own trace.
        metadata = getattr(getattr(context, "message", None), "metadata", None) or {}
        token = attach_context(
            extract_parent_context(BedrockAgentCoreContext.get_request_headers())
            or extract_parent_context(metadata))
        try:
            with agent_span("supervisor", conversation_id=context.context_id,
                            context_id=context.context_id,
                      session_id=BedrockAgentCoreContext.get_session_id(),
                      question_chars=len(question or "")) as sp:
                log.info("received question (context_id=%s)", context.context_id)
                try:
                    # The backend sends the conversation's earlier turns and the
                    # signed-in username in the message metadata — the same
                    # channel as the trace context above. Neither is required:
                    # without them the turn runs with no history and no memory.
                    history = metadata.get("history") if isinstance(metadata.get("history"), list) else []
                    user_id = str(metadata.get("user_id") or "") or None
                    response: SupervisorResponse = await orchestrate(
                        question, context.context_id, history=history, user_id=user_id)
                except Exception as exc:
                    log.exception("orchestrate() failed")
                    await event_queue.enqueue_event(_text_message(
                        f"supervisor failed: {type(exc).__name__}: {exc}",
                        context_id=context.context_id))
                    raise
                set_span_attrs(sp, answerable=response.decision.answerable,
                               calls=len(response.calls),
                               render_target=response.render_target)
                await event_queue.enqueue_event(_text_message(
                    response.model_dump_json(), context_id=context.context_id))
        finally:
            detach_context(token)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        log.info("cancel requested (context_id=%s) — nothing to clean up",
                 context.context_id)


def _text_message(text: str, context_id: str) -> Message:
    return Message(message_id=f"supervisor-{context_id}", role=Role.agent,
                   parts=[Part(root=TextPart(text=text))], context_id=context_id)


if __name__ == "__main__":
    # STEP 3a — load settings BEFORE serving, so a missing parameter, unset
    # secret or unrendered prompt variable fails the container start, which
    # AgentCore reports in /aws/bedrock-agentcore/runtimes/.
    from supervisor.config import settings
    loaded = settings()
    log.info("settings loaded: model=%s prompt_version=%s guardrail=%s@%s",
             loaded.openai_model, loaded.prompt_version,
             loaded.guardrail_id, loaded.guardrail_version)
    # STEP 3b — serve. No explicit port: the A2A contract's fixed 9000.
    serve_a2a(SupervisorExecutor())
