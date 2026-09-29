"""Trial Graph A2A server entrypoint.

This is the process that Amazon Bedrock AgentCore Runtime starts.

Dockerfile CMD:

    opentelemetry-instrument python main.py

High-level startup flow:

    Container starts
          │
          ▼
    STEP 1
    Install LangChain/OpenInference instrumentation
          │
          ▼
    STEP 2
    Import application and A2A dependencies
          │
          ▼
    STEP 3
    Load configuration
          │
          ▼
    serve_a2a(TrialGraphExecutor())
          │
          ▼
    AgentCore Runtime is now listening for A2A requests


Per-request flow:

    Supervisor / A2A client
          │
          │ A2A request
          ▼
    TrialGraphExecutor.execute()
          │
          ├── read user question
          │
          ├── extract caller trace context
          │
          ├── attach trace context
          │
          ├── create "trial_graph" span
          │
          ├── orchestrate(question)
          │       │
          │       └── LangGraph agent
          │             ├── LLM
          │             ├── find_entity_by_name
          │             ├── validate_cypher
          │             └── execute_cypher
          │
          ├── record result metadata on span
          │
          └── send TrialGraphResponse through A2A EventQueue


IMPORTANT IMPORT ORDER

LangChain/OpenInference instrumentation must happen BEFORE importing
LangChain-dependent application modules.

Why?

The instrumentor patches LangChain's callback/instrumentation machinery.

If LangChain is imported first and instrumentation is installed later,
the already-imported components may not be instrumented.

Therefore:

    instrument()
       ↓
    import application
       ↓
    run agent

The OpenInference LangChain instrumentor is intentionally used here.


TELEMETRY MUST NOT PREVENT THE AGENT FROM STARTING

Instrumentation is wrapped in try/except.

If telemetry initialization fails:

    [otel] langchain instrumentation unavailable: ...

the container continues starting.

The agent therefore remains available even when tracing is unavailable.


WHY /ping IS NOT IMPLEMENTED HERE

serve_a2a() provides the A2A server infrastructure and health handling.

This file only needs to provide the AgentExecutor implementation.


WHY PORT 9000 IS NOT SPECIFIED

The A2A runtime contract uses port 9000.

serve_a2a() handles the server setup according to the AgentCore Runtime
contract, so this application does not manually create a server or specify
another port.
"""


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — ENABLE LANGCHAIN TRACING/INSTRUMENTATION
# ═══════════════════════════════════════════════════════════════════════
#
# IMPORTANT:
#
# This must execute BEFORE importing modules that use LangChain.
#
# OpenInference patches LangChain's instrumentation/callback mechanisms.
#
# If this import or initialization fails, we don't want the whole
# AgentCore Runtime container to fail.
#
# The agent can still run without automatic LangChain telemetry.

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # Install OpenInference instrumentation for LangChain.
    #
    # Once installed, LangChain/LangGraph model and tool activity can
    # appear as nested telemetry spans under our application trace.
    LangChainInstrumentor().instrument()

except Exception as _otel_exc:  # pragma: no cover

    # Telemetry failure should never prevent the agent from serving.
    #
    # flush=True is useful because this process is running inside a
    # container and we want the startup warning immediately visible
    # in the container logs.
    print(
        f"[otel] langchain instrumentation unavailable: {_otel_exc}",
        flush=True
    )


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — IMPORT APPLICATION AND RUNTIME DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════
#
# These imports intentionally occur AFTER instrumentation has been
# installed.
#
# noqa: E402 is used because these imports are deliberately placed
# after executable instrumentation code.

import atexit  # noqa: E402
import logging  # noqa: E402

# A2A server interfaces.
#
# AgentExecutor:
#     Base class that our TrialGraphExecutor implements.
#
# RequestContext:
#     Contains information about the incoming A2A request.
#
from a2a.server.agent_execution import AgentExecutor, RequestContext  # noqa: E402
from a2a.server.events import EventQueue  # noqa: E402

# A2A message types used to send the response back to the caller.
from a2a.types import Message, Part, Role, TextPart  # noqa: E402

# Amazon Bedrock AgentCore Runtime A2A server.
from bedrock_agentcore.runtime import serve_a2a  # noqa: E402

# Provides access to AgentCore request/session information,
# including HTTP headers and session ID.
from bedrock_agentcore.runtime.context import (
    BedrockAgentCoreContext
)  # noqa: E402


# Our actual Trial Graph business logic.
#
# IMPORTANT:
#
# main.py does NOT implement entity resolution, Cypher generation,
# validation, Neo4j execution, etc.
#
# All of that lives behind:
#
#     orchestrate(question)
#
# main.py is the runtime/A2A adapter around it.
from trial_graph.core import orchestrate  # noqa: E402


# Response schema returned by orchestrate().
from trial_graph.schemas import TrialGraphResponse  # noqa: E402


# Application tracing helpers.
#
# agent_span()
#     Creates the main Trial Graph span.
#
# attach_context()
#     Makes the caller's trace context current.
#
# detach_context()
#     Restores the previous trace context.
#
# extract_parent_context()
#     Reads traceparent information from headers/metadata.
#
# set_span_attrs()
#     Adds result information to the current span.
#
# shutdown()
#     Flushes/shuts down tracing exporters.
from trial_graph.tracing import (  # noqa: E402
    agent_span,
    attach_context,
    detach_context,
    extract_parent_context,
    set_span_attrs,
    shutdown
)


# ═══════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
#
# uvicorn may install logging handlers before our application gets here.
#
# Without force=True, basicConfig() could become a no-op because a handler
# already exists.
#
# force=True ensures our logging configuration is actually applied.

logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

# Create a logger specifically for this A2A server.
log = logging.getLogger("agent.trial_graph.main")


# Register tracing shutdown when the Python process exits.
#
# This gives telemetry exporters an opportunity to flush buffered spans.
atexit.register(shutdown)


class TrialGraphExecutor(AgentExecutor):
    """A2A adapter around the Trial Graph agent.

    AgentCore Runtime communicates through the A2A AgentExecutor contract.

    This class translates:

        A2A Request
             ↓
        question string
             ↓
        orchestrate(question)
             ↓
        TrialGraphResponse
             ↓
        A2A Message/Event

    It intentionally contains very little business logic.

    The actual agent workflow lives in:

        trial_graph.core.orchestrate()
    """

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """Process one A2A request.

        Request lifecycle:

            1. Extract question.
            2. Extract caller trace context.
            3. Attach caller trace context.
            4. Create Trial Graph root span.
            5. Run orchestrate().
            6. Record result metadata.
            7. Send response through EventQueue.
            8. Detach trace context.
        """

        # ──────────────────────────────────────────────────────────────
        # STEP 1 — GET THE USER QUESTION
        # ──────────────────────────────────────────────────────────────
        #
        # RequestContext abstracts the incoming A2A message.
        #
        # The returned value is the actual natural-language question
        # that should be passed to the Trial Graph agent.

        question = context.get_user_input()


        # ──────────────────────────────────────────────────────────────
        # STEP 2 — EXTRACT CALLER TRACE CONTEXT
        # ──────────────────────────────────────────────────────────────
        #
        # We want the Trial Graph agent to join the caller's distributed
        # trace instead of creating an unrelated trace.
        #
        # Preferred source:
        #
        #     HTTP trace headers
        #
        # Fallback:
        #
        #     A2A message metadata
        #
        # The fallback is important because signed trace headers can
        # interfere with SigV4 signing in this architecture.
        #
        # The Supervisor therefore places trace information into
        # A2A metadata when necessary.

        metadata = (
            getattr(
                getattr(context, "message", None),
                "metadata",
                None
            )
            or {}
        )


        # Try to extract a parent trace from:
        #
        #     1. AgentCore request headers
        #
        # If that doesn't exist, try:
        #
        #     2. A2A message metadata
        #
        # If neither contains a trace context, attach_context(None)
        # effectively starts this request without a parent trace.

        token = attach_context(
            extract_parent_context(
                BedrockAgentCoreContext.get_request_headers()
            )
            or extract_parent_context(metadata)
        )


        try:

            # ──────────────────────────────────────────────────────────
            # STEP 3 — CREATE THE TRIAL GRAPH ROOT SPAN
            # ──────────────────────────────────────────────────────────
            #
            # This is the application-level span for one Trial Graph
            # request.
            #
            # Any LangGraph/model/tool spans created underneath
            # orchestrate() can become children of this span.
            #
            # Conceptually:
            #
            # trial_graph.request
            #       │
            #       ├── LangGraph
            #       ├── OpenAI model
            #       ├── find_entity_by_name
            #       ├── validate_cypher
            #       └── execute_cypher

            with agent_span(
                "trial_graph",

                # Conversation/request context.
                conversation_id=context.context_id,

                # Same context ID is also used as the A2A request context.
                context_id=context.context_id,

                # AgentCore Runtime session identifier.
                session_id=(
                    BedrockAgentCoreContext.get_session_id()
                ),

                # Don't store the actual user question as an attribute.
                # Instead, record only its size.
                #
                # This reduces unnecessary sensitive/user content
                # in telemetry.
                question_chars=len(question or "")
            ) as sp:

                log.info(
                    "received question (context_id=%s)",
                    context.context_id
                )


                try:

                    # ────────────────────────────────────────────────
                    # STEP 4 — RUN THE ACTUAL TRIAL GRAPH AGENT
                    # ────────────────────────────────────────────────
                    #
                    # This is where main.py hands control to core.py.
                    #
                    # orchestrate() performs:
                    #
                    #     connect_tools()
                    #          ↓
                    #     build_agent()
                    #          ↓
                    #     LangChain agent loop
                    #          ↓
                    #     entity resolution
                    #          ↓
                    #     Cypher validation
                    #          ↓
                    #     Neo4j execution
                    #          ↓
                    #     TrialGraphResponse

                    response: TrialGraphResponse = await orchestrate(
                        question
                    )


                except Exception as exc:

                    # Log the complete exception and traceback.
                    log.exception("orchestrate() failed")

                    # Even though the request is failing, send an A2A
                    # message describing the failure before re-raising.
                    #
                    # Re-raising allows the runtime/framework to recognize
                    # the request as failed as well.
                    await event_queue.enqueue_event(
                        _text_message(
                            (
                                f"trial_graph failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            context_id=context.context_id
                        )
                    )

                    raise


                # ──────────────────────────────────────────────────────
                # STEP 5 — RECORD RESULT METADATA IN THE TRACE
                # ──────────────────────────────────────────────────────
                #
                # We don't put the entire graph/table result into
                # telemetry attributes.
                #
                # Instead record useful metrics:
                #
                #     result_shape
                #     number of rows
                #     number of nodes
                #
                # Example:
                #
                #     result_shape = "table"
                #     rows = 10
                #     nodes = 0

                set_span_attrs(
                    sp,
                    result_shape=response.result_shape,
                    rows=len(response.rows),
                    nodes=len(response.nodes)
                )


                # ──────────────────────────────────────────────────────
                # STEP 6 — SEND THE RESPONSE THROUGH A2A
                # ──────────────────────────────────────────────────────
                #
                # TrialGraphResponse is a Pydantic model.
                #
                # Convert it into JSON because the A2A message carries
                # text content here.
                #
                # The Supervisor can then deserialize the JSON back into
                # its expected TrialGraphResponse structure.

                await event_queue.enqueue_event(
                    _text_message(
                        response.model_dump_json(),
                        context_id=context.context_id
                    )
                )

        finally:

            # ──────────────────────────────────────────────────────────
            # STEP 7 — RESTORE PREVIOUS TRACE CONTEXT
            # ──────────────────────────────────────────────────────────
            #
            # attach_context() changes the current tracing context.
            #
            # It must always be detached, even if orchestrate() fails.
            #
            # That's why this happens in finally.

            detach_context(token)


    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """Handle an A2A cancellation request.

        Trial Graph currently has no special cancellation cleanup.

        Therefore this method only logs the cancellation request.
        """

        log.info(
            "cancel requested (context_id=%s) — nothing to clean up",
            context.context_id
        )


def _text_message(
    text: str,
    context_id: str
) -> Message:
    """Create an A2A agent message containing text.

    The Trial Graph response is serialized to JSON and placed inside
    TextPart.

    Structure:

        Message
          │
          ├── message_id
          ├── role = agent
          ├── context_id
          └── parts
                └── TextPart
                      └── JSON response
    """

    return Message(
        # Give the response a deterministic identifier based on the
        # request context.
        message_id=f"trial_graph-{context_id}",

        # This message is generated by the Trial Graph agent.
        role=Role.agent,

        # A2A messages contain one or more Parts.
        parts=[
            Part(
                root=TextPart(text=text)
            )
        ],

        # Associate the response with the original conversation context.
        context_id=context_id
    )


# ═══════════════════════════════════════════════════════════════════════
# APPLICATION ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ──────────────────────────────────────────────────────────────────
    # STEP 3a — LOAD AND VALIDATE CONFIGURATION
    # ──────────────────────────────────────────────────────────────────
    #
    # Configuration is deliberately loaded BEFORE the A2A server starts.
    #
    # This means configuration problems fail during container startup
    # instead of appearing later when the first user sends a request.
    #
    # Examples:
    #
    #     missing Parameter Store value
    #     missing Secrets Manager value
    #     invalid prompt configuration
    #     missing guardrail configuration
    #
    # AgentCore will therefore report the startup failure clearly in
    # the runtime logs.

    from trial_graph.config import settings

    # settings() loads and validates the complete configuration.
    loaded = settings()

    # Log only useful configuration metadata.
    #
    # Do NOT log:
    #
    #     API keys
    #     passwords
    #     secrets
    #     complete prompts
    #
    # The model name, prompt version, and guardrail identifier are
    # useful for operational debugging.
    log.info(
        "settings loaded: model=%s prompt_version=%s guardrail=%s@%s",
        loaded.openai_model,
        loaded.prompt_version,
        loaded.guardrail_id,
        loaded.guardrail_version
    )


    # ──────────────────────────────────────────────────────────────────
    # STEP 3b — START THE A2A SERVER
    # ──────────────────────────────────────────────────────────────────
    #
    # AgentCore Runtime now takes ownership of the process.
    #
    # serve_a2a():
    #
    #     - starts the A2A server
    #     - registers TrialGraphExecutor
    #     - handles incoming A2A requests
    #     - creates RequestContext/EventQueue objects
    #     - invokes execute() for each request
    #
    # There is intentionally no explicit port argument here.
    #
    # The AgentCore A2A runtime contract uses port 9000.

    serve_a2a(
        TrialGraphExecutor()
    )