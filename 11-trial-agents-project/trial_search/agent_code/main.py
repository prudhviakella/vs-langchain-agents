"""trial_search's A2A server entrypoint — what AgentCore Runtime runs.

This file is the ENTRYPOINT of the trial_search specialist.

At container startup:

    Dockerfile
        │
        │ CMD:
        │ opentelemetry-instrument python main.py
        │
        ▼
    STEP 1
    LangChain OpenInference instrumentation
        │
        ▼
    STEP 2
    Import A2A / AgentCore / application modules
        │
        ▼
    STEP 3
    Load configuration
        │
        ▼
    serve_a2a(TrialSearchExecutor())
        │
        ▼
    AgentCore Runtime listens on the A2A contract port 9000


PER-REQUEST FLOW
----------------

Every incoming A2A request follows:

    A2A request
        │
        ▼
    execute()
        │
        ├── Read user question
        │
        ├── Extract caller trace context
        │
        ├── Attach caller context
        │
        ├── Create trial_search agent span
        │
        ├── orchestrate(question)
        │       │
        │       ├── LangGraph
        │       ├── LLM
        │       └── Gateway/MCP tools
        │
        ├── Record result attributes
        │
        ├── Return TrialSearchResponse through A2A
        │
        └── Detach caller context
                │
                ▼
             DONE


IMPORT ORDER IS LOAD-BEARING
----------------------------

The OpenInference LangChain instrumentor patches LangChain's callback
machinery.

Therefore it MUST run before LangChain is imported.

Correct:

    LangChainInstrumentor().instrument()
                    │
                    ▼
            import LangChain
                    │
                    ▼
            create LangGraph agent


Incorrect:

    import LangChain
            │
            ▼
    LangChainInstrumentor().instrument()

In that case previously imported LangChain components may not be
instrumented.

The particularly important detail here is that the instrumentation
package is:

    openinference.instrumentation.langchain

and not:

    opentelemetry-instrumentation-langchain

The OpenInference instrumentation is used because it is compatible with
the current LangGraph setup in this application.


TELEMETRY NEVER STOPS THE AGENT
-------------------------------

Instrumentation is deliberately wrapped in try/except.

If OpenTelemetry instrumentation fails:

    container
       │
       ▼
    application still starts
       │
       ▼
    agent works
       │
       └── without LangChain spans

Observability is useful, but it must never become a runtime dependency
for answering a clinical-trial question.


WHY /ping IS NOT IMPLEMENTED HERE
---------------------------------

`serve_a2a()` registers the A2A server endpoints, including the runtime
health/ping endpoint.

Therefore this file does not need to define a separate `/ping` route.


WHY PORT 9000
------------

Port 9000 is the port defined by the A2A/AgentCore Runtime contract.

`serve_a2a()` owns the server setup.

This file therefore does not explicitly start another HTTP server or
choose another application port.
"""


# ===========================================================================
# STEP 1 — OpenInference LangChain instrumentation
# ===========================================================================

# IMPORTANT:
#
# This must execute BEFORE importing anything from LangChain/LangGraph.
#
# The instrumentor patches LangChain's callback/instrumentation machinery.
#
# If LangChain is imported first, the resulting application can still run,
# but expected LangChain/OpenInference spans may not appear.
try:

    # OpenInference instrumentation used for LangChain/LangGraph tracing.
    from openinference.instrumentation.langchain import (
        LangChainInstrumentor,
    )

    # Install the instrumentation globally for this process.
    LangChainInstrumentor().instrument()

except Exception as _otel_exc:  # pragma: no cover

    # Telemetry must never prevent the application from starting.
    #
    # The exception is printed immediately because logging has not yet been
    # configured at this point in startup.
    print(
        f"[otel] langchain instrumentation unavailable: {_otel_exc}",
        flush=True,
    )


# ===========================================================================
# STEP 2 — Application imports
# ===========================================================================

# Everything below is intentionally imported only AFTER the OpenInference
# instrumentation has been installed.
import atexit  # noqa: E402

# Standard Python logging.
import logging  # noqa: E402


# ---------------------------------------------------------------------------
# A2A server interfaces
# ---------------------------------------------------------------------------

# Base executor class for handling incoming A2A requests.
from a2a.server.agent_execution import (  # noqa: E402
    AgentExecutor,
    RequestContext,
)

# Event queue used to send responses/events back to the A2A caller.
from a2a.server.events import EventQueue  # noqa: E402

# A2A message structures used to construct the specialist response.
from a2a.types import (  # noqa: E402
    Message,
    Part,
    Role,
    TextPart,
)


# ---------------------------------------------------------------------------
# AgentCore Runtime
# ---------------------------------------------------------------------------

# Starts the A2A-compatible AgentCore Runtime server.
from bedrock_agentcore.runtime import serve_a2a  # noqa: E402

# Provides request/session information from the AgentCore Runtime,
# including inbound HTTP headers and session ID.
from bedrock_agentcore.runtime.context import (  # noqa: E402
    BedrockAgentCoreContext,
)


# ---------------------------------------------------------------------------
# Trial Search application
# ---------------------------------------------------------------------------

# Core specialist orchestration:
#
#     connect tools
#         ↓
#     build agent
#         ↓
#     retrieval middleware
#         ↓
#     LangGraph/LLM/tool loop
#         ↓
#     TrialSearchResponse
from trial_search.core import orchestrate  # noqa: E402


# Typed final response returned by the specialist.
from trial_search.schemas import TrialSearchResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Tracing helpers
# ---------------------------------------------------------------------------

# These functions implement the distributed trace lifecycle:
#
#     caller trace
#          ↓
#     extract
#          ↓
#     attach
#          ↓
#     agent span
#          ↓
#     detach
from trial_search.tracing import (  # noqa: E402
    agent_span,
    attach_context,
    detach_context,
    extract_parent_context,
    set_span_attrs,
    shutdown,
)


# ===========================================================================
# Logging setup
# ===========================================================================

# uvicorn may install its own logging handlers before this application
# configures logging.
#
# Without force=True, Python's basicConfig() can become a no-op and
# application log.info() statements may disappear.
logging.basicConfig(
    level=logging.INFO,
    force=True,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    ),
)


# Logger used by this A2A server.
log = logging.getLogger(
    "agent.trial_search.main"
)


# Register OpenTelemetry shutdown handling.

# When the process exits, shutdown() attempts to flush any pending spans.
#
# This is best-effort and does not affect application behavior.
atexit.register(
    shutdown
)


# ===========================================================================
# A2A Executor
# ===========================================================================

class TrialSearchExecutor(
    AgentExecutor
):
    """Bridge the A2A request/event-queue interface to orchestrate().

    AgentCore/A2A communicates with this class.

    The actual Trial Search logic lives in:

        trial_search.core.orchestrate()

    Therefore this executor is intentionally thin.

    Its responsibilities are:

        1. Receive the A2A request.
        2. Extract the user question.
        3. Join the caller's distributed trace.
        4. Create the specialist's root span.
        5. Invoke orchestrate().
        6. Serialize the response.
        7. Put the response on the A2A event queue.
        8. Restore the previous tracing context.
    """


    # =======================================================================
    # Request execution
    # =======================================================================

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute one incoming trial_search A2A request."""

        # Extract the actual user question from the A2A request.
        question = context.get_user_input()


        # -------------------------------------------------------------------
        # Extract caller trace context
        # -------------------------------------------------------------------

        # The supervisor normally places trace context inside A2A message
        # metadata.
        #
        # This is intentional because sending traceparent as a signed AWS
        # request header caused SigV4 signature problems in this architecture.
        #
        # The metadata therefore looks conceptually like:
        #
        #     {
        #         "traceparent": "00-....",
        #         ...
        #     }
        metadata = (
            getattr(
                getattr(
                    context,
                    "message",
                    None,
                ),
                "metadata",
                None,
            )
            or {}
        )


        # Try two possible trace-context sources:
        #
        # 1. AgentCore inbound HTTP request headers
        #
        # 2. A2A message metadata
        #
        # The header is preferred when present.
        #
        # If neither contains a trace context, attach_context(None) becomes
        # a no-op and this request starts its own trace.
        token = attach_context(
            extract_parent_context(
                BedrockAgentCoreContext.get_request_headers()
            )
            or extract_parent_context(
                metadata
            )
        )


        # IMPORTANT:
        #
        # The tracing context must be detached even if orchestrate(),
        # serialization, or event_queue operations fail.
        #
        # Otherwise this long-lived runtime could accidentally reuse the
        # previous request's trace for a later unrelated request.
        try:

            # ---------------------------------------------------------------
            # Create the root span for this specialist execution
            # ---------------------------------------------------------------

            # The span is named:
            #
            #     invoke_agent trial_search
            #
            # and carries AgentCore GenAI Observability attributes.
            #
            # Additional application-specific information is attached:
            #
            #     context_id
            #     session_id
            #     question_chars
            with agent_span(
                "trial_search",

                # A2A conversation/context identifier.
                conversation_id=context.context_id,

                # Application-specific trace attributes.
                context_id=context.context_id,

                # AgentCore session identifier.
                session_id=(
                    BedrockAgentCoreContext.get_session_id()
                ),

                # Store only question length rather than the actual question.
                #
                # This provides useful telemetry without unnecessarily
                # copying user content into traces.
                question_chars=len(
                    question or ""
                ),
            ) as sp:

                # Record receipt of the request.
                #
                # The question itself is intentionally not logged.
                log.info(
                    "received question (context_id=%s)",
                    context.context_id,
                )


                try:

                    # -------------------------------------------------------
                    # Execute the actual specialist agent
                    # -------------------------------------------------------

                    # This is where the complete retrieval workflow runs:
                    #
                    #     orchestrate()
                    #          │
                    #          ├── connect to Gateway
                    #          │
                    #          ├── create LangGraph agent
                    #          │
                    #          ├── Guardrail
                    #          │
                    #          ├── RetrievalMiddleware
                    #          │
                    #          ├── OpenAI model
                    #          │
                    #          ├── semantic_search
                    #          ├── expand_neighbors
                    #          └── expand_table
                    #
                    # The result is a validated TrialSearchResponse.
                    response: TrialSearchResponse = (
                        await orchestrate(
                            question
                        )
                    )


                except Exception as exc:

                    # Log the complete exception with traceback for
                    # operational debugging.
                    log.exception(
                        "orchestrate() failed"
                    )


                    # Send an A2A error message back to the caller before
                    # re-raising the exception.
                    #
                    # The caller therefore receives an explicit failure
                    # event while AgentCore still sees the execution as
                    # failed.
                    await event_queue.enqueue_event(
                        _text_message(
                            (
                                "trial_search failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            context_id=context.context_id,
                        )
                    )

                    # Re-raise so the runtime's normal error handling and
                    # observability can record the failed invocation.
                    raise


                # -----------------------------------------------------------
                # Record specialist-level result metadata
                # -----------------------------------------------------------

                # These attributes become available on the root
                # trial_search span.
                #
                # Example:
                #
                #     trial_search.result_shape = passages
                #     trial_search.passages = 5
                set_span_attrs(
                    sp,
                    result_shape=response.result_shape,
                    passages=len(
                        response.passages
                    ),
                )


                # -----------------------------------------------------------
                # Return the specialist response through A2A
                # -----------------------------------------------------------

                # Serialize the complete Pydantic response as JSON.
                #
                # The Supervisor can deserialize this structured payload
                # and use it as evidence for its own final response.
                await event_queue.enqueue_event(
                    _text_message(
                        response.model_dump_json(),
                        context_id=context.context_id,
                    )
                )


        finally:

            # Always restore the tracing context that existed before this
            # request.
            #
            # This is critical for a long-lived AgentCore server.
            detach_context(
                token
            )


    # =======================================================================
    # Request cancellation
    # =======================================================================

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle an A2A cancellation request.

        trial_search does not currently maintain any external resources
        that require explicit cancellation cleanup here.

        Therefore this method only records the cancellation event.
        """

        log.info(
            "cancel requested (context_id=%s) — nothing to clean up",
            context.context_id,
        )


# ===========================================================================
# A2A response message helper
# ===========================================================================

def _text_message(
    text: str,
    context_id: str,
) -> Message:
    """Wrap plain text in an A2A agent Message.

    The Trial Search response is serialized to JSON before being passed
    here.

    Therefore the resulting A2A message contains:

        Message
          └── Part
                └── TextPart
                      └── TrialSearchResponse JSON
    """

    return Message(
        # Give the response a deterministic identifier for this context.
        message_id=f"trial_search-{context_id}",

        # Mark this message as coming from the agent.
        role=Role.agent,

        # A2A messages contain parts; this response uses one text part.
        parts=[
            Part(
                root=TextPart(
                    text=text
                )
            )
        ],

        # Associate the response with the originating A2A context.
        context_id=context_id,
    )


# ===========================================================================
# Application startup
# ===========================================================================

if __name__ == "__main__":

    # =======================================================================
    # STEP 3a — Load runtime configuration
    # =======================================================================

    # Import configuration only at startup.
    #
    # settings() loads and validates:
    #
    #     - AWS Parameter Store configuration
    #     - Secrets Manager values
    #     - Bedrock Prompt Management prompt
    #     - Guardrail configuration
    #     - Gateway configuration
    #     - retrieval budgets
    #
    # If required configuration is missing or invalid, startup fails.
    #
    # This is preferable to discovering configuration problems only when
    # the first user request arrives.
    from trial_search.config import settings


    # Load the immutable runtime settings.
    loaded = settings()


    # Log non-sensitive configuration details.
    #
    # The OpenAI API key/secret is intentionally not logged.
    log.info(
        "settings loaded: model=%s prompt_version=%s guardrail=%s@%s",
        loaded.openai_model,
        loaded.prompt_version,
        loaded.guardrail_id,
        loaded.guardrail_version,
    )


    # =======================================================================
    # STEP 3b — Start the A2A server
    # =======================================================================

    # Start the AgentCore A2A runtime using the TrialSearchExecutor.
    #
    # There is no explicit port argument because the A2A contract uses
    # the fixed runtime port 9000.
    #
    # serve_a2a() owns:
    #
    #     - HTTP server startup
    #     - A2A routing
    #     - /ping/health handling
    #     - request dispatch to TrialSearchExecutor
    serve_a2a(
        TrialSearchExecutor()
    )