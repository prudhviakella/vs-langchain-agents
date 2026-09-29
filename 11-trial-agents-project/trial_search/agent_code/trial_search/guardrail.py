"""Bedrock Guardrail, applied through the ApplyGuardrail API.

HIGH-LEVEL FLOW
---------------

The guardrail is implemented as AgentMiddleware rather than as a model-
specific configuration.

    Analyst Question
          │
          ▼
    before_agent
          │
          ▼
    ApplyGuardrail(source="INPUT")
          │
          ├── GUARDRAIL_INTERVENED
          │       │
          │       └── GuardrailBlocked
          │
          └── allowed
                  │
                  ▼
          Agent Loop
          ┌───────────────┐
          │               │
          │    Model      │
          │      ↕        │
          │    Tools      │
          │               │
          └───────┬───────┘
                  │
                  ▼
             after_model
                  │
                  ▼
        ApplyGuardrail(source="OUTPUT")
                  │
                  ├── GUARDRAIL_INTERVENED
                  │       │
                  │       └── GuardrailBlocked
                  │
                  └── allowed


WHY MIDDLEWARE INSTEAD OF guardrail_config
------------------------------------------

`guardrail_config` is a parameter understood by Bedrock's Converse API.

This application uses an OpenAI chat model.

Therefore the OpenAI model would never receive or enforce a Bedrock
Converse `guardrail_config`.

That could result in a particularly dangerous failure mode:

    application appears configured with a guardrail
                         │
                         ▼
                  OpenAI model
                         │
                         ▼
                guardrail_config ignored

There would be no obvious model error.

Instead, this implementation calls the provider-independent:

    Bedrock Runtime ApplyGuardrail API

directly.

Therefore the same Bedrock Guardrail can be applied regardless of which
LLM provider is being used.


WHAT IS CHECKED
---------------

INPUT:

    The original analyst question is checked.

    This includes input-oriented policies such as:

        PROMPT_ATTACK


OUTPUT:

    Text authored by the model is checked.

    This includes:

        - normal AIMessage content
        - ModelDecision.note
        - ModelDecision.clarifying_question


WHAT IS NOT CHECKED
-------------------

Retrieved clinical evidence is deliberately NOT sent through the
output guardrail.

Clinical trial protocols can legitimately contain words describing:

    - deaths
    - overdoses
    - violence
    - adverse events
    - serious medical outcomes

Running a violence/content filter over the retrieved evidence could
therefore block legitimate clinical information.

The retrieved material is evidence, not model-authored output.

It is passed to the model inside:

    <untrusted_data>
    ...
    </untrusted_data>


WHAT THIS MODULE DOES NOT DO
----------------------------

It does not rewrite or sanitize model text.

If the guardrail intervenes:

    ApplyGuardrail
         │
         ▼
    GuardrailBlocked
         │
         ▼
    orchestrate()
         │
         ▼
    TrialSearchResponse(
        result_shape="unanswerable"
    )

The configured guardrail message is carried with the exception so the
caller can decide what to show to the user.
"""


# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# Used to execute the synchronous AWS ApplyGuardrail call without blocking
# the asynchronous agent event loop.
import asyncio

# Used for operational logging when a guardrail intervention occurs.
import logging


# ---------------------------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------------------------

# Base class for implementing LangChain agent middleware.
from langchain.agents.middleware import AgentMiddleware

# Used to distinguish model-authored messages from user messages.
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)


# ---------------------------------------------------------------------------
# Local tracing helpers
# ---------------------------------------------------------------------------

# set_span_attrs() adds guardrail information to the current tracing span.
#
# span() creates the trace span around the ApplyGuardrail API call.
from .tracing import (
    set_span_attrs,
    span,
)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

log = logging.getLogger(
    "agent.guardrail"
)


# ===========================================================================
# Lazy Bedrock client
# ===========================================================================

# Process-local Bedrock Runtime client.

# It starts as None and is created only when the first guardrail check
# occurs.
_client = None


def _bedrock():
    """Return a cached Bedrock Runtime client.

    The client is created lazily rather than during module import.

    This has two benefits:

        1. Importing the module does not immediately create an AWS client.
        2. Warm containers reuse the same boto3 client across requests.

    The client is therefore created once per warm runtime process.
    """

    global _client

    # Create the client only on first use.
    if _client is None:

        # Import boto3 lazily.
        import boto3

        # Bedrock Runtime provides the ApplyGuardrail API.
        _client = boto3.client(
            "bedrock-runtime"
        )

    # Return the cached client.
    return _client


# ===========================================================================
# Guardrail exception
# ===========================================================================

class GuardrailBlocked(Exception):
    """Exception raised when Bedrock Guardrails intervene.

    The exception carries two pieces of information:

        source
            INPUT or OUTPUT

        message
            The configured message returned by the guardrail.

    orchestrate() catches this exception and converts it into an
    unanswerable application-level response.
    """

    def __init__(
        self,
        source: str,
        message: str,
    ):
        # Create a concise exception message for normal exception handling.
        super().__init__(
            f"{source} blocked by guardrail"
        )

        # Preserve the source and configured guardrail message so callers
        # can use them without parsing the exception string.
        self.source = source
        self.message = message


# ===========================================================================
# Core ApplyGuardrail operation
# ===========================================================================

def check(
    text: str,
    source: str,
    guardrail_id: str,
    guardrail_version: str,
    client=None,
) -> None:
    """Run Bedrock ApplyGuardrail against a piece of text.

    Parameters
    ----------
    text:
        Text to evaluate.

    source:
        Bedrock guardrail source.

        Typically:

            INPUT
            OUTPUT

    guardrail_id:
        Bedrock Guardrail identifier.

    guardrail_version:
        Exact Guardrail version to use.

    client:
        Optional Bedrock client.

        This is primarily useful for tests, where a mocked client can
        be supplied instead of creating a real AWS client.

    Returns
    -------
    None

    Raises
    ------
    GuardrailBlocked
        When Bedrock returns GUARDRAIL_INTERVENED.
    """

    # Do nothing for empty/whitespace-only content.
    #
    # There is nothing meaningful for the guardrail to inspect.
    if not text or not text.strip():
        return


    # Create a tracing span around the ApplyGuardrail API call.
    #
    # Example span names:
    #
    #     guardrail.input
    #     guardrail.output
    with span(
        f"guardrail.{source.lower()}",
        chars=len(text),
    ) as sp:

        # Use the explicitly supplied client when testing.
        #
        # Otherwise use the lazily-created production Bedrock client.
        response = (
            client or _bedrock()
        ).apply_guardrail(

            # Guardrail identifier.
            guardrailIdentifier=guardrail_id,

            # Exact Guardrail version.
            guardrailVersion=guardrail_version,

            # Tell Bedrock whether this is input or output content.
            #
            # This matters because some guardrail policies are directional.
            source=source,

            # ApplyGuardrail expects content blocks.
            content=[
                {
                    "text": {
                        "text": text
                    }
                }
            ],
        )

        # Record the guardrail action in the tracing span.
        #
        # Typical value:
        #
        #     NONE
        #
        # or:
        #
        #     GUARDRAIL_INTERVENED
        set_span_attrs(
            sp,
            action=response.get(
                "action"
            ),
        )


    # Check whether the guardrail actually intervened.
    if response.get(
        "action"
    ) == "GUARDRAIL_INTERVENED":

        # Bedrock can return one or more configured output messages.
        #
        # Select the first non-empty text response.
        #
        # If none is returned, use a safe generic fallback.
        message = next(
            (
                output.get("text")
                for output in response.get(
                    "outputs",
                    [],
                )
                if output.get("text")
            ),
            "This request was blocked by a content policy.",
        )

        # Record the intervention in the application logs.
        #
        # We intentionally do not log the actual user/model text here.
        log.warning(
            "guardrail intervened on %s",
            source,
        )

        # Stop the agent loop immediately.
        #
        # The caller is responsible for converting this into an
        # application-level response.
        raise GuardrailBlocked(
            source,
            message,
        )


# ===========================================================================
# Message-content extraction
# ===========================================================================

def _text(
    content,
) -> str:
    """Normalize LangChain message content into plain text.

    LangChain message content can be represented as:

        str

    or:

        [
            {
                "type": "text",
                "text": "..."
            }
        ]

    The guardrail only needs the textual portion.
    """

    # Most messages contain a simple string.
    if isinstance(
        content,
        str,
    ):
        return content

    # Some messages contain a list of content blocks.
    if isinstance(
        content,
        list,
    ):

        # Extract only text blocks and join them into one string.
        return " ".join(
            block.get(
                "text",
                "",
            )
            for block in content
            if (
                isinstance(block, dict)
                and block.get("type")
                == "text"
            )
        )

    # Unknown/non-text content is ignored.
    return ""


# ===========================================================================
# Guardrail middleware
# ===========================================================================

class GuardrailMiddleware(AgentMiddleware):
    """Apply INPUT guardrail once and OUTPUT guardrail after each model call.

    Lifecycle:

        before_agent()
             │
             └── INPUT guardrail

        model
             │
             ▼
        after_model()
             │
             └── OUTPUT guardrail

    Both synchronous and asynchronous middleware hooks are implemented.
    """

    def __init__(
        self,
        guardrail_id: str,
        guardrail_version: str,
        decision_tool: str,
        client=None,
    ):
        """Initialize the middleware.

        Parameters
        ----------
        guardrail_id:
            Bedrock Guardrail ID.

        guardrail_version:
            Exact Guardrail version.

        decision_tool:
            Name of the structured decision tool.

            For trial_search this is:

                ModelDecision

            It is used to locate the model's `note` and
            `clarifying_question` fields.

        client:
            Optional injected Bedrock client, primarily for testing.
        """

        # Initialize the LangChain middleware base class.
        super().__init__()

        # Store the two ApplyGuardrail arguments together.
        #
        # This lets the middleware later call:
        #
        #     check(..., *self.args)
        self.args = (
            guardrail_id,
            guardrail_version,
        )

        # Store the structured decision tool name.
        self.decision_tool = decision_tool

        # Optional injected client for unit tests.
        self.client = client


    # =======================================================================
    # INPUT GUARDRAIL
    # =======================================================================

    def _question(
        self,
        state,
    ) -> str:
        """Find the latest HumanMessage in the current agent state.

        We do not simply use messages[-1] because the final message in an
        agent loop may be a ToolMessage or another internal message.

        The guardrail should inspect the actual analyst/user question.
        """

        # Collect all HumanMessages from the current state.
        human = [
            message
            for message in state.get(
                "messages",
                []
            )
            if isinstance(
                message,
                HumanMessage,
            )
        ]

        # Use the most recent human message.
        #
        # If no human message exists, return an empty string.
        return (
            _text(
                human[-1].content
            )
            if human
            else ""
        )


    def before_agent(
        self,
        state,
        runtime,
    ):
        """Run the INPUT guardrail before the agent loop starts.

        This happens once for the user turn.

        If the guardrail intervenes, check() raises GuardrailBlocked and
        the agent loop does not continue.
        """

        # Extract the analyst's question and evaluate it as INPUT.
        check(
            self._question(
                state
            ),
            "INPUT",
            *self.args,
            client=self.client,
        )

        # Returning None means there is no middleware state update.
        return None


    async def abefore_agent(
        self,
        state,
        runtime,
    ):
        """Async version of before_agent().

        boto3's ApplyGuardrail call is synchronous.

        Calling it directly from an async agent would block the event
        loop.

        Therefore the synchronous check() is moved to a worker thread
        using asyncio.to_thread().
        """

        # Execute the synchronous AWS API call outside the async event
        # loop.
        await asyncio.to_thread(
            check,
            self._question(
                state
            ),
            "INPUT",
            *self.args,
            client=self.client,
        )

        # No state modification.
        return None


    # =======================================================================
    # OUTPUT GUARDRAIL
    # =======================================================================

    def _authored(
        self,
        state,
    ) -> str:
        """Extract text actually authored by the model.

        IMPORTANT
        ---------

        We deliberately locate the newest AIMessage instead of using:

            messages[-1]

        Why?

        During structured output, create_agent can append:

            AIMessage
                containing the ModelDecision tool call

            ToolMessage
                acknowledging/handling that structured decision

        Therefore:

            messages[-1]

        can be the ToolMessage rather than the model's actual final
        decision.

        If we inspected only messages[-1], the final ModelDecision.note
        and clarifying_question could escape output guardrail checking.
        """

        # Search backwards through the message history and find the
        # newest AI-authored message.
        message = next(
            (
                m
                for m in reversed(
                    state.get(
                        "messages",
                        [],
                    )
                )
                if isinstance(
                    m,
                    AIMessage,
                )
            ),
            None,
        )

        # If there is no model message, there is nothing to check.
        if message is None:
            return ""


        # Start with normal model message content.
        parts = [
            _text(
                message.content
            )
        ]


        # -------------------------------------------------------------------
        # Inspect structured ModelDecision
        # -------------------------------------------------------------------

        # The model may have emitted tool calls.
        #
        # We specifically inspect the structured decision tool because
        # important model-authored text can live in its arguments rather
        # than in AIMessage.content.
        for call in message.tool_calls or []:

            # Gateway/tool names can contain prefixes, therefore use
            # suffix matching.
            if call[
                "name"
            ].endswith(
                self.decision_tool
            ):

                # Read the structured tool arguments.
                args = call.get(
                    "args",
                    {},
                )

                # ModelDecision.note is model-authored text.
                parts += [
                    str(
                        args.get(
                            "note"
                        )
                        or ""
                    ),

                    # ModelDecision.clarifying_question is also
                    # model-authored text and must be checked.
                    str(
                        args.get(
                            "clarifying_question"
                        )
                        or ""
                    ),
                ]


        # Combine all model-authored text into a single string that can
        # be sent to ApplyGuardrail.
        return " ".join(
            part
            for part in parts
            if part
        )


    def after_model(
        self,
        state,
        runtime,
    ):
        """Run the OUTPUT guardrail after every model call.

        This catches:

            - normal model-generated content
            - structured decision notes
            - structured clarifying questions
        """

        # Extract only text authored by the model.
        authored = self._authored(
            state
        )

        # Evaluate that content as OUTPUT.
        check(
            authored,
            "OUTPUT",
            *self.args,
            client=self.client,
        )

        # No middleware state update.
        return None


    async def aafter_model(
        self,
        state,
        runtime,
    ):
        """Async version of after_model().

        ApplyGuardrail is a synchronous boto3 API call, so execute it
        through asyncio.to_thread() to avoid blocking the event loop.
        """

        # Extract the model-authored text first.
        authored = self._authored(
            state
        )

        # Execute the blocking ApplyGuardrail request in a worker thread.
        await asyncio.to_thread(
            check,
            authored,
            "OUTPUT",
            *self.args,
            client=self.client,
        )

        # No middleware state update.
        return None