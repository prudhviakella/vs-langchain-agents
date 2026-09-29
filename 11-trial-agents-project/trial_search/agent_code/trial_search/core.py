"""trial_search core: the agent loop and middleware that bounds it.

HIGH-LEVEL FLOW
---------------

One analyst question follows this path:

    orchestrate(question)
        │
        ├── connect_tools()
        │       │
        │       └── SigV4-signed MCP connection to AgentCore Gateway
        │               │
        │               ├── semantic_search
        │               ├── expand_neighbors
        │               └── expand_table
        │
        ├── build_agent(tools)
        │       │
        │       └── LangGraph create_agent
        │               │
        │               ├── OpenAI model
        │               ├── ToolStrategy(ModelDecision)
        │               ├── GuardrailMiddleware
        │               └── RetrievalMiddleware
        │
        └── assemble()
                │
                └── TrialSearchResponse


RETRIEVAL MIDDLEWARE
--------------------

RetrievalMiddleware is the main control layer around the search tools.

BEFORE a tool runs:

    - enforce per-tool call limits
    - enforce shared expansion-token budget
    - clamp expand_neighbors window
    - inject max_tokens
    - inject exclude_ids

AFTER a tool returns:

    - parse MCP result
    - record calls
    - record token usage
    - capture passages
    - record executed searches
    - create a compact readable ToolMessage for the model


IMPORTANT
---------

The model's requested arguments are NOT trusted for resource limits.

For example, if the model asks:

    window=1000

the middleware changes it before the Lambda sees it:

    window=min(1000, MAX_WINDOW)

Similarly, the middleware calculates:

    remaining expansion tokens

and sends that value to the Lambda as:

    max_tokens=remaining

Therefore these limits are enforced by code, not merely by prompt
instructions.


WHY FULL PASSAGE TEXT IS SHOWN TO THE MODEL
--------------------------------------------

trial_graph only gives the model a compact result summary.

trial_search is different.

Its purpose is to READ protocol passages and determine whether the
available evidence is sufficient.

For example, the model may need to inspect the END of a passage to
determine whether a sentence continues into another chunk.

A 300-character preview could hide the exact information required.

Therefore trial_search exposes the complete retrieved passage text.

The amount of retrieved text is controlled by the expansion token budget.


STATE
-----

RetrievalState extends AgentState.

The middleware writes counters and captured data using LangGraph
reducers.

For example:

    neighbor_calls: Annotated[int, operator.add]

means:

    Command(update={"neighbor_calls": 1})

adds one to the existing state value rather than replacing it.


RUNTIME-VERIFIED BEHAVIOR
-------------------------

1. Middleware custom state must be declared through `state_schema`.

2. Async agent execution requires both:
       wrap_tool_call
       awrap_tool_call

3. MCP tool results can arrive as:
       ToolMessage.artifact["structured_content"]
   or:
       ToolMessage.content
   where content may itself be a list of text blocks.

4. Gateway tool names may contain a target prefix, so suffix matching
   is used.


WHAT THIS FILE DOES NOT DO
--------------------------

It does NOT decide whether retrieval should happen.

The model decides:

    search
    expand neighbors
    expand table
    stop

This file only controls what the model is allowed to do.

It also does NOT rank passages.

Ranking happens inside the search Lambda, including the Cohere reranking
logic.

After retrieval, this file preserves the passages and finally sorts them
by:

    document
    reading position

so the final response follows document reading order rather than
relevance-score order.
"""


# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# Used to parse JSON returned by MCP tool calls.
import json

# Used for operational logging.
import logging

# Used as the reducer operation for additive LangGraph state fields.
import operator

# Used to implement the async context manager for the MCP connection.
from contextlib import asynccontextmanager

# Used to declare LangGraph state fields with reducers.
from typing import Annotated


# ---------------------------------------------------------------------------
# LangChain / LangGraph imports
# ---------------------------------------------------------------------------

# Creates the LangGraph agent loop.
from langchain.agents import create_agent

# AgentMiddleware provides before/after tool-call interception.
#
# AgentState is the base class that custom middleware state must extend.
from langchain.agents.middleware import AgentMiddleware, AgentState

# ToolStrategy tells the model to produce the structured ModelDecision
# through the tool-based structured-output mechanism.
from langchain.agents.structured_output import ToolStrategy

# ToolMessage is used to send middleware-generated tool results/messages
# back into the agent loop.
from langchain_core.messages import ToolMessage

# Command allows middleware to update LangGraph state after a tool call.
from langgraph.types import Command


# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------

# Loads the cached AWS configuration.
from .config import settings

# Guardrail middleware validates user input and model-authored output.
from .guardrail import (
    GuardrailBlocked,
    GuardrailMiddleware,
)

# Output/input schemas and usage aggregation helpers.
from .schemas import (
    ModelDecision,
    Passage,
    RetrievalStats,
    SearchQuery,
    TrialSearchResponse,
    collect_usage,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(
    "agent.trial_search.core"
)


# ---------------------------------------------------------------------------
# Gateway tool names
# ---------------------------------------------------------------------------

# Logical tool names used by trial_search.
#
# AgentCore Gateway may prefix the actual runtime tool names, therefore
# _kind() below performs suffix matching.
SEARCH = "semantic_search"
NEIGHBORS = "expand_neighbors"
TABLE = "expand_table"


def _kind(
    tool_name: str,
) -> str | None:
    """Convert a Gateway tool name into one of our logical tool names.

    Gateway names may look like:

        semantic_search

or:

        trial-search-tools___semantic_search

Therefore exact equality is not safe.

Suffix matching allows the middleware to recognize either form.
"""

    # Return the first logical tool name that matches the end of the
    # Gateway-provided tool name.
    return next(
        (
            kind
            for kind in (
                SEARCH,
                NEIGHBORS,
                TABLE,
            )
            if tool_name.endswith(kind)
        ),
        None,
    )


# ===========================================================================
# MCP connection
# ===========================================================================

@asynccontextmanager
async def connect_tools():
    """Create a SigV4-authenticated MCP session to AgentCore Gateway.

    Flow:

        trial_search
             │
             │ HTTPS + SigV4
             ▼
        AgentCore Gateway
             │
             ▼
        MCP tools
             │
             ├── semantic_search
             ├── expand_neighbors
             └── expand_table

    The Gateway is the MCP server.

    This function does not implement the tools itself. It loads the
    Gateway-exposed tools and converts them into LangChain-compatible tools.
    """

    # Imported lazily so MCP dependencies are only loaded when the agent
    # actually connects to the Gateway.
    from mcp import ClientSession

    # AWS-specific MCP transport that signs HTTP requests with SigV4.
    from mcp_proxy_for_aws.client import (
        aws_iam_streamablehttp_client,
    )

    # Converts MCP tools into LangChain tools.
    from langchain_mcp_adapters.tools import (
        load_mcp_tools,
    )

    # Get the cached configuration.
    s = settings()

    # Create the AWS IAM/SigV4 authenticated HTTP transport.
    #
    # IMPORTANT:
    #
    # aws_service MUST be:
    #
    #     bedrock-agentcore
    #
    # because that is the service name expected when signing requests
    # to the AgentCore Gateway.
    async with aws_iam_streamablehttp_client(
        endpoint=s.gateway_url,
        aws_service="bedrock-agentcore",
        aws_region=s.region,
    ) as (
        read,
        write,
        _,
    ):

        # Build the MCP client session on top of the authenticated
        # read/write streams.
        async with ClientSession(
            read,
            write,
        ) as session:

            # Perform the MCP initialization handshake.
            await session.initialize()

            # Discover the tools exposed by the Gateway and convert them
            # into LangChain tools.
            yield await load_mcp_tools(
                session
            )


# ===========================================================================
# Middleware state
# ===========================================================================

class RetrievalState(AgentState):
    """State maintained by RetrievalMiddleware.

    Every field written by the middleware is declared here.

    The counters/lists use additive reducers.

    Example:

        current neighbor_calls = 2

        Command(
            update={
                "neighbor_calls": 1
            }
        )

    results in:

        neighbor_calls = 3

    rather than:

        neighbor_calls = 1
    """

    # Number of semantic_search calls performed.
    #
    # operator.add makes this an additive reducer.
    search_calls: Annotated[
        int,
        operator.add,
    ]

    # Number of neighbor expansion calls.
    neighbor_calls: Annotated[
        int,
        operator.add,
    ]

    # Number of table expansion calls.
    table_calls: Annotated[
        int,
        operator.add,
    ]

    # Total expansion tokens consumed.
    #
    # semantic_search does not contribute to this counter.
    expansion_tokens: Annotated[
        int,
        operator.add,
    ]

    # All passages captured from successful retrieval calls.
    #
    # The list reducer appends newly captured passages.
    captured_passages: Annotated[
        list[dict],
        operator.add,
    ]

    # Record of searches executed during this question.
    #
    # This is later exposed through TrialSearchResponse.
    searches: Annotated[
        list[dict],
        operator.add,
    ]


# ===========================================================================
# Reading MCP tool results
# ===========================================================================

def _payload(
    result,
) -> dict | None:
    """Extract the Lambda's structured JSON payload from an MCP result.

    MCP/LangChain adapters can expose tool output in more than one form.

    Possible forms include:

        ToolMessage.artifact["structured_content"]

    or:

        ToolMessage.content

    where content may be:

        - a dictionary
        - a list of text blocks
        - a JSON string

    This function normalizes all supported forms into:

        dict | None
    """

    # We only know how to interpret LangChain ToolMessage instances.
    if not isinstance(
        result,
        ToolMessage,
    ):
        return None

    # Some MCP adapters put structured output into the artifact field.
    artifact = getattr(
        result,
        "artifact",
        None,
    )

    # Prefer structured_content when it is already a dictionary.
    if (
        isinstance(artifact, dict)
        and isinstance(
            artifact.get(
                "structured_content"
            ),
            dict,
        )
    ):
        return artifact[
            "structured_content"
        ]

    # Fall back to the normal ToolMessage content.
    content = result.content

    # Some integrations already provide a dictionary.
    if isinstance(
        content,
        dict,
    ):
        return content

    # MCP text output can arrive as a list of content blocks.
    #
    # Example:
    #
    # [
    #     {"type": "text", "text": "{\"passages\": [...]}"}
    # ]
    #
    # Combine all text blocks into one string.
    if isinstance(
        content,
        list,
    ):

        content = "".join(
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

    # Attempt to parse the final content as JSON.
    try:
        parsed = json.loads(
            content
        )

    # Invalid/non-string content is treated as unreadable rather than
    # crashing the agent loop.
    except (
        TypeError,
        json.JSONDecodeError,
    ):
        return None

    # Only dictionaries are accepted as tool payloads.
    return (
        parsed
        if isinstance(parsed, dict)
        else None
    )


# ===========================================================================
# Untrusted passage fencing
# ===========================================================================

def _fence(
    text: str,
) -> str:
    """Remove untrusted-data fence markers from retrieved passage text.

    Clinical protocol text ultimately originates from external documents.

    Therefore it must be treated as untrusted data.

    The surrounding model-facing message uses:

        <untrusted_data>
        ...
        </untrusted_data>

    If a document itself contained these markers, it could potentially
    interfere with the intended boundary.

    This function strips them before the passage is inserted into the
    model's context.
    """

    # Convert the value to a string and remove both fence markers.
    return str(
        text
    ).replace(
        "<untrusted_data>",
        "",
    ).replace(
        "</untrusted_data>",
        "",
    )


# ===========================================================================
# Retrieval middleware
# ===========================================================================

class RetrievalMiddleware(AgentMiddleware):
    """Bounds and records semantic/table/neighbor retrieval calls.

    The middleware has two responsibilities:

        BEFORE the tool call
            control what the Lambda is allowed to execute

        AFTER the tool call
            capture what actually happened

    It does not decide whether a retrieval should happen.
    """

    # IMPORTANT:
    #
    # LangChain middleware must explicitly declare custom state through
    # state_schema.
    #
    # Without this, keys written through Command can be ignored.
    state_schema = RetrievalState


    def __init__(
        self,
        cfg=None,
    ):
        """Initialize retrieval limits and state-field mappings."""

        # Initialize AgentMiddleware.
        super().__init__()

        # Use injected configuration for tests, otherwise load the
        # production configuration.
        cfg = cfg or settings()

        # Keep the complete configuration object.
        self.cfg = cfg

        # Map each logical tool to its maximum allowed calls.
        self.limits = {
            SEARCH: cfg.max_searches_per_turn,
            NEIGHBORS: cfg.max_neighbor_calls,
            TABLE: cfg.max_table_calls,
        }

        # Map each logical tool to the corresponding state counter.
        self.counter = {
            SEARCH: "search_calls",
            NEIGHBORS: "neighbor_calls",
            TABLE: "table_calls",
        }


    # -----------------------------------------------------------------------
    # Synchronous tool interception
    # -----------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request,
        handler,
    ):
        """Intercept synchronous tool execution.

        The same logic is used for sync and async execution.

        Flow:

            request
              │
              ▼
            _gate()
              │
         ┌────┴────┐
         │         │
       refuse     allow
         │         │
       message   handler()
                   │
                   ▼
                _record()
        """

        # First enforce all limits and rewrite arguments if required.
        gated = self._gate(
            request
        )

        # If _gate() returned a ToolMessage, the call was refused.
        #
        # Do not invoke the actual Lambda.
        if isinstance(
            gated,
            ToolMessage,
        ):
            return gated

        # Execute the tool and record the actual result.
        return self._record(
            gated,
            handler(gated),
        )


    # -----------------------------------------------------------------------
    # Asynchronous tool interception
    # -----------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request,
        handler,
    ):
        """Async equivalent of wrap_tool_call.

        `ainvoke()` requires an async middleware path.

        Therefore both:

            wrap_tool_call
            awrap_tool_call

        are implemented.
        """

        # Apply the same gate used by synchronous execution.
        gated = self._gate(
            request
        )

        # Refused calls never reach the actual tool.
        if isinstance(
            gated,
            ToolMessage,
        ):
            return gated

        # Await the actual asynchronous tool invocation, then record it.
        return self._record(
            gated,
            await handler(gated),
        )


    # =======================================================================
    # BEFORE TOOL CALL
    # =======================================================================

    def _gate(
        self,
        request,
    ):
        """Enforce retrieval budgets and rewrite tool arguments.

        This method runs BEFORE the Lambda receives the request.

        That is important because these limits are code-enforced rather
        than merely instructions in the system prompt.
        """

        # Convert the actual Gateway tool name into one of our logical
        # tool names.
        kind = _kind(
            request.tool_call[
                "name"
            ]
        )

        # This middleware only controls the three retrieval tools.
        #
        # Any other tool is passed through unchanged.
        if kind is None:
            return request

        # Get the current agent state.
        state = request.state or {}

        # The tool-call ID is needed when generating a ToolMessage.
        call_id = request.tool_call[
            "id"
        ]


        # -------------------------------------------------------------------
        # STEP 1 — Per-tool call budget
        # -------------------------------------------------------------------

        # Read how many times this tool has already been executed.
        used_calls = state.get(
            self.counter[kind],
            0,
        )

        # If the limit has already been reached, refuse the call.
        if used_calls >= self.limits[
            kind
        ]:

            return ToolMessage(
                tool_call_id=call_id,
                content=(
                    f"REFUSED: {kind} call limit reached "
                    f"({used_calls}/{self.limits[kind]} this question). "
                    "Answer from the passages you have and state in `note` "
                    "what is missing."
                ),
            )

        # semantic_search does not consume the shared expansion-token
        # budget.
        #
        # Its arguments therefore pass through unchanged.
        if kind == SEARCH:
            return request


        # -------------------------------------------------------------------
        # STEP 2 — Shared expansion token budget
        # -------------------------------------------------------------------

        # Calculate how many expansion tokens remain.
        #
        # Example:
        #
        #     budget = 10,000
        #     already used = 6,000
        #     remaining = 4,000
        remaining = (
            self.cfg.expansion_token_budget
            - state.get(
                "expansion_tokens",
                0,
            )
        )

        # Once the expansion budget is exhausted, refuse the call.
        if remaining <= 0:

            return ToolMessage(
                tool_call_id=call_id,
                content=(
                    f"REFUSED: expansion token budget exhausted "
                    f"({self.cfg.expansion_token_budget} tokens). "
                    "Answer from what you have and state in `note` "
                    "what is missing."
                ),
            )


        # -------------------------------------------------------------------
        # STEP 3 — Rewrite tool arguments
        # -------------------------------------------------------------------

        # Copy the model's requested arguments.
        #
        # We modify the copy rather than the original request.
        args = dict(
            request.tool_call[
                "args"
            ]
        )

        # Tell the Lambda exactly how many expansion tokens remain.
        #
        # The model cannot override this because this value is inserted
        # after the model generated its tool arguments.
        args[
            "max_tokens"
        ] = remaining

        # Exclude every passage that has already been captured.
        #
        # This is how deduplication works during expansion.
        #
        # Example:
        #
        # First request:
        #     window=2
        #
        # Second request:
        #     window=5
        #
        # The second call receives exclude_ids, so it returns only newly
        # discovered chunks rather than charging for the same chunks again.
        args[
            "exclude_ids"
        ] = sorted(
            {
                passage["chunk_id"]
                for passage in state.get(
                    "captured_passages",
                    [],
                )
            }
        )

        # Neighbor expansion has an additional window constraint.
        if kind == NEIGHBORS:

            # Read the model-requested window.
            requested_window = int(
                args.get(
                    "window"
                )
                or 2
            )

            # Clamp it into:
            #
            #     1 <= window <= max_window
            #
            # The Lambda therefore never receives an oversized expansion
            # request.
            args[
                "window"
            ] = max(
                1,
                min(
                    requested_window,
                    self.cfg.max_window,
                ),
            )

        # Return a copy of the request containing our controlled arguments.
        return request.override(
            tool_call={
                **request.tool_call,
                "args": args,
            }
        )


    # =======================================================================
    # AFTER TOOL CALL
    # =======================================================================

    def _record(
        self,
        request,
        result,
    ):
        """Capture the result of an executed retrieval tool.

        Responsibilities:

            - identify the tool
            - parse its MCP payload
            - record searches
            - count failed calls
            - accumulate expansion tokens
            - capture passages
            - send a readable ToolMessage back to the model
        """

        # Determine the logical tool name.
        kind = _kind(
            request.tool_call[
                "name"
            ]
        )

        # If this is not one of our tools, or the tool already returned
        # a LangGraph Command, do not modify the result.
        if (
            kind is None
            or isinstance(
                result,
                Command,
            )
        ):
            return result

        # Tool-call ID used for the generated ToolMessage.
        call_id = request.tool_call[
            "id"
        ]

        # Convert the MCP result into a normalized dictionary.
        payload = _payload(
            result
        )


        # -------------------------------------------------------------------
        # Record semantic searches
        # -------------------------------------------------------------------

        # A tool result is considered failed when:
        #
        #     - the payload could not be parsed
        #     - the payload explicitly reports an error
        failed = (
            payload is None
            or bool(
                payload.get(
                    "error"
                )
            )
        )

        # Only semantic_search calls are recorded in the "Queries"
        # metadata.
        searches = []

        if kind == SEARCH:

            # Read the arguments that actually reached the tool.
            #
            # This is important:
            #
            # the recorded query should describe what was actually
            # executed, not merely what the model initially requested.
            args = request.tool_call.get(
                "args",
                {},
            )

            # Record useful search metadata.
            searches = [
                {
                    "query": str(
                        args.get(
                            "query",
                            "",
                        )
                    ),

                    "doc_id": (
                        args.get(
                            "doc_id"
                        )
                        or None
                    ),

                    "content_type": (
                        args.get(
                            "content_type"
                        )
                        or None
                    ),

                    "top_k": args.get(
                        "top_k"
                    ),

                    # A failed search has no candidates/results.
                    "candidates": (
                        0
                        if failed
                        else int(
                            payload.get(
                                "candidates",
                                0,
                            )
                            or 0
                        )
                    ),

                    "results": (
                        0
                        if failed
                        else len(
                            payload.get(
                                "passages",
                                [],
                            )
                        )
                    ),

                    "reranked": (
                        None
                        if failed
                        else payload.get(
                            "reranked"
                        )
                    ),

                    "succeeded": not failed,
                }
            ]


        # -------------------------------------------------------------------
        # Failed tool call
        # -------------------------------------------------------------------

        # Failed calls still consume a call from the per-tool budget.
        #
        # This prevents a failing tool from being retried indefinitely.
        if failed:

            # Prefer an explicit error detail from the payload.
            #
            # If there is no structured payload, fall back to the raw
            # ToolMessage content.
            detail = (
                (payload or {}).get(
                    "detail"
                )
                or _fence(
                    getattr(
                        result,
                        "content",
                        "",
                    )
                )
            )

            # Update state:
            #
            #     counter += 1
            #     record search metadata
            #     provide readable error to model
            return Command(
                update={
                    self.counter[kind]: 1,

                    "searches": searches,

                    "messages": [
                        ToolMessage(
                            tool_call_id=call_id,
                            content=(
                                f"ERROR from {kind}: {detail}"
                            ),
                        )
                    ],
                }
            )


        # -------------------------------------------------------------------
        # Successful tool result
        # -------------------------------------------------------------------

        # Extract passages returned by the tool.
        passages = payload.get(
            "passages",
            [],
        )

        # Only expansion tools consume expansion tokens.
        #
        # semantic_search contributes zero here.
        tokens = (
            int(
                payload.get(
                    "tokens_used",
                    0,
                )
            )
            if kind != SEARCH
            else 0
        )

        # Read current state.
        state = request.state or {}

        # Build the current counters.
        stats = {
            key: state.get(
                key,
                0,
            )
            for key in (
                "search_calls",
                "neighbor_calls",
                "table_calls",
                "expansion_tokens",
            )
        }

        # Increment the tool's call counter locally.
        stats[
            self.counter[kind]
        ] += 1

        # Add newly consumed expansion tokens.
        stats[
            "expansion_tokens"
        ] += tokens


        # -------------------------------------------------------------------
        # Update LangGraph state
        # -------------------------------------------------------------------

        return Command(
            update={

                # Add one to the appropriate tool counter.
                self.counter[kind]: 1,

                # Add the tokens consumed by this expansion.
                "expansion_tokens": tokens,

                # Preserve the complete returned passages in state.
                #
                # These are used later during final response assembly.
                "captured_passages": passages,

                # Record semantic-search metadata.
                "searches": searches,

                # Send a readable representation back to the model.
                #
                # The model sees passage text, but not necessarily every
                # internal field returned by the Lambda.
                "messages": [
                    ToolMessage(
                        tool_call_id=call_id,
                        content=self._view(
                            kind,
                            payload,
                            passages,
                            stats,
                        ),
                    )
                ],
            }
        )


    # =======================================================================
    # Model-facing tool result
    # =======================================================================

    def _view(
        self,
        kind,
        payload,
        passages,
        stats,
    ) -> str:
        """Build the model-facing representation of a retrieval result.

        This is different from the final TrialSearchResponse.

        The model needs enough information to reason about the evidence,
        while the application keeps the structured state separately.
        """

        # Configuration used for displaying current budget values.
        c = self.cfg

        # Start with the number of passages returned.
        lines = [
            f"{kind}: {len(passages)} passage(s)"
        ]


        # -------------------------------------------------------------------
        # Neighbor-specific information
        # -------------------------------------------------------------------

        if kind == NEIGHBORS:

            # Tell the model what expansion window was actually used and
            # why the expansion stopped.
            lines.append(
                f"window={payload.get('window')} around position "
                f"{payload.get('seed_position')}, stopped by: "
                f"{payload.get('stopped_by')}"
            )


        # -------------------------------------------------------------------
        # Table-specific information
        # -------------------------------------------------------------------

        if kind == TABLE:

            # Table expansion may discover multiple table fragments.
            lines.append(
                f"table has {payload.get('n_fragments')} fragment(s), "
                f"stopped by: {payload.get('stopped_by')}"
            )


        # -------------------------------------------------------------------
        # Empty result
        # -------------------------------------------------------------------

        if not passages:

            # Different wording is used for semantic search because
            # "nothing matched" has a different implication from an
            # expansion that simply found nothing new.
            lines.append(
                "Nothing new was returned."
                if kind != SEARCH
                else
                "No passage matched. The corpus may not cover this — "
                "do not invent an answer."
            )


        # -------------------------------------------------------------------
        # Untrusted passage boundary
        # -------------------------------------------------------------------

        # Everything below this point originates from retrieved documents.
        #
        # Treat it explicitly as untrusted data.
        lines.append(
            "<untrusted_data>"
        )


        # Add each returned passage.
        for p in passages:

            # Build the passage metadata header.
            #
            # Example:
            #
            #     [chunk-123] doc=protocol-1 pos=20 p5 type=paragraph
            head = (
                f"[{p['chunk_id']}] "
                f"doc={p['doc_id']} "
                f"pos={p.get('position')} "
                f"p{p.get('page')} "
                f"type={p.get('content_type')}"
            )

            # Semantic search results may have a relevance score.
            if p.get(
                "score"
            ) is not None:

                head += (
                    f" score={p['score']:.3f}"
                )


            # Table summaries are special.
            #
            # The model is told that expand_table(chunk_id) can retrieve
            # the exact rows.
            if p.get(
                "content_type"
            ) == "table_summary":

                head += (
                    f" n_fragments={p.get('n_fragments')} "
                    "-> expand_table(chunk_id) returns the exact rows"
                )

            # Add metadata header.
            lines.append(
                head
            )

            # Add section/headings metadata.
            lines.append(
                f"headings: {p.get('headings')}"
            )

            # Add the full passage text.
            #
            # This is intentionally NOT truncated because trial_search
            # needs to determine whether the end of a passage contains
            # the information needed to answer the question.
            #
            # _fence() removes any fake untrusted-data boundary markers
            # from the source document.
            lines.append(
                _fence(
                    p.get(
                        "text",
                        "",
                    )
                )
            )

            # Blank line between passages.
            lines.append("")


        # Close the untrusted-data boundary.
        lines.append(
            "</untrusted_data>"
        )


        # -------------------------------------------------------------------
        # Budget status
        # -------------------------------------------------------------------

        # Give the model visibility into the resources already consumed.
        #
        # This helps the model understand why another expansion may be
        # refused, but the actual enforcement still happens in _gate().
        lines.append(
            f"budget used: "
            f"search {stats['search_calls']}/{c.max_searches_per_turn}, "
            f"neighbors {stats['neighbor_calls']}/{c.max_neighbor_calls}, "
            f"table {stats['table_calls']}/{c.max_table_calls}, "
            f"expansion tokens "
            f"{stats['expansion_tokens']}/{c.expansion_token_budget}"
        )

        # Return the model-facing text.
        return "\n".join(
            lines
        )


# ===========================================================================
# Agent construction
# ===========================================================================

def build_agent(
    tools: list,
    model=None,
    cfg=None,
):
    """Build the trial_search LangGraph agent.

    Dependencies can be injected:

        model
        cfg

    This makes the agent testable without requiring live AWS/OpenAI
    resources.

    Middleware ordering is intentional:

        GuardrailMiddleware
                ↓
        RetrievalMiddleware

    Therefore the input guardrail is evaluated before retrieval behavior
    begins.
    """

    # Use injected configuration for tests or load production settings.
    s = cfg or settings()

    # Create the LangGraph agent.
    return create_agent(

        # Logical agent name.
        name="trial_search",

        # Injected model for tests or configured production model.
        model=model or s.chat_model(),

        # MCP tools loaded from AgentCore Gateway.
        tools=tools,

        # Version-pinned system prompt.
        system_prompt=s.system_prompt,

        # The model's only structured output is ModelDecision.
        #
        # ToolStrategy ensures structured output is represented as a
        # tool call rather than relying on a bare schema.
        response_format=ToolStrategy(
            ModelDecision
        ),

        # Middleware stack.
        middleware=[
            # Guardrail runs first.
            #
            # decision_tool tells it which structured decision tool to
            # inspect for model-authored note/clarifying_question text.
            GuardrailMiddleware(
                s.guardrail_id,
                s.guardrail_version,
                decision_tool="ModelDecision",
            ),

            # RetrievalMiddleware bounds and records retrieval calls.
            RetrievalMiddleware(
                s
            ),
        ],
    )


# ===========================================================================
# Final response assembly
# ===========================================================================

def assemble(
    result: dict,
    cfg=None,
) -> TrialSearchResponse:
    """Build TrialSearchResponse from the finished LangGraph state.

    The final response is assembled from actual state rather than asking
    the model to reproduce the retrieved passages.

    PROCESS
    -------

    STEP 1
        Deduplicate passages by chunk_id.

        The first occurrence wins.

        This means if semantic_search found a chunk with a relevance
        score and a later neighbor expansion returned the same chunk,
        the original search result is preserved.

    STEP 2
        Sort passages by document and reading position.

        Final response order is therefore document reading order rather
        than relevance score.

    STEP 3
        Determine result_shape based on what actually happened.
    """

    # Load configuration.
    s = cfg or settings()

    # Extract the structured model decision.
    #
    # ModelDecision contains only the small decision fields:
    #
    #     entities
    #     answerable
    #     note
    decision: ModelDecision = result[
        "structured_response"
    ]


    # -----------------------------------------------------------------------
    # Build retrieval statistics
    # -----------------------------------------------------------------------

    stats = RetrievalStats(

        # Number of semantic searches executed.
        search_calls=result.get(
            "search_calls",
            0,
        ),

        # Number of neighbor expansions.
        neighbor_calls=result.get(
            "neighbor_calls",
            0,
        ),

        # Number of table expansions.
        table_calls=result.get(
            "table_calls",
            0,
        ),

        # Total expansion tokens consumed.
        expansion_tokens=result.get(
            "expansion_tokens",
            0,
        ),

        # Configured maximum expansion budget.
        expansion_token_budget=s.expansion_token_budget,
    )


    # -----------------------------------------------------------------------
    # Aggregate LLM usage
    # -----------------------------------------------------------------------

    # Collect token usage across all model calls in the agent loop.
    usage = collect_usage(
        result.get(
            "messages"
        ),
        s.openai_model,
    )


    # -----------------------------------------------------------------------
    # Deduplicate captured passages
    # -----------------------------------------------------------------------

    # Dictionary keyed by chunk_id.
    #
    # The dictionary preserves the FIRST captured instance.
    unique: dict[
        str,
        dict,
    ] = {}

    # Process every passage captured during the retrieval loop.
    for p in result.get(
        "captured_passages",
        [],
    ):

        # setdefault() ensures the first occurrence wins.
        #
        # Later duplicate expansions do not overwrite the original
        # search result, including its relevance score.
        unique.setdefault(
            p["chunk_id"],
            p,
        )


    # -----------------------------------------------------------------------
    # Convert to Passage models and sort in reading order
    # -----------------------------------------------------------------------

    passages = sorted(

        # Convert raw dictionaries into validated Passage objects.
        (
            Passage(**p)
            for p in unique.values()
        ),

        # Sort by:
        #
        #     1. document ID
        #     2. position inside the document
        #
        # Missing positions are treated as zero.
        key=lambda p: (
            p.doc_id,
            p.position
            if p.position is not None
            else 0,
        ),
    )


    # -----------------------------------------------------------------------
    # Common response fields
    # -----------------------------------------------------------------------

    common = dict(

        # Retrieval statistics.
        stats=stats,

        # LLM token usage.
        usage=usage,

        # Entities identified by the model.
        entities=decision.entities,

        # Factual note produced by the model.
        result_note=decision.note,

        # Search metadata.
        searches=[
            SearchQuery(**query)
            for query in result.get(
                "searches",
                [],
            )
        ],
    )


    # -----------------------------------------------------------------------
    # No search executed
    # -----------------------------------------------------------------------

    # If the model completed without performing a search, make that
    # explicit in the structured response.
    if stats.search_calls == 0:

        return TrialSearchResponse(
            result_shape="not_executed",

            **{
                **common,

                "result_note": (
                    "No search was executed."
                ),
            },
        )


    # -----------------------------------------------------------------------
    # Model determined the question is unanswerable
    # -----------------------------------------------------------------------

    if not decision.answerable:

        return TrialSearchResponse(
            result_shape="unanswerable",
            **common,
        )


    # -----------------------------------------------------------------------
    # Search executed but produced no passages
    # -----------------------------------------------------------------------

    if not passages:

        return TrialSearchResponse(
            result_shape="empty",
            **common,
        )


    # -----------------------------------------------------------------------
    # Successful passage result
    # -----------------------------------------------------------------------

    return TrialSearchResponse(
        result_shape="passages",

        # Complete deduplicated passages are attached programmatically.
        passages=passages,

        **common,
    )


# ===========================================================================
# Public orchestration entry point
# ===========================================================================

async def orchestrate(
    question: str,
) -> TrialSearchResponse:
    """Run one bounded trial_search question.

    Flow:

        question
           │
           ▼
        connect_tools()
           │
           ▼
        build_agent()
           │
           ▼
        ainvoke()
           │
           ├── Guardrail
           ├── model
           ├── retrieval middleware
           ├── semantic_search
           ├── expand_neighbors
           ├── expand_table
           └── structured ModelDecision
           │
           ▼
        assemble()
           │
           ▼
        TrialSearchResponse


    If the guardrail blocks either the input or model-authored output,
    the turn is converted into an `unanswerable` response.
    """

    try:

        # Open the MCP connection to AgentCore Gateway.
        #
        # The connection remains alive for the duration of this question.
        async with connect_tools() as tools:

            # Build the LangGraph agent and execute exactly one user
            # question.
            #
            # The model may perform multiple bounded tool calls inside
            # this invocation.
            result = await build_agent(
                tools
            ).ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                }
            )

    except GuardrailBlocked as blocked:

        # A guardrail intervention ends the turn as unanswerable.
        #
        # The guardrail's own message is retained so the caller knows
        # why the request was blocked.
        return TrialSearchResponse(
            result_shape="unanswerable",

            result_note=(
                f"Blocked by guardrail ({blocked.source}): "
                f"{blocked.message}"
            ),
        )


    # Convert the finished LangGraph state into the public response
    # contract.
    return assemble(
        result
    )