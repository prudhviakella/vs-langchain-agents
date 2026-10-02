"""trial_graph core: the agent loop and the middleware that bounds it.

High-level workflow:

    orchestrate(question)
        │
        ├─ connect_tools()
        │      │
        │      └─ Creates a SigV4-signed MCP connection to the
        │         AgentCore Gateway.
        │
        │         Gateway exposes:
        │           - find_entity_by_name
        │           - validate_cypher
        │           - execute_cypher
        │
        ├─ build_agent(tools)
        │      │
        │      └─ Creates the LangChain agent.
        │         The model must produce a ModelDecision using
        │         ToolStrategy.
        │
        │         Middleware:
        │           1. GuardrailMiddleware
        │           2. CypherMiddleware
        │
        │         CypherMiddleware:
        │           - rejects write operations
        │           - injects/clamps LIMIT
        │           - tracks query execution
        │           - tracks failed queries
        │           - stores complete results in agent state
        │           - exposes only a compact result summary to the model
        │
        └─ assemble(result)
               │
               └─ Converts the final agent STATE into
                  TrialGraphResponse.

IMPORTANT DESIGN PRINCIPLE:

    The LLM does NOT become the source of truth.

    Neo4j result
        ↓
    Lambda
        ↓
    Agent state
        ↓
    TrialGraphResponse

    The LLM receives only a compact summary of the result.
    This prevents the model from retyping large result sets and
    accidentally changing values before they are used in later steps.


WHERE EVERYTHING COMES FROM

    model
        OpenAI chat model.
        Model name and credentials are loaded from configuration.

    system prompt
        Bedrock Prompt Management.
        The prompt version is pinned through Parameter Store.

    limits
        Parameter Store:
            - row_cap
            - max_repairs
            - graph_node_cap

    guardrail
        Amazon Bedrock Guardrail through ApplyGuardrail.

    Configuration is loaded through config.settings().


LOOP ENGINEERING — HOW THE LOOP IS BOUNDED

    writes
        Rejected before the query reaches the Gateway.

    row cap
        LIMIT is automatically injected when missing.
        LIMIT is reduced when the model asks for more than row_cap.

    repair budget
        Failed execute_cypher calls increment repair_count.
        Once the configured limit is reached, execute_cypher
        is refused.

    result handling
        The complete result is stored in state.
        The model sees only a compact summary.

    These are runtime controls implemented in middleware.
    They are not merely instructions in the system prompt.


IMPORTANT RUNTIME FACTS

    1. Middleware custom state must be declared with state_schema
       using a subclass of AgentState.

    2. Both synchronous and asynchronous tool middleware methods
       are implemented because the agent is invoked with ainvoke().

    3. AgentCore Gateway tool names may contain a target prefix,
       for example:

           trial-graph-tools___execute_cypher

       Therefore tool matching uses suffix matching rather than
       exact string matching.

    4. MCP tool results can arrive either as:
           - structured_content inside an artifact
           - dictionary content
           - list of text blocks

       _payload() normalizes all of these into a Python dictionary.

    5. ToolStrategy is used instead of a bare response schema.
       This allows the model to continue making tool calls while
       still producing the final structured ModelDecision.


WHY THE MODEL NEVER SEES ALL ROWS

    Neo4j may return a large result.

    Instead of sending the entire result back to the model:

        Neo4j
          ↓
        complete result
          ↓
        state["captured"]
          ↓
        summarize()
          ↓
        model

    The model sees:
        - result counts
        - column names
        - up to a few sample rows

    The complete result remains in state and is eventually returned
    through TrialGraphResponse.


WHAT THIS DOES NOT DO

    - It does not write to Neo4j.
    - It does not rank or reorder Neo4j rows.
    - It does not allow the model to bypass the row limit.
    - It does not trust the model's generated result values as
      the source of truth.

"""

from __future__ import annotations

import json
import logging
import operator
import re
from contextlib import asynccontextmanager
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .config import settings
from .guardrail import GuardrailBlocked, GuardrailMiddleware
from .schemas import ModelDecision, TrialGraphResponse, collect_usage
from .tools import summarize


# Logger used for debugging and operational visibility.
log = logging.getLogger("agent.trial_graph.core")


# These are the actual MCP/Gateway tool names that this agent cares about.
#
# The Gateway may add a prefix to the actual tool name, therefore we
# don't compare the incoming name directly. _kind() below checks
# whether the name ends with one of these values.
RESOLVE = "find_entity_by_name"
VALIDATE = "validate_cypher"
EXECUTE = "execute_cypher"


# Cypher operations that can modify the database.
#
# Trial Graph is intentionally READ-ONLY.
#
# Whole-word matching is used so a legitimate trial/entity name such as
# "Creating a Registry" does not accidentally match CREATE.
_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s*\{[^}]*\bCREATE\b)\b",
    re.IGNORECASE
)


# Used to find a LIMIT at the end of a Cypher query.
#
# Examples:
#
#     LIMIT 10
#     LIMIT 100;
#
# This allows us to inspect and potentially reduce a model-generated LIMIT.
_LIMIT = re.compile(
    r"\bLIMIT\s+(\d+)\s*;?\s*$",
    re.IGNORECASE
)


def _kind(tool_name: str) -> str | None:
    """Identify which Trial Graph tool is being called.

    AgentCore Gateway may prefix the original tool name.

    Example:

        trial-graph-tools___execute_cypher

    should still be recognized as:

        execute_cypher

    Therefore suffix matching is intentionally used here.
    """

    return next(
        (
            kind
            for kind in (RESOLVE, VALIDATE, EXECUTE)
            if tool_name.endswith(kind)
        ),
        None
    )


@asynccontextmanager
async def connect_tools():
    """Create the authenticated MCP connection to AgentCore Gateway.

    Flow:

        Trial Graph Agent
              │
              │ SigV4
              ▼
        AgentCore Gateway
              │
              ▼
        MCP tools

    The Gateway exposes the Neo4j-related tools to the agent.

    The AWS service name MUST be "bedrock-agentcore" because that is
    the signing name expected by the Gateway.
    """

    # Imports are kept inside the function so that the module itself
    # does not require these packages until the MCP connection is needed.
    from mcp import ClientSession
    from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
    from langchain_mcp_adapters.tools import load_mcp_tools

    # Load application configuration.
    s = settings()

    # Create a SigV4-authenticated HTTP connection to AgentCore Gateway.
    async with aws_iam_streamablehttp_client(
        endpoint=s.gateway_url,
        aws_service="bedrock-agentcore",
        aws_region=s.region
    ) as (read, write, _):

        # Create the MCP client session over the authenticated connection.
        async with ClientSession(read, write) as session:

            # Perform MCP initialization/handshake.
            await session.initialize()

            # Convert MCP tools into LangChain-compatible tools.
            yield await load_mcp_tools(session)


class CypherState(AgentState):
    """Additional state maintained by CypherMiddleware.

    AgentState already contains the normal LangChain agent state.

    These fields are added specifically for Trial Graph.

    Reducers:

        repair_count
        execute_calls

    use operator.add, meaning:

        current_value + update_value

    This is important because middleware can return:

        {"repair_count": 1}

    and the value is accumulated rather than blindly overwritten.
    """

    # Number of failed execute_cypher attempts.
    #
    # Reducer:
    #     0 + 1 + 1 + ...
    repair_count: Annotated[int, operator.add]

    # Number of successful execute_cypher calls.
    execute_calls: Annotated[int, operator.add]

    # Complete payload returned by the latest successful execution.
    #
    # This stays in state rather than being sent in full to the LLM.
    captured: dict

    # Exact Cypher query that was actually executed.
    #
    # This may differ from the original model-generated query because
    # CypherMiddleware can inject or reduce LIMIT.
    executed_cypher: str


def _payload(result) -> dict | None:
    """Extract the Lambda JSON payload from an MCP ToolMessage.

    Depending on how langchain_mcp_adapters receives the MCP response,
    the structured payload may appear in different locations.

    Supported formats:

        1. result.artifact["structured_content"]

        2. result.content as a dictionary

        3. result.content as a list of text blocks

        4. result.content as JSON text

    The function normalizes all of them into:

        dict | None

    Returning None means the tool response could not be understood.
    """

    # Only ToolMessage results are expected here.
    if not isinstance(result, ToolMessage):
        return None

    # Some MCP responses expose structured content through artifact.
    artifact = getattr(result, "artifact", None)

    if (
        isinstance(artifact, dict)
        and isinstance(artifact.get("structured_content"), dict)
    ):
        return artifact["structured_content"]

    # Some adapters may provide the content directly as a dictionary.
    content = result.content

    if isinstance(content, dict):
        return content

    # MCP text responses may arrive as a list of blocks:
    #
    # [
    #     {"type": "text", "text": "..."}
    # ]
    #
    # Combine all text blocks before parsing JSON.
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    # Convert JSON text into a Python object.
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None

    # Only dictionaries are valid Trial Graph tool payloads.
    return parsed if isinstance(parsed, dict) else None


class CypherMiddleware(AgentMiddleware):
    """Guard and record all Trial Graph Cypher tool calls.

    This middleware is the main runtime safety/control layer.

    BEFORE the tool executes:

        - identify the tool
        - enforce repair budget
        - reject write operations
        - enforce row limit

    AFTER the tool executes:

        - parse the tool response
        - convert errors into model-readable ToolMessages
        - save successful query results into state
        - send only a compact summary to the model
    """

    # Tell LangChain that this middleware owns CypherState.
    state_schema = CypherState

    def __init__(self, cfg=None):
        super().__init__()

        # Allow tests to inject configuration.
        # Otherwise load the real application configuration.
        self.cfg = cfg or settings()

    def wrap_tool_call(self, request, handler):
        """Synchronous tool-call middleware.

        The same logic is shared with the asynchronous implementation.

        Flow:

            model tool call
                ↓
            _gate()
                ↓
            blocked OR modified request
                ↓
            actual tool
                ↓
            _record()
        """

        # Step1
        # Input validation
        # Apply all BEFORE-execution controls.
        gated = self._gate(request)

        # If _gate() returned a ToolMessage, the call was blocked.
        if isinstance(gated, ToolMessage):
            return gated

        # Step2
        # Execute Tool
        response = handler(gated)

        #Step3
        # Otherwise execute the tool and process its result.
        return self._record(gated, response)

    async def awrap_tool_call(self, request, handler):
        """Asynchronous version used by ainvoke()."""

        # Apply exactly the same BEFORE-execution controls.
        gated = self._gate(request)

        # A ToolMessage means middleware rejected the call.
        if isinstance(gated, ToolMessage):
            return gated

        # Execute asynchronously and process the result.
        return self._record(
            gated,
            await handler(gated)
        )

    # ──────────────────────────────────────────────────────────────────
    # BEFORE TOOL EXECUTION
    # ──────────────────────────────────────────────────────────────────

    def _gate(self, request):
        """Apply runtime controls before a tool reaches the Gateway.

        Controls:

            1. Repair budget
            2. Write protection
            3. Row-limit enforcement
        """

        # Determine whether this is resolve, validate, or execute.
        kind = _kind(request.tool_call["name"])

        # Only validate and execute require Cypher-specific protection.
        if kind not in (VALIDATE, EXECUTE):
            return request

        # Read the current agent state.
        state = request.state or {}

        # Unique identifier used to associate middleware messages
        # with the original tool call.
        call_id = request.tool_call["id"]

        # Copy tool arguments so we can safely modify the query.
        args = dict(request.tool_call["args"])

        # Extract the model-generated Cypher.
        query = str(args.get("query", ""))

        # ──────────────────────────────────────────────────────────────
        # STEP 1 — REPAIR BUDGET
        # ──────────────────────────────────────────────────────────────

        if kind == EXECUTE:

            # Number of previous failed executions.
            used = state.get("repair_count", 0)

            # Once the configured repair budget is exhausted,
            # don't allow another query to reach Neo4j.
            if used >= self.cfg.max_repairs:

                return ToolMessage(
                    tool_call_id=call_id,
                    content=(
                        f"REFUSED: {used} queries have already failed "
                        f"(limit {self.cfg.max_repairs}). "
                        "Stop rewriting. Set answerable=false and explain "
                        "in `note` — an honest gap is the correct outcome here."
                    )
                )

        # ──────────────────────────────────────────────────────────────
        # STEP 2 — WRITE PROTECTION
        # ──────────────────────────────────────────────────────────────

        # Search the query for any Cypher mutation keyword.
        if found := _WRITE.search(query):

            return ToolMessage(
                tool_call_id=call_id,
                content=(
                    f"REJECTED: {found.group(0).upper()} is a write operation "
                    "and this graph is read-only. "
                    "Rewrite as MATCH ... RETURN. "
                    "This does not count against your repair budget."
                )
            )

        # ──────────────────────────────────────────────────────────────
        # STEP 3 — ROW LIMIT
        # ──────────────────────────────────────────────────────────────

        if kind == EXECUTE:

            # Check whether the model already supplied a LIMIT.
            match = _LIMIT.search(query)

            if match is None:

                # No LIMIT was supplied.
                #
                # Add the configured row cap automatically.
                #
                # Example:
                #
                #     MATCH (t:Trial) RETURN t
                #
                # becomes:
                #
                #     MATCH (t:Trial) RETURN t LIMIT 100
                query = (
                    f"{query.rstrip().rstrip(';')} "
                    f"LIMIT {self.cfg.row_cap}"
                )

            elif int(match.group(1)) > self.cfg.row_cap:

                # The model requested too many rows.
                #
                # Example:
                #
                #     LIMIT 10000
                #
                # becomes:
                #
                #     LIMIT 100
                query = _LIMIT.sub(
                    f"LIMIT {self.cfg.row_cap}",
                    query
                )

            # Store the modified query back into the tool arguments.
            args["query"] = query

            # Return a modified tool request.
            #
            # The actual Gateway receives this bounded query,
            # not the original model-generated query.
            return request.override(
                tool_call={
                    **request.tool_call,
                    "args": args
                }
            )

        # validate_cypher reaches this point without modification.
        return request

    # ──────────────────────────────────────────────────────────────────
    # AFTER TOOL EXECUTION
    # ──────────────────────────────────────────────────────────────────

    def _record(self, request, result):
        """Process the result after a Gateway tool call.

        Different tools have different responsibilities:

            find_entity_by_name
                → return resolved entity candidates

            validate_cypher
                → tell the model whether Cypher is valid

            execute_cypher
                → save complete result in state and return a summary
        """

        # Identify the actual Trial Graph tool.
        kind = _kind(request.tool_call["name"])

        # If another middleware already returned a Command,
        # don't process it again.
        if kind is None or isinstance(result, Command):
            return result

        # Tool-call identifier used when creating a ToolMessage.
        call_id = request.tool_call["id"]

        # Normalize the MCP response into a dictionary.
        payload = _payload(result)

        # ──────────────────────────────────────────────────────────────
        # UNREADABLE RESPONSE
        # ──────────────────────────────────────────────────────────────

        # An unreadable response is treated as a failure.
        #
        # We do NOT silently continue because the model could otherwise
        # assume the tool succeeded.
        if payload is None:

            return Command(
                update={
                    # Only execute failures consume repair budget.
                    "repair_count": 1 if kind == EXECUTE else 0,

                    # Tell the model exactly what happened.
                    "messages": [
                        ToolMessage(
                            tool_call_id=call_id,
                            content=(
                                f"ERROR from {kind}: unreadable response."
                            )
                        )
                    ]
                }
            )

        # ──────────────────────────────────────────────────────────────
        # TOOL RETURNED AN ERROR
        # ──────────────────────────────────────────────────────────────

        if payload.get("error"):

            detail = payload.get(
                "detail",
                "unknown error"
            )

            # Execution errors tell the model to repair the query.
            hint = " Rewrite the query." if kind == EXECUTE else ""

            return Command(
                update={
                    # Failed execute calls consume repair budget.
                    "repair_count": 1 if kind == EXECUTE else 0,

                    "messages": [
                        ToolMessage(
                            tool_call_id=call_id,
                            content=(
                                f"{payload.get('error_class', 'Error')}: "
                                f"{detail}{hint}"
                            )
                        )
                    ]
                }
            )

        # ──────────────────────────────────────────────────────────────
        # ENTITY RESOLUTION
        # ──────────────────────────────────────────────────────────────

        if kind == RESOLVE:

            # Convert the entity-resolution payload into a compact
            # message that the LLM can use to construct its next MATCH.
            return ToolMessage(
                tool_call_id=call_id,
                content=self._resolved(payload)
            )

        # ──────────────────────────────────────────────────────────────
        # CYPHER VALIDATION
        # ──────────────────────────────────────────────────────────────

        if kind == VALIDATE:

            # Validation succeeded.
            if payload.get("valid"):

                message = (
                    "VALID — the query parses and every label and property "
                    "in it exists."
                )

            # Validation failed.
            else:

                message = (
                    f"INVALID: "
                    f"{payload.get('error', 'unknown')}"
                )

            # Send only the validation result back to the model.
            return ToolMessage(
                tool_call_id=call_id,
                content=message
            )

        # ──────────────────────────────────────────────────────────────
        # CYPHER EXECUTION
        # ──────────────────────────────────────────────────────────────

        # The complete Neo4j result is stored in state.
        #
        # The model does NOT receive all rows.
        #
        # Instead it receives summarize(payload, ...).
        return Command(
            update={
                # Count successful execution calls.
                "execute_calls": 1,

                # Preserve the complete result for final assembly.
                "captured": payload,

                # Record the exact Cypher that was actually executed.
                "executed_cypher": (
                    request.tool_call["args"].get("query", "")
                ),

                # Give the model only a compact summary.
                "messages": [
                    ToolMessage(
                        tool_call_id=call_id,
                        content=summarize(
                            payload,
                            self.cfg
                        )
                    )
                ]
            }
        )

    def _resolved(self, payload: dict) -> str:
        """Convert entity-resolution results into an LLM-readable message.

        The entity lookup returns candidate nodes.

        The important part is that the model is told to use the indexed
        identity property exactly as returned.

        This avoids unreliable free-text matching such as:

            WHERE name CONTAINS "..."

        """

        candidates = payload.get("candidates", [])

        # No entity was found.
        if not candidates:

            return (
                "NO_MATCH: no entity matches that name. "
                "It is not in this graph under that spelling. "
                "Do NOT retry with CONTAINS or a wildcard — "
                "report the gap honestly in `note`."
            )

        # Start with the number of candidates.
        lines = [
            f"{len(candidates)} candidate(s), best first:"
        ]

        # Add each candidate in a form that can be copied into
        # the next Cypher MATCH statement.
        lines += [
            (
                f"  (:{c['label']} "
                f"{{{c['property']}: {json.dumps(c['value'])}}})  "
                f"name={c['name']}  score={c['score']:.2f}"
            )
            for c in candidates
        ]

        # Explicitly tell the model how to use the identity.
        #
        # We intentionally don't truncate these values because identifiers
        # such as NCT numbers or document IDs must be copied exactly.
        lines.append(
            "Anchor your MATCH on the pattern shown, exactly — "
            "that property is the indexed identity. "
            "Do not filter on the raw name, and do not use the node's "
            "`key` property."
        )

        return "\n".join(lines)


def build_agent(tools: list, model=None, cfg=None):
    """Create the Trial Graph LangChain agent.

    The agent contains:

        LLM
         │
         ├── find_entity_by_name
         ├── validate_cypher
         └── execute_cypher
                │
                ▼
        CypherMiddleware

    Structured final output:

        ModelDecision

    Middleware order matters:

        GuardrailMiddleware
                ↓
        CypherMiddleware

    Therefore guardrail checks happen around the model interaction,
    while CypherMiddleware controls the actual database tool calls.
    """

    # Use injected config during tests; otherwise load production config.
    s = cfg or settings()

    return create_agent(
        name="trial_graph",

        # Use injected test model or the configured OpenAI model.
        model=model or s.chat_model(),

        # MCP tools loaded from AgentCore Gateway.
        tools=tools,

        # System prompt retrieved from the configured prompt source.
        system_prompt=s.system_prompt,

        # ToolStrategy allows the model to continue using tools and
        # eventually produce a structured ModelDecision.
        response_format=ToolStrategy(ModelDecision),

        # Middleware order is deliberate.
        middleware=[
            GuardrailMiddleware(
                s.guardrail_id,
                s.guardrail_version,
                decision_tool="ModelDecision"
            ),

            CypherMiddleware(s)
        ]
    )


def assemble(result: dict, cfg=None) -> TrialGraphResponse:
    """Convert the finished LangChain agent state into TrialGraphResponse.

    The final result is determined primarily from STATE, not simply
    from what the model says.

    Decision order:

        1. No execute call
              → not_executed

        2. Query executed but model says unanswerable
              → unanswerable

        3. Otherwise use Neo4j's reported result shape
              → graph / table / empty
    """

    # Load configuration.
    s = cfg or settings()

    # Retrieve the structured decision produced by ToolStrategy.
    decision: ModelDecision = result["structured_response"]

    # Complete execution result captured by CypherMiddleware.
    captured = result.get("captured") or {}

    # Extract token/cost usage from the agent messages.
    usage = collect_usage(
        result.get("messages"),
        s.openai_model
    )

    # Common fields returned regardless of result shape.
    common = dict(
        usage=usage,
        entities=decision.entities,
        cypher=result.get("executed_cypher", "")
    )

    # ──────────────────────────────────────────────────────────────────
    # STEP 1 — NOTHING WAS EXECUTED
    # ──────────────────────────────────────────────────────────────────

    # Even if the model says "answerable=true", we don't trust that
    # declaration if no query was actually executed.
    if not result.get("execute_calls"):

        return TrialGraphResponse(
            result_shape="not_executed",
            result_note="No query was executed.",
            **common
        )

    # ──────────────────────────────────────────────────────────────────
    # STEP 2 — MODEL DECLARED THE REQUEST UNANSWERABLE
    # ──────────────────────────────────────────────────────────────────

    if not decision.answerable:

        return TrialGraphResponse(
            result_shape="unanswerable",
            result_note=decision.note,
            **common
        )

    # ──────────────────────────────────────────────────────────────────
    # STEP 3 — USE THE ACTUAL NEO4J RESULT SHAPE
    # ──────────────────────────────────────────────────────────────────

    # Lambda determines whether the result is:
    #
    #     graph
    #     table
    #     empty
    #
    # Never blindly trust an unexpected value.
    shape = captured.get(
        "result_shape",
        "empty"
    )

    # Start with the model's note.
    note = decision.note

    # If the result was capped, make that explicit to the caller.
    if captured.get("truncated"):

        note = (
            (note + " ") if note else ""
        ) + (
            f"Result capped at {s.row_cap} rows."
        )

    return TrialGraphResponse(
        result_shape=(
            shape
            if shape in ("graph", "table", "empty")
            else "empty"
        ),

        # Graph data.
        nodes=captured.get("nodes", []),
        relationships=captured.get("relationships", []),

        # Table data.
        columns=captured.get("columns", []),
        rows=captured.get("rows", []),

        # Human/model-readable note.
        result_note=note,

        **common
    )


async def orchestrate(question: str) -> TrialGraphResponse:
    """Execute one complete Trial Graph request.

    End-to-end flow:

        Question
           │
           ▼
        connect_tools()
           │
           ▼
        build_agent()
           │
           ▼
        agent.ainvoke()
           │
           ├── LLM
           ├── entity resolution
           ├── Cypher generation
           ├── validation
           ├── execution
           ├── repair if required
           └── ModelDecision
           │
           ▼
        assemble()
           │
           ▼
        TrialGraphResponse

    The loop is bounded by middleware controls.
    """

    try:

        # Open a SigV4-authenticated MCP connection to AgentCore Gateway.
        async with connect_tools() as tools:

            # Create the Trial Graph agent and execute one user question.
            #
            # The agent may perform multiple tool calls internally,
            # but middleware controls the loop.
            result = await build_agent(tools).ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }
            )

    except GuardrailBlocked as blocked:

        # If the input/output guardrail blocks the interaction,
        # return a controlled unanswerable response instead of
        # allowing the workflow to continue.
        return TrialGraphResponse(
            result_shape="unanswerable",
            result_note=(
                f"Blocked by guardrail ({blocked.source}): "
                f"{blocked.message}"
            )
        )

    # Convert the finished agent state into the API/domain response.
    return assemble(result)