"""Lambda handler for trial_graph's tools, invoked through AgentCore Gateway.

ARCHITECTURE
------------

The trial_graph agent does NOT connect directly to Neo4j.

Instead, the flow is:

    trial_graph agent
    (LangGraph / AgentCore Runtime)
            |
            | MCP tool call
            | HTTPS + SigV4
            v
    AgentCore Gateway
    (MCP server)
            |
            | invokes Lambda target
            v
    THIS LAMBDA
            |
            | Neo4j Bolt connection
            v
    Neo4j Aura


The Gateway acts as the MCP server.

This Lambda is simply the Gateway's target that implements the actual
tool operations.


WHY THE TOOL LOGIC LIVES HERE
-----------------------------

In the reference ACT Xerebro/NLC architecture, the LangGraph agent and
the database tools could live inside the same container.

This architecture deliberately separates them.

The agent container:

    - does not have Neo4j credentials
    - does not create a Neo4j driver
    - does not execute Cypher directly

Instead:

    Agent
       |
       | MCP
       v
    AgentCore Gateway
       |
       | IAM-authorized Lambda invocation
       v
    Lambda
       |
       | Secrets Manager credentials
       v
    Neo4j


This moves the database trust boundary away from the agent runtime.

The Gateway controls access to the Lambda, while the Lambda owns:

    - Neo4j credentials
    - Neo4j connection
    - entity search
    - Cypher validation
    - Cypher execution


TOOL DISPATCH
-------------

The Gateway exposes multiple tools but invokes the same Lambda.

The tool name is provided through the Lambda invocation context:

    context.client_context.custom[
        "bedrockAgentCoreToolName"
    ]

The event body contains only the arguments for the selected tool.

For example:

    find_entity_by_name
        event = {
            "name": "Diabetes",
            "entity_type": "disease"
        }

There is no additional MCP dispatch envelope inside the event.


WHAT THIS FILE DOES NOT DO
--------------------------

This file does NOT implement:

    - LangGraph orchestration
    - model reasoning
    - structured ModelDecision
    - Cypher repair loops
    - guardrails
    - row/graph limits
    - result summarization
    - supervisor routing

Those responsibilities belong to the agent process.

This Lambda performs only the three database operations:

    1. find_entity_by_name()
    2. validate_cypher()
    3. execute_cypher()


READ-ONLY SAFETY
----------------

The agent's CypherMiddleware is responsible for enforcing:

    - write rejection
    - LIMIT injection/clamping
    - repair budget

This Lambda trusts that upstream guard.

As defense in depth, the Neo4j credential used here should also be
read-only.
"""


# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# Used to parse the JSON secret returned by Secrets Manager and to
# serialize/deserialize the final Lambda response.
import json

# Used for Lambda logging.
import logging

# Used to read the Neo4j secret identifier from the Lambda environment.
import os


# ---------------------------------------------------------------------------
# AWS / Neo4j imports
# ---------------------------------------------------------------------------

# AWS SDK used to retrieve the Neo4j credentials from Secrets Manager.
import boto3

# Official Neo4j Python driver.
from neo4j import GraphDatabase


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

# Lambda's root logger is used so messages appear in CloudWatch Logs.
log = logging.getLogger()

# INFO provides useful operational information without enabling verbose
# DEBUG logging in production.
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Neo4j driver cache
# ---------------------------------------------------------------------------

# The Neo4j driver is intentionally stored at module level.
#
# Lambda execution environments can be reused for multiple invocations.
#
# Therefore:
#
#     Cold start
#         -> create driver
#
#     Warm invocation
#         -> reuse existing driver
#
# This avoids creating a new Neo4j driver and TLS connection for every
# tool invocation.
_driver = None


def _get_driver():
    """Return the shared Neo4j driver.

    The driver is created once per warm Lambda execution environment.

    WHY CACHE THE DRIVER
    --------------------

    Creating a fresh Neo4j driver for every invocation can result in:

        Lambda
          |
          +-- TLS handshake
          +-- authentication
          +-- connection setup
          +-- query
          |
          +-- destroy driver

    With the module-level cache:

        Cold start
          |
          +-- create driver
          |
        Warm invocation
          |
          +-- reuse driver
          |
          +-- query

    This reduces connection setup overhead.


    CREDENTIAL SOURCE
    ------------------

    The Neo4j credentials come from AWS Secrets Manager.

    The Lambda environment contains only:

        NEO4J_SECRET_ID

    It does NOT contain the actual:

        - URI
        - username
        - password

    This reduces accidental credential exposure through Lambda
    configuration inspection.
    """

    # Tell Python that we intend to modify the module-level variable.
    global _driver

    # Only create the driver during the first invocation in this
    # execution environment.
    if _driver is None:

        # Create the Secrets Manager client.
        #
        # boto3 automatically uses the Lambda execution role and
        # Lambda/AWS region configuration.
        secrets = boto3.client(
            "secretsmanager"
        )

        # Retrieve the secret referenced by the environment variable.
        #
        # The environment variable contains only the secret identifier,
        # not the actual credentials.
        secret = json.loads(
            secrets.get_secret_value(
                SecretId=os.environ[
                    "NEO4J_SECRET_ID"
                ]
            )[
                "SecretString"
            ]
        )

        # Create the Neo4j driver.
        #
        # Expected secret structure is approximately:
        #
        # {
        #     "uri": "...",
        #     "user": "neo4j",
        #     "password": "..."
        # }
        #
        # If "user" is not supplied, default to "neo4j".
        _driver = GraphDatabase.driver(
            secret["uri"],
            auth=(
                secret.get(
                    "user",
                    "neo4j",
                ),
                secret["password"],
            ),
        )

    # Return the cached driver.
    return _driver


# ===========================================================================
# Tool 1 — Entity resolution
# ===========================================================================


# ---------------------------------------------------------------------------
# Supported entity types
# ---------------------------------------------------------------------------

# Maps the logical entity type supplied by the model to the actual
# Neo4j labels that can be searched.
#
# Example:
#
#     entity_type = "trial"
#
# becomes:
#
#     labels = ["Trial"]
#
# "any" searches across all supported entity labels.
_ENTITY_LABELS = {
    "trial": ["Trial"],
    "sponsor": ["Sponsor"],
    "drug": ["Drug"],
    "disease": ["Disease"],
    "site": ["Site"],
    "cro": ["CRO"],

    # Search across all supported graph entities.
    "any": [
        "Trial",
        "Sponsor",
        "Drug",
        "Disease",
        "Site",
        "CRO",
    ],
}


# ---------------------------------------------------------------------------
# Identity properties
# ---------------------------------------------------------------------------

# Defines the property that uniquely identifies each entity type.
#
# This is important because the graph may contain another convenience
# property such as `key`, but the model must receive the actual identity
# property that can be used for an exact follow-up query.
#
# Examples:
#
#     Trial  -> nctId
#     Sponsor -> name
#     Site -> facility
_KEY_PROPERTY = {
    "Trial": "nctId",
    "Sponsor": "name",
    "Drug": "name",
    "Disease": "name",
    "Site": "facility",
    "CRO": "name",
}


# ---------------------------------------------------------------------------
# Neo4j full-text index
# ---------------------------------------------------------------------------

# Name of the Neo4j full-text index used for entity lookup.
#
# The index is expected to contain the entity names/identities needed
# for entity resolution.
_FULLTEXT_INDEX = "trial_entity_names"


def find_entity_by_name(args: dict) -> dict:
    """Find candidate graph entities matching a human-readable name.

    This is an entity-resolution tool.

    Example:

        {
            "name": "AstraZeneca",
            "entity_type": "sponsor"
        }

    The function:

        1. Reads the search name.
        2. Determines which Neo4j labels are allowed.
        3. Searches the full-text index.
        4. Returns candidate entities ranked by Neo4j score.

    IMPORTANT:
    ----------
    This function does not ask the model to guess the canonical
    identifier.

    Instead, it returns the identity property and value discovered
    from the graph.

    The model can then use that exact identity in a Cypher query.
    """

    # Human-readable entity name supplied by the model/tool caller.
    name = args[
        "name"
    ]

    # Optional entity type.
    #
    # If not provided, search all supported entity types.
    entity_type = args.get(
        "entity_type",
        "any",
    )

    # Convert the logical entity type into Neo4j labels.
    #
    # Unknown entity types fall back to searching all supported labels.
    labels = _ENTITY_LABELS.get(
        entity_type,
        _ENTITY_LABELS["any"],
    )

    # Open a Neo4j session.
    #
    # The driver itself is reused across warm Lambda invocations.
    with _get_driver().session() as session:

        # Execute the full-text search query.
        #
        # IMPORTANT:
        #
        # Neo4j parameters are passed explicitly through:
        #
        #     parameters={...}
        #
        # rather than as Python keyword arguments.
        #
        # This avoids colliding with the Neo4j driver's own Session.run()
        # function parameters.
        result = session.run(
            "CALL db.index.fulltext.queryNodes($index, $search) "
            "YIELD node, score "
            "WHERE any(l IN labels(node) WHERE l IN $labels) "
            "RETURN labels(node) AS labels, properties(node) AS props, score "
            "ORDER BY score DESC LIMIT 10",

            # All dynamic values are supplied as Cypher parameters.
            parameters={
                "index": _FULLTEXT_INDEX,

                # Escape Lucene special characters before searching.
                "search": _escape_lucene(
                    name
                ),

                # Restrict matches to the requested entity labels.
                "labels": labels,
            },
        )

        # Candidate entities returned by Neo4j.
        candidates = []

        # Iterate through the full-text search results.
        for row in result:

            # The first label is used as the entity type.
            #
            # The graph model is expected to have one primary entity label
            # for these searchable nodes.
            label = row[
                "labels"
            ][0]

            # Determine which property is the canonical identity property.
            #
            # Examples:
            #
            #     Trial -> nctId
            #     Site  -> facility
            key_prop = _KEY_PROPERTY.get(
                label,
                "name",
            )

            # Construct a normalized candidate record.
            candidates.append(
                {
                    # Neo4j entity label.
                    "label": label,

                    # Property that should be used for exact identity.
                    "property": key_prop,

                    # Canonical identity value from the graph.
                    "value": row["props"].get(
                        key_prop,
                        "",
                    ),

                    # Human-readable display name.
                    #
                    # Trials generally do not have a `name` property,
                    # so briefTitle is used as the human-readable fallback.
                    "name": (
                        row["props"].get("name")
                        or row["props"].get("facility")
                        or row["props"].get("briefTitle", "")
                    ),

                    # Full-text search relevance score.
                    "score": row[
                        "score"
                    ],
                }
            )

    # Return candidates to the MCP/Gateway layer.
    return {
        "candidates": candidates
    }


# ---------------------------------------------------------------------------
# Lucene escaping
# ---------------------------------------------------------------------------

def _escape_lucene(text: str) -> str:
    """Escape Lucene special characters.

    Neo4j's full-text search uses Lucene query syntax.

    Characters such as:

        +
        -
        &&
        ||
        !
        (
        )
        {
        }
        [
        ]
        ^
        "
        ~
        *
        ?
        :
        /

have special meaning in Lucene.

If they are passed directly from a user/model-generated entity name,
the search can fail with a Lucene syntax error rather than simply
returning no matches.

Therefore every entity-search input is escaped before it reaches
the full-text index.
"""

    # Lucene characters that need escaping.
    special = r'+-&&||!(){}[]^"~*?:\/'

    # Prefix every special character with a backslash.
    return "".join(
        f"\\{c}"
        if c in special
        else c
        for c in text
    )


# ===========================================================================
# Tool 2 — Cypher validation
# ===========================================================================

def validate_cypher(args: dict) -> dict:
    """Validate a Cypher query without executing it.

    The query is wrapped with:

        EXPLAIN <query>

    Neo4j parses/plans the query without actually running it.

    This is used by the agent's Cypher repair loop:

        model-generated Cypher
                |
                v
        CypherMiddleware
                |
                v
        validate_cypher
                |
          ┌─────┴─────┐
          │           │
        valid       invalid
          │           │
          v           v
       execute     repair
    """

    # Extract the Cypher query supplied by the caller.
    query = args[
        "query"
    ]

    # Open a Neo4j session.
    with _get_driver().session() as session:

        try:

            # EXPLAIN validates/plans the query without executing it.
            session.run(
                f"EXPLAIN {query}"
            )

            # Successful EXPLAIN means the query is syntactically/
            # semantically acceptable to Neo4j.
            return {
                "valid": True
            }

        except Exception as exc:

            # Return a compact error to the model.
            #
            # The error is capped at 500 characters so an unexpected
            # Neo4j error cannot flood the model context.
            return {
                "valid": False,
                "error": str(exc)[
                    :500
                ],
            }


# ===========================================================================
# Tool 3 — Cypher execution
# ===========================================================================

def execute_cypher(args: dict) -> dict:
    """Execute an already-validated Cypher query.

    The actual safety controls are expected to have happened upstream
    in CypherMiddleware.

    That middleware is responsible for things such as:

        - rejecting write queries
        - injecting/clamping LIMIT
        - enforcing repair budgets

    This function therefore focuses on executing the query and converting
    Neo4j's raw driver objects into a serializable result shape.
    """

    # Extract the Cypher query.
    query = args[
        "query"
    ]

    # Open a Neo4j session.
    with _get_driver().session() as session:

        try:

            # Execute the Cypher query.
            result = session.run(
                query
            )

            # Materialize all returned records.
            #
            # Each Neo4j Record is converted into a normal Python dict.
            records = [
                dict(record)
                for record in result
            ]

        except Exception as exc:

            # Do not expose a potentially huge raw exception.
            #
            # Return enough information for the agent to understand
            # whether another repair attempt may be useful.
            return {
                "error": True,
                "error_class": type(exc).__name__,
                "detail": str(exc)[
                    :500
                ],
            }

    # Convert raw Neo4j records into the graph/table contract expected
    # by the trial_graph agent.
    return _shape_result(
        records
    )


# ===========================================================================
# Neo4j result shaping
# ===========================================================================

def _shape_result(records: list[dict]) -> dict:
    """Convert raw Neo4j records into the agent's result contract.

    Neo4j can return two broad kinds of results.

    GRAPH-SHAPED
    ------------

        RETURN n, r, p

    may produce Neo4j:

        Node
        Relationship
        Path

    objects.

    TABLE-SHAPED
    ------------

        RETURN n.nctId AS trial_id, n.status AS status

    produces ordinary scalar values.

    This function detects which type was returned and converts it into
    serializable dictionaries.

    The resulting structure matches what trial_graph/core.py expects.
    """

    # No records means the query executed successfully but returned
    # nothing.
    if not records:
        return {
            "result_shape": "empty"
        }

    # Containers for discovered graph objects.
    nodes = []
    rels = []

    # Sets prevent duplicate nodes/relationships when the same graph
    # element appears in multiple returned records.
    seen_nodes = set()
    seen_rels = set()


    def _visit(value):
        """Recursively inspect a Neo4j result value.

        The Neo4j Python driver can return:

            Node
            Relationship
            Path
            scalar values
            lists
            dictionaries

        Graph objects are extracted recursively so that a returned Path,
        for example, contributes both its nodes and relationships.
        """

        # ---------------------------------------------------------------
        # Neo4j Node
        # ---------------------------------------------------------------
        #
        # Nodes expose:
        #
        #     labels
        #     items()
        #     element_id
        #
        # This combination is used to identify node objects.
        if (
            hasattr(value, "labels")
            and hasattr(value, "items")
        ):

            # Avoid adding the same node more than once.
            if value.element_id not in seen_nodes:

                seen_nodes.add(
                    value.element_id
                )

                # Convert the Neo4j Node into the application's
                # serializable GraphNode-like structure.
                nodes.append(
                    {
                        "element_id": value.element_id,
                        "labels": list(
                            value.labels
                        ),
                        "properties": dict(
                            value.items()
                        ),
                    }
                )


        # ---------------------------------------------------------------
        # Neo4j Relationship
        # ---------------------------------------------------------------
        #
        # Relationships expose:
        #
        #     type
        #     nodes
        #     element_id
        #     items()
        #
        # A relationship also points to its start and end nodes.
        elif (
            hasattr(value, "type")
            and hasattr(value, "nodes")
        ):

            # Avoid duplicate relationships.
            if value.element_id not in seen_rels:

                seen_rels.add(
                    value.element_id
                )

                # Neo4j exposes the relationship's endpoint nodes.
                start, end = value.nodes

                # Store the relationship itself.
                rels.append(
                    {
                        "element_id": value.element_id,
                        "type": value.type,

                        # Use node element IDs to describe the edge.
                        "start": start.element_id,
                        "end": end.element_id,

                        # Convert relationship properties to a normal dict.
                        "properties": dict(
                            value.items()
                        ),
                    }
                )

                # Also visit the endpoint nodes.
                #
                # This ensures a query returning only a relationship still
                # produces the corresponding nodes in the graph result.
                _visit(start)
                _visit(end)


        # ---------------------------------------------------------------
        # Neo4j Path
        # ---------------------------------------------------------------
        #
        # A Path contains:
        #
        #     path.nodes
        #     path.relationships
        #
        # Visit every component recursively.
        elif (
            hasattr(value, "relationships")
            and hasattr(value, "nodes")
        ):

            # Extract all nodes contained in the path.
            for node in value.nodes:
                _visit(node)

            # Extract all relationships contained in the path.
            for relationship in value.relationships:
                _visit(relationship)


    # -------------------------------------------------------------------
    # Inspect every returned record
    # -------------------------------------------------------------------

    # Each record may contain multiple returned values:
    #
    #     RETURN trial, sponsor, relationship
    #
    # Therefore every value in every record is inspected.
    for record in records:

        for value in record.values():

            _visit(
                value
            )


    # -------------------------------------------------------------------
    # Graph result
    # -------------------------------------------------------------------

    # If at least one Node or Relationship was discovered, treat the
    # entire result as graph-shaped.
    if nodes or rels:

        return {
            "result_shape": "graph",
            "nodes": nodes,
            "relationships": rels,
        }


    # -------------------------------------------------------------------
    # Table result
    # -------------------------------------------------------------------

    # No graph objects were found.
    #
    # Therefore the query returned scalar/structured table values.
    #
    # Use the first record to establish the column ordering.
    columns = list(
        records[0].keys()
    )

    # Convert every record into a list whose values follow the same
    # column ordering.
    rows = [
        [
            record[column]
            for column in columns
        ]
        for record in records
    ]

    return {
        "result_shape": "table",
        "columns": columns,
        "rows": rows,
    }


# ===========================================================================
# Lambda tool dispatch
# ===========================================================================

# Map the tool name supplied by AgentCore Gateway to the actual Python
# function that implements it.
#
# The Gateway invokes the same Lambda for all three tools.
#
# This dictionary is therefore the local dispatch table.
_TOOLS = {
    "find_entity_by_name": find_entity_by_name,
    "validate_cypher": validate_cypher,
    "execute_cypher": execute_cypher,
}


def lambda_handler(
    event,
    context,
):
    """Entry point invoked by AWS Lambda.

    The AgentCore Gateway calls this same Lambda for all three tools.

    The tool name is NOT expected in `event`.

    Instead, the Gateway provides it through:

        context.client_context.custom[
            "bedrockAgentCoreToolName"
        ]

    The `event` itself contains the selected tool's input arguments.

    Flow:

        Agent
          |
          | MCP
          v
        Gateway
          |
          | tool name -> Lambda context
          | arguments -> Lambda event
          v
        lambda_handler()
          |
          v
        _TOOLS[matched](event)
          |
          v
        Neo4j
    """

    # Attempt to retrieve Lambda's client context.
    #
    # getattr(..., None) keeps this defensive in case the context object
    # does not contain client_context.
    tool_name = getattr(
        context,
        "client_context",
        None,
    )

    # Extract the AgentCore Gateway tool name.
    #
    # If client_context/custom is missing, use an empty string.
    tool_name = (
        tool_name.custom.get(
            "bedrockAgentCoreToolName",
            "",
        )
        if tool_name and tool_name.custom
        else ""
    )

    # -------------------------------------------------------------------
    # Resolve the Gateway tool name to one of our local functions
    # -------------------------------------------------------------------

    # AgentCore Gateway may prefix the tool name with the target name.
    #
    # Example:
    #
    #     LambdaTarget___execute_cypher
    #
    # Our local tool name is only:
    #
    #     execute_cypher
    #
    # Therefore exact equality is intentionally NOT used.
    #
    # suffix matching allows both:
    #
    #     execute_cypher
    #
    # and:
    #
    #     LambdaTarget___execute_cypher
    #
    # to resolve to the same function.
    matched = next(
        (
            name
            for name in _TOOLS
            if tool_name.endswith(name)
        ),
        None,
    )

    # Unknown tool names should fail cleanly rather than accidentally
    # executing the wrong operation.
    if matched is None:

        log.error(
            "unknown tool requested: %r",
            tool_name,
        )

        return {
            "error": True,
            "detail": f"unknown tool: {tool_name}",
        }

    # Log which tool is being dispatched.
    log.info(
        "dispatching to %s",
        matched,
    )

    try:

        # Execute the selected tool.
        #
        # `event` contains the arguments defined by that tool's schema.
        result = _TOOLS[
            matched
        ](
            event
        )

    except KeyError as exc:

        # A missing required input argument reaches this branch.
        #
        # Example:
        #
        #     execute_cypher({})
        #
        # would raise KeyError("query").
        return {
            "error": True,
            "detail": f"missing required argument: {exc}",
        }


    # -------------------------------------------------------------------
    # Final Lambda serialization boundary
    # -------------------------------------------------------------------

    # Lambda serializes the returned Python object as JSON.
    #
    # Neo4j can return types that Python's standard json encoder does
    # not understand, for example:
    #
    #     neo4j.time.Date
    #     neo4j.time.DateTime
    #     neo4j.spatial.Point
    #
    # `default=str` converts unsupported Neo4j values to strings.
    #
    # The json.loads() afterwards converts the serialized JSON string
    # back into a normal Python JSON-compatible object before returning
    # it to the Lambda runtime.
    #
    # This means serialization errors are handled here, immediately after
    # successful query execution, instead of causing the Lambda invocation
    # to fail after Neo4j already completed the query.
    return json.loads(
        json.dumps(
            result,
            default=str,
        )
    )