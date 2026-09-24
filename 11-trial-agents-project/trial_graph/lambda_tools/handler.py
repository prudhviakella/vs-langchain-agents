"""Lambda handler for trial_graph's tools, invoked via AgentCore Gateway's
MCP target.

    trial_graph agent (LangGraph, in AgentCore Runtime)
        |
        | MCP tool call, over the Gateway's HTTPS endpoint
        v
    AgentCore Gateway  (this is the MCP server — we do not implement
        |               MCP ourselves; the Gateway does)
        v
    THIS LAMBDA         (the Gateway's target — one function, three tools,
        |                dispatched by name)
        v
    Neo4j (Aura)

WHY THE TOOL LOGIC LIVES HERE AND NOT IN THE AGENT'S OWN PROCESS

This is a deliberate redirect from how NLC/NLQ actually work in the
reference ACT Xerebro system — there, tools call Neo4j/OpenSearch directly,
in-process, inside the same container as the LangGraph loop. Here, the
agent container has no direct Neo4j credentials at all; it only knows how
to speak MCP to the Gateway. The Gateway is what's IAM-authorized to
invoke this Lambda, and this Lambda is what holds the Neo4j credentials
(via Secrets Manager) and the query logic. This is a genuine architecture
choice, not a smaller version of the same thing — it moves the trust
boundary from "the agent container" to "the Gateway + this Lambda."

HOW THE GATEWAY TELLS THIS FUNCTION WHICH TOOL WAS CALLED

The Gateway invokes one Lambda for every tool defined on its target. Which
tool the caller actually meant arrives on the Lambda context object, not
in the event body: `context.client_context.custom['bedrockAgentCoreToolName']`.
The event body itself is just that tool's own arguments, matching its
declared inputSchema — there is no dispatch envelope to unwrap.

WHAT THIS DOES NOT DO

    - It does not do the LangGraph loop, the middleware stack, or the
      structured-output decision. All of that lives in the agent's own
      core.py, running in AgentCore Runtime, calling THIS through MCP.
      This file is pure tool execution — three Neo4j operations, nothing
      about how or when the model decides to call them.
    - It does not enforce the read-only/LIMIT/EXPLAIN guarantees. Those
      belong to CypherGuardMiddleware, running in the agent process
      BEFORE a query ever reaches the Gateway. This function trusts that
      the query it receives has already been guarded — it does not
      re-validate write-safety itself, though it does still run against
      a read-only Neo4j credential as defense in depth.
"""
import json
import logging
import os

import boto3
from neo4j import GraphDatabase

log = logging.getLogger()
log.setLevel(logging.INFO)

_driver = None  # module-level: reused across warm Lambda invocations


def _get_driver():
    """Created once per warm Lambda container, not per invocation — a
    fresh Neo4j driver per call would pay a new TLS handshake every time,
    on a Lambda that may be invoked many times per minute.

    Credentials come from Secrets Manager, read once per cold start —
    not from environment variables. An env var on a Lambda is visible to
    anyone with lambda:GetFunctionConfiguration, a far wider audience
    than "can read this one secret."
    """
    global _driver
    if _driver is None:
        secrets = boto3.client("secretsmanager")
        secret = json.loads(secrets.get_secret_value(
            SecretId=os.environ["NEO4J_SECRET_ID"])["SecretString"])
        _driver = GraphDatabase.driver(
            secret["uri"], auth=(secret.get("user", "neo4j"), secret["password"]))
    return _driver


# ── The three tools ──────────────────────────────────────────────────────

_ENTITY_LABELS = {
    "trial": ["Trial"], "sponsor": ["Sponsor"], "drug": ["Drug"],
    "disease": ["Disease"], "site": ["Site"], "cro": ["CRO"],
    "any": ["Trial", "Sponsor", "Drug", "Disease", "Site", "CRO"],
}
_KEY_PROPERTY = {
    "Trial": "nctId", "Sponsor": "name", "Drug": "name",
    "Disease": "name", "Site": "facility", "CRO": "name",
}
_FULLTEXT_INDEX = "trial_entity_names"


def find_entity_by_name(args: dict) -> dict:
    name = args["name"]
    entity_type = args.get("entity_type", "any")
    labels = _ENTITY_LABELS.get(entity_type, _ENTITY_LABELS["any"])

    with _get_driver().session() as session:
        result = session.run(
            "CALL db.index.fulltext.queryNodes($index, $query) "
            "YIELD node, score "
            "WHERE any(l IN labels(node) WHERE l IN $labels) "
            "RETURN labels(node) AS labels, properties(node) AS props, score "
            "ORDER BY score DESC LIMIT 10",
            index=_FULLTEXT_INDEX, query=_escape_lucene(name), labels=labels,
        )
        candidates = []
        for row in result:
            label = row["labels"][0]
            key_prop = _KEY_PROPERTY.get(label, "name")
            candidates.append({
                "label": label,
                "key": row["props"].get(key_prop, ""),
                "name": row["props"].get("name") or row["props"].get("facility", ""),
                "score": row["score"],
            })
    return {"candidates": candidates}


def _escape_lucene(text: str) -> str:
    """Lucene special characters must be escaped or the query syntax
    error, not just a bad match. Ported directly from the same need in
    NLC's own entity_search.py — this is a well-known Lucene gotcha, not
    invented here.
    """
    special = r'+-&&||!(){}[]^"~*?:\/'
    return "".join(f"\\{c}" if c in special else c for c in text)


def validate_cypher(args: dict) -> dict:
    query = args["query"]
    with _get_driver().session() as session:
        try:
            session.run(f"EXPLAIN {query}")
            return {"valid": True}
        except Exception as exc:
            return {"valid": False, "error": str(exc)[:500]}


def execute_cypher(args: dict) -> dict:
    query = args["query"]
    with _get_driver().session() as session:
        try:
            result = session.run(query)
            records = [dict(r) for r in result]
        except Exception as exc:
            return {"error": True, "error_class": type(exc).__name__,
                    "detail": str(exc)[:500]}

    return _shape_result(records)


def _shape_result(records: list[dict]) -> dict:
    """Detect graph-shaped vs table-shaped results from the raw Neo4j
    driver output, matching NLC's own captured{} shape so the agent's
    _summarize() logic (already written, in the agent's own tools.py)
    needs no changes to consume this.
    """
    if not records:
        return {"result_shape": "empty"}

    nodes, rels, seen_nodes, seen_rels = [], [], set(), set()

    def _visit(value):
        # Neo4j's python driver returns Node/Relationship/Path objects for
        # graph-shaped RETURNs; everything else is a plain scalar/list/dict.
        if hasattr(value, "labels") and hasattr(value, "items"):
            if value.element_id not in seen_nodes:
                seen_nodes.add(value.element_id)
                nodes.append({"element_id": value.element_id,
                             "labels": list(value.labels),
                             "properties": dict(value.items())})
        elif hasattr(value, "type") and hasattr(value, "nodes"):
            if value.element_id not in seen_rels:
                seen_rels.add(value.element_id)
                start, end = value.nodes
                rels.append({"element_id": value.element_id, "type": value.type,
                            "start": start.element_id, "end": end.element_id,
                            "properties": dict(value.items())})
                _visit(start)
                _visit(end)
        elif hasattr(value, "relationships") and hasattr(value, "nodes"):
            for n in value.nodes:
                _visit(n)
            for r in value.relationships:
                _visit(r)

    for record in records:
        for value in record.values():
            _visit(value)

    if nodes or rels:
        return {"result_shape": "graph", "nodes": nodes, "relationships": rels}

    columns = list(records[0].keys())
    rows = [[record[c] for c in columns] for record in records]
    return {"result_shape": "table", "columns": columns, "rows": rows}


# ── Dispatch ──────────────────────────────────────────────────────────

_TOOLS = {
    "find_entity_by_name": find_entity_by_name,
    "validate_cypher": validate_cypher,
    "execute_cypher": execute_cypher,
}


def lambda_handler(event, context):
    tool_name = getattr(context, "client_context", None)
    tool_name = (tool_name.custom.get("bedrockAgentCoreToolName", "")
                if tool_name and tool_name.custom else "")
    # The Gateway may prefix the tool name with the target name
    # ("LambdaTarget___execute_cypher") — match on suffix, not equality.
    matched = next((name for name in _TOOLS if tool_name.endswith(name)), None)

    if matched is None:
        log.error("unknown tool requested: %r", tool_name)
        return {"error": True, "detail": f"unknown tool: {tool_name}"}

    log.info("dispatching to %s", matched)
    try:
        return _TOOLS[matched](event)
    except KeyError as exc:
        return {"error": True, "detail": f"missing required argument: {exc}"}
