"""Create the AgentCore Gateway and register the tools Lambda as its MCP target.

    deploy.py
        |
        v
    infra/gateway.py
        |
        |-- create_gateway()        the MCP server itself — IAM-authorized,
        |                           no Cognito/OAuth setup needed since
        |                           the only caller is our own Supervisor
        |
        `-- create_lambda_target()  registers the tools Lambda, with an
                                    inline JSON schema for its three
                                    tools — the Gateway uses this schema
                                    to expose them over MCP; it never
                                    inspects the Lambda's own code

WHY authorizerType='AWS_IAM', NOT 'CUSTOM_JWT'

Every public AWS example builds a Cognito user pool for this, because
their Gateway is meant to be called by an external, untrusted client —
someone else's app, a third party. This Gateway has exactly one caller:
our own agent, running in our own AgentCore Runtime, under our own IAM
role. AWS_IAM authorization means "a validly SigV4-signed request from a
principal with bedrock-agentcore:InvokeGateway" — which our agent's own
execution role already has once granted, with nothing extra to stand up.
Cognito would be solving a problem we do not have.

WHAT THIS DOES NOT DO

    - It does not attempt semantic tool search (protocolConfiguration's
      searchType). Three tools fit in a model's context whole; semantic
      search over a tool list earns its complexity on gateways serving
      dozens of tools, not three.

TWO THINGS list_* RESULTS DO NOT CONTAIN

Verified against the real boto3 service model, not assumed:

    ListGateways items carry ONLY gatewayId, name, status, protocolType,
    authorizerType and timestamps. There is no gatewayArn and no
    gatewayUrl on a list item. Reading them off a match raised KeyError
    on every re-run — the first deploy worked, the second failed.
    get_gateway() is called instead, which does return both.

    ListGatewayTargets items likewise carry no targetConfiguration, so
    an existing target cannot be compared field-by-field. The target is
    updated unconditionally instead: skipping the update would leave
    the Gateway serving a stale tool schema after a tool is added or a
    description changed, with no error to notice.
"""
import time

import boto3

client = boto3.client("bedrock-agentcore-control")

GATEWAY_NAME = "trial-graph-gateway"
TARGET_NAME = "trial-graph-tools"

_TOOL_SCHEMA = [
    {
        "name": "find_entity_by_name",
        "description": (
            "Resolve a trial/sponsor/drug/disease/site NAME to candidate "
            "entities. Use this FIRST, before writing any query that "
            "filters on a name."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "entity_type": {"type": "string",
                               "enum": ["trial", "sponsor", "drug", "disease",
                                       "site", "cro", "any"]},
            },
            "required": ["name"],
        },
    },
    {
        "name": "validate_cypher",
        "description": "Check whether a Cypher query is valid WITHOUT running it.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "execute_cypher",
        "description": "Run a read-only Cypher query against the trial graph.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def create_gateway(role_arn: str) -> dict:
    """Create the gateway if it doesn't exist yet. Returns
    {"gatewayId", "gatewayArn", "gatewayUrl"} either way.
    """
    existing = client.list_gateways(maxResults=100).get("items", [])
    match = next((g for g in existing if g["name"] == GATEWAY_NAME), None)
    if match:
        print(f"  gateway {GATEWAY_NAME!r} already exists")
        gw = client.get_gateway(gatewayIdentifier=match["gatewayId"])
        return {"gatewayId": gw["gatewayId"], "gatewayArn": gw["gatewayArn"],
                "gatewayUrl": gw["gatewayUrl"]}

    print(f"  creating gateway {GATEWAY_NAME!r}")
    response = client.create_gateway(
        name=GATEWAY_NAME,
        description="MCP gateway for trial_graph's Neo4j tools",
        roleArn=role_arn,
        protocolType="MCP",
        protocolConfiguration={"mcp": {"supportedVersions": ["2025-03-26"]}},
        authorizerType="AWS_IAM",
    )
    # Gateway creation is asynchronous (status starts at CREATING) — poll
    # rather than assume it's ready the instant this call returns, or the
    # target-creation call right after this can fail with a state error.
    gateway_id = response["gatewayId"]
    deadline = time.time() + 180
    while True:
        status = client.get_gateway(gatewayIdentifier=gateway_id)["status"]
        if status == "READY":
            break
        if status in ("FAILED", "UPDATE_UNSUCCESSFUL"):
            raise RuntimeError(f"gateway {GATEWAY_NAME!r} entered {status}")
        if time.time() > deadline:
            raise RuntimeError(f"gateway {GATEWAY_NAME!r} not READY after 180s "
                              f"(status={status})")
        time.sleep(5)

    return {"gatewayId": gateway_id, "gatewayArn": response["gatewayArn"],
            "gatewayUrl": response["gatewayUrl"]}


def create_lambda_target(gateway_id: str, lambda_arn: str) -> str:
    """Register the tools Lambda as this gateway's target, with the
    inline tool schema above. Returns the target id.
    """
    config = {"mcp": {"lambda": {"lambdaArn": lambda_arn,
                                 "toolSchema": {"inlinePayload": _TOOL_SCHEMA}}}}
    creds = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]

    existing = client.list_gateway_targets(
        gatewayIdentifier=gateway_id, maxResults=100).get("items", [])
    match = next((t for t in existing if t["name"] == TARGET_NAME), None)

    if match:
        print(f"  updating target {TARGET_NAME!r} ({len(_TOOL_SCHEMA)} tools)")
        client.update_gateway_target(
            gatewayIdentifier=gateway_id, targetId=match["targetId"], name=TARGET_NAME,
            targetConfiguration=config, credentialProviderConfigurations=creds)
        return match["targetId"]

    print(f"  creating target {TARGET_NAME!r} ({len(_TOOL_SCHEMA)} tools)")
    return client.create_gateway_target(
        gatewayIdentifier=gateway_id, name=TARGET_NAME,
        targetConfiguration=config, credentialProviderConfigurations=creds)["targetId"]
