"""AgentCore Gateway for trial_search — three MCP tools on one Lambda target.

    create_gateway()          MCP server, authorizerType=AWS_IAM (SigV4).
                              The only caller is this project's own agent,
                              so no Cognito / JWT is needed.
    create_lambda_target()    registers the Lambda with the tool schema
                              below. If the target already exists, it is
                              UPDATED — otherwise a redeploy that adds a
                              tool would leave the Gateway serving the old
                              tool list with no error.

WHAT THIS DOES NOT DO

    It does not validate budgets. max_tokens and exclude_ids appear in the
    schema only so they survive argument validation on their way to the
    Lambda; RetrievalMiddleware always overwrites whatever the model sends.
"""
import time

import boto3

client = boto3.client("bedrock-agentcore-control")

GATEWAY_NAME = "trial-search-gateway"
TARGET_NAME = "trial-search-tools"

_MIDDLEWARE_OWNED = {
    "max_tokens": {"type": "integer",
                   "description": "Set by the agent. Leave unset; any value is replaced."},
    "exclude_ids": {"type": "array", "items": {"type": "string"},
                    "description": "Set by the agent. Leave unset; any value is replaced."},
}

TOOL_SCHEMA = [
    {
        "name": "semantic_search",
        "description": "Find passages in the trial protocol corpus by meaning. "
                       "Always the first step.",
        "inputSchema": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "description": "How many passages (max 20)."},
            "content_type": {"type": "string",
                             "enum": ["text", "table", "figure", "table_summary"]},
            "doc_id": {"type": "string", "description": "Restrict to one document."},
        }},
    },
    {
        "name": "expand_neighbors",
        "description": "Return the chunks immediately before and after a passage, "
                       "nearest first. Use when a passage is cut off at its boundary.",
        "inputSchema": {"type": "object", "required": ["chunk_id"], "properties": {
            "chunk_id": {"type": "string"},
            "window": {"type": "integer", "description": "Hops each side. Start with 2."},
            **_MIDDLEWARE_OWNED,
        }},
    },
    {
        "name": "expand_table",
        "description": "From a table_summary passage, return the table's actual rows "
                       "with exact values.",
        "inputSchema": {"type": "object", "required": ["chunk_id"], "properties": {
            "chunk_id": {"type": "string",
                         "description": "chunk_id of a content_type=table_summary passage."},
            **_MIDDLEWARE_OWNED,
        }},
    },
]


def create_gateway(role_arn: str) -> dict:
    existing = client.list_gateways(maxResults=100).get("items", [])
    match = next((g for g in existing if g["name"] == GATEWAY_NAME), None)
    if match:
        print(f"  gateway {GATEWAY_NAME!r} already exists")
        gw = client.get_gateway(gatewayIdentifier=match["gatewayId"])
        return {"gatewayId": gw["gatewayId"], "gatewayArn": gw["gatewayArn"],
                "gatewayUrl": gw["gatewayUrl"]}

    print(f"  creating gateway {GATEWAY_NAME!r}")
    response = client.create_gateway(
        name=GATEWAY_NAME, description="MCP gateway for trial_search",
        roleArn=role_arn, protocolType="MCP",
        protocolConfiguration={"mcp": {"supportedVersions": ["2025-03-26"]}},
        authorizerType="AWS_IAM")
    gateway_id = response["gatewayId"]

    deadline = time.time() + 180
    while (status := client.get_gateway(gatewayIdentifier=gateway_id)["status"]) != "READY":
        if status in ("FAILED", "UPDATE_UNSUCCESSFUL"):
            raise RuntimeError(f"gateway {GATEWAY_NAME!r} entered {status}")
        if time.time() > deadline:
            raise RuntimeError(f"gateway {GATEWAY_NAME!r} not READY after 180s ({status})")
        time.sleep(5)
    return {"gatewayId": gateway_id, "gatewayArn": response["gatewayArn"],
            "gatewayUrl": response["gatewayUrl"]}


def create_lambda_target(gateway_id: str, lambda_arn: str) -> str:
    config = {"mcp": {"lambda": {"lambdaArn": lambda_arn,
                                 "toolSchema": {"inlinePayload": TOOL_SCHEMA}}}}
    creds = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]

    existing = client.list_gateway_targets(
        gatewayIdentifier=gateway_id, maxResults=100).get("items", [])
    match = next((t for t in existing if t["name"] == TARGET_NAME), None)

    if match:
        print(f"  updating target {TARGET_NAME!r} ({len(TOOL_SCHEMA)} tools)")
        client.update_gateway_target(
            gatewayIdentifier=gateway_id, targetId=match["targetId"], name=TARGET_NAME,
            targetConfiguration=config, credentialProviderConfigurations=creds)
        return match["targetId"]

    print(f"  creating target {TARGET_NAME!r} ({len(TOOL_SCHEMA)} tools)")
    return client.create_gateway_target(
        gatewayIdentifier=gateway_id, name=TARGET_NAME,
        targetConfiguration=config, credentialProviderConfigurations=creds)["targetId"]
