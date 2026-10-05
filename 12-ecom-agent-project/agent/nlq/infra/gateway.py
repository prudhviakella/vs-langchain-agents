"""AgentCore Gateway for the ecom NLQ agent — two MCP tools (execute_sql, explain_sql) on one Lambda target.

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

# A Gateway tool's inputSchema accepts ONLY: type, properties, required, items,
# description (checked against the service model). No enum, default, minimum
# or format — those are rejected at create_gateway_target. Allowed values are
# stated in the description, and the Lambda enforces them itself.

client = boto3.client("bedrock-agentcore-control")

GATEWAY_NAME = "ecom-nlq-gateway"
TARGET_NAME = "nlq-sql-tools"
TOOL_SCHEMA = [
    {
        "name": "execute_sql",
        "description": "Run one read-only SELECT on the ecom schema and return the rows.",
        "inputSchema": {"type": "object", "required": ["sql"], "properties": {
            "sql": {"type": "string"},
            "row_cap": {"type": "integer", "description": "Maximum rows returned."},
            "timeout_s": {"type": "integer", "description": "Statement timeout in seconds."},
        }},
    },
    {
        "name": "explain_sql",
        "description": "Plan a query without running it.",
        "inputSchema": {"type": "object", "required": ["sql"], "properties": {"sql": {"type": "string"}}},
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
        name=GATEWAY_NAME, description="MCP gateway for the ecom NLQ agent",
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
