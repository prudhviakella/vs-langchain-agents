"""AgentCore Gateway for the ecom NLQ agent — two MCP tools (execute_sql, explain_sql) on one Lambda target.

    deploy.py STEP 6
        │
        ├─ create_gateway(role)          MCP server, authorizerType=AWS_IAM (SigV4).
        │                                Exists already -> reused, and its role is
        │                                corrected if it is not `role`.
        │
        ├─ lambda_deploy.allow_gateway_invoke(gateway_arn)      ◄── BEFORE the target
        │
        └─ create_lambda_target()        registers the Lambda with TOOL_SCHEMA.
                                         Exists already -> UPDATED, so a redeploy that
                                         changes a tool is served at once.

WHY THE ORDER MATTERS
    create/update_gateway_target checks, there and then, that the Gateway may
    invoke the Lambda. Two things can allow it:

        the Lambda's resource policy   (lambda_deploy.allow_gateway_invoke)   effective at once
        the Gateway role's policy      (iam.gateway_role)                     effective after IAM propagates

    An earlier version added the resource policy AFTER the target. On a fresh
    account the role's policy had not propagated yet, the target call failed
    with "Gateway execution role lacks permission to invoke Lambda", and the
    resource policy was never added — so every re-run failed the same way.
    Now the resource policy goes first, and the target call still waits for
    IAM (infra/iam_ready.py) in case the check reads the role.

WHY AN EXISTING GATEWAY'S ROLE IS CHECKED
    A gateway created by an earlier deploy keeps the role it was created
    with. If that is not the role this deploy just granted, the grant goes
    to a role the gateway does not use, and the target check fails on every
    run. update_gateway puts the right role on it.

WHAT THIS DOES NOT DO
    It does not grant permissions — iam.py and lambda_deploy.py do. It does
    not add Cognito / JWT: the only caller is this project's own agent.
"""
import time

import boto3

from infra import iam_ready

# A Gateway tool's inputSchema accepts ONLY: type, properties, required, items,
# description (checked against the service model). No enum, default, minimum
# or format — those are rejected at create_gateway_target. Allowed values are
# stated in the description, and the Lambda enforces them itself.

client = boto3.client("bedrock-agentcore-control")

GATEWAY_NAME = "ecom-nlq-gateway"
DESCRIPTION = "MCP gateway for the ecom NLQ agent"
TARGET_NAME = "nlq-sql-tools"
PROTOCOL = {"mcp": {"supportedVersions": ["2025-03-26"]}}
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


def _wait_ready(gateway_id: str) -> None:
    deadline = time.time() + 180
    while (status := client.get_gateway(gatewayIdentifier=gateway_id)["status"]) != "READY":
        if status in ("FAILED", "UPDATE_UNSUCCESSFUL"):
            raise RuntimeError(f"gateway {GATEWAY_NAME!r} entered {status}")
        if time.time() > deadline:
            raise RuntimeError(f"gateway {GATEWAY_NAME!r} not READY after 180s ({status})")
        time.sleep(5)


def _ids(gw: dict) -> dict:
    return {"gatewayId": gw["gatewayId"], "gatewayArn": gw["gatewayArn"], "gatewayUrl": gw["gatewayUrl"]}


def create_gateway(role_arn: str) -> dict:
    """The gateway, created or reused, always using `role_arn`. Returns {gatewayId, gatewayArn, gatewayUrl}."""
    existing = client.list_gateways(maxResults=100).get("items", [])
    match = next((g for g in existing if g["name"] == GATEWAY_NAME), None)

    if match:
        gw = client.get_gateway(gatewayIdentifier=match["gatewayId"])
        if gw.get("roleArn") == role_arn:
            print(f"  gateway {GATEWAY_NAME!r} already exists")
            return _ids(gw)
        # STEP 1b — see WHY AN EXISTING GATEWAY'S ROLE IS CHECKED
        print(f"  gateway {GATEWAY_NAME!r} exists with role {gw.get('roleArn')} — switching it to {role_arn}")
        client.update_gateway(gatewayIdentifier=gw["gatewayId"], name=GATEWAY_NAME, description=DESCRIPTION,
                              roleArn=role_arn, protocolType="MCP", protocolConfiguration=PROTOCOL,
                              authorizerType="AWS_IAM")
        _wait_ready(gw["gatewayId"])
        return _ids(gw)

    print(f"  creating gateway {GATEWAY_NAME!r}")
    response = client.create_gateway(name=GATEWAY_NAME, description=DESCRIPTION, roleArn=role_arn,
                                     protocolType="MCP", protocolConfiguration=PROTOCOL,
                                     authorizerType="AWS_IAM")
    _wait_ready(response["gatewayId"])
    return _ids(response)


def create_lambda_target(gateway_id: str, lambda_arn: str) -> str:
    """Register (or update) the Lambda target. Call AFTER lambda_deploy.allow_gateway_invoke."""
    config = {"mcp": {"lambda": {"lambdaArn": lambda_arn,
                                 "toolSchema": {"inlinePayload": TOOL_SCHEMA}}}}
    creds = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
    what = f"the gateway to invoke {lambda_arn.rsplit(':', 1)[-1]}"

    existing = client.list_gateway_targets(gatewayIdentifier=gateway_id, maxResults=100).get("items", [])
    match = next((t for t in existing if t["name"] == TARGET_NAME), None)

    if match:
        print(f"  updating target {TARGET_NAME!r} ({len(TOOL_SCHEMA)} tools)")
        iam_ready.until_ready(lambda: client.update_gateway_target(
            gatewayIdentifier=gateway_id, targetId=match["targetId"], name=TARGET_NAME,
            targetConfiguration=config, credentialProviderConfigurations=creds),
            iam_ready.GATEWAY_TARGET, what)
        return match["targetId"]

    print(f"  creating target {TARGET_NAME!r} ({len(TOOL_SCHEMA)} tools)")
    return iam_ready.until_ready(lambda: client.create_gateway_target(
        gatewayIdentifier=gateway_id, name=TARGET_NAME,
        targetConfiguration=config, credentialProviderConfigurations=creds),
        iam_ready.GATEWAY_TARGET, what)["targetId"]
