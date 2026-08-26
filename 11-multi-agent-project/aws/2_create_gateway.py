"""Step 2 — turn the Lambda into MCP tools with an AgentCore Gateway.

    python 2_create_gateway.py

This is the step worth understanding. The Gateway is a managed MCP server: you
point it at a Lambda and it exposes each handler as an MCP tool, with a schema,
discoverable at an endpoint. No MCP server to write, host or patch.

    Lambda  ──▶  Gateway  ──▶  MCP endpoint  ──▶  any agent
                                                  (yours, Claude Desktop,
                                                   Cursor, anything)

TWO DIRECTIONS OF AUTH, AND THEY ARE DIFFERENT

    inbound    who may call the Gateway         JWT (Cognito or your IdP)
    outbound   how the Gateway calls Lambda     IAM, the role created below

Inbound is JWT rather than SigV4. That is a real constraint: agent-to-agent
calls on AgentCore Runtime are SigV4, but the Gateway wants a bearer token. If
you are avoiding Cognito, the options are to bring your own OIDC issuer, or to
call the Lambda directly from the agent and skip the Gateway — which costs you
MCP discovery and gains you one less moving part.

Read this script before running it. It creates a Cognito user pool.
"""

import json
from pathlib import Path

import boto3

import config

control = boto3.client("bedrock-agentcore-control", region_name=config.REGION)
iam = boto3.client("iam", region_name=config.REGION)


def gateway_role() -> str:
    """The role the Gateway assumes to invoke the Lambda.

    Scoped to that one function. A wildcard here would let the Gateway invoke
    anything in the account, which is exactly the sort of grant that is invisible
    until something else goes wrong.
    """
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Action": "sts:AssumeRole"}]}
    try:
        return iam.get_role(RoleName=config.GATEWAY_ROLE)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    state = json.loads(Path(config.STATE_FILE).read_text())
    role = iam.create_role(RoleName=config.GATEWAY_ROLE,
                           AssumeRolePolicyDocument=json.dumps(trust))["Role"]
    iam.put_role_policy(
        RoleName=config.GATEWAY_ROLE, PolicyName="invoke-tools",
        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Action": "lambda:InvokeFunction",
            "Resource": state["lambda_arn"]}]}))
    print(f"  created role: {config.GATEWAY_ROLE}")
    return role["Arn"]


def tool_schema() -> list[dict]:
    """The tool definitions the Gateway advertises over MCP.

    This IS the interface an agent sees. The description is what a model reads
    when deciding whether to call the tool, so it is written for a model — say
    what the tool does AND what it is not for, or it wins every question.
    """
    return [{"name": name,
             "description": spec["description"],
             "inputSchema": spec["input_schema"]}
            for name, spec in config.TOOLS.items()]


def main() -> None:
    state = json.loads(Path(config.STATE_FILE).read_text())
    if "lambda_arn" not in state:
        raise SystemExit("run 1_deploy_tools.py first")

    print(f"creating gateway in {config.REGION}\n")
    role_arn = gateway_role()

    # Inbound authorisation. The starter toolkit will create a Cognito pool for
    # you; doing it explicitly makes visible that a Gateway is a public endpoint
    # and something has to decide who may call it.
    from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

    client = GatewayClient(region_name=config.REGION)
    cognito = client.create_oauth_authorizer_with_cognito(config.GATEWAY_NAME)
    print("  created a Cognito authorizer for inbound calls")

    gateway = client.create_mcp_gateway(
        name=config.GATEWAY_NAME,
        role_arn=role_arn,
        authorizer_config=cognito["authorizer_config"],
        # Semantic search over tool descriptions. Worth it once you have more
        # than a handful: the agent searches for a capability rather than
        # receiving every tool definition in its context on every turn.
        enable_semantic_search=True)
    print(f"  gateway: {gateway['gatewayId']}")

    client.create_mcp_gateway_target(
        gateway=gateway,
        name=f"{config.PROJECT}-tools",
        target_type="lambda",
        target_payload={"lambdaArn": state["lambda_arn"],
                        "toolSchema": {"inlinePayload": tool_schema()}})
    print(f"  target: {len(config.TOOLS)} tools from one Lambda")

    state.update({
        "gateway_id": gateway["gatewayId"],
        "gateway_url": gateway.get("gatewayUrl"),
        "cognito": {k: v for k, v in cognito.items() if k != "authorizer_config"},
    })
    Path(config.STATE_FILE).write_text(json.dumps(state, indent=2))

    print(f"\n  MCP endpoint: {state['gateway_url']}")
    print("  next: python 3_deploy_agents.py")


if __name__ == "__main__":
    main()
