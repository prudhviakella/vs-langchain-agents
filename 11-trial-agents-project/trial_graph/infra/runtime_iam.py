"""IAM role for trial_graph's AgentCore Runtime — the agent's own
compute, separate from the Gateway/Lambda tool-execution roles in
infra/iam.py.

WHY THIS ROLE NEEDS NO SECRETS MANAGER ACCESS

Worth stating explicitly, because it confirms the architecture's own
separation of concerns actually holds: the Runtime role never touches
Neo4j credentials at all. The agent never talks to Neo4j directly — it
only speaks MCP to the Gateway, and the Gateway's Lambda (a completely
separate execution role, in infra/iam.py) is what holds the Neo4j
secret. If this role ever needed Secrets Manager access, that would be
a sign the "agent process never holds tool credentials" boundary had
been broken somewhere.
"""
import json
import time

import boto3

iam = boto3.client("iam")

RUNTIME_ROLE_NAME = "trial-graph-runtime-role"

_RUNTIME_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}


def runtime_role(gateway_arn: str, ecr_repo_arn: str, model_id: str,
                 guardrail_arn: str) -> str:
    """Create the Runtime execution role if it doesn't exist yet.

    Four permissions, each scoped as narrowly as the API allows:
      - bedrock:InvokeModel, on the specific model this agent calls
      - bedrock:ApplyGuardrail, on this agent's own guardrail
      - bedrock-agentcore:InvokeGateway, on this agent's own gateway
        (confirmed as the exact required action and ARN shape directly
        against AWS's own IAM permissions reference, not assumed)
      - ecr:GetDownloadUrlForLayer / BatchGetImage / GetAuthorizationToken,
        so the Runtime can actually pull this agent's own container image
    """
    try:
        role = iam.get_role(RoleName=RUNTIME_ROLE_NAME)
        return role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"  creating IAM role {RUNTIME_ROLE_NAME!r}")
    role = iam.create_role(RoleName=RUNTIME_ROLE_NAME,
                           AssumeRolePolicyDocument=json.dumps(_RUNTIME_TRUST))
    time.sleep(10)  # IAM propagation delay

    iam.put_role_policy(RoleName=RUNTIME_ROLE_NAME, PolicyName="invoke-model", PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "bedrock:InvokeModel",
                       "Resource": f"arn:aws:bedrock:*::foundation-model/{model_id}"}],
    }))
    iam.put_role_policy(RoleName=RUNTIME_ROLE_NAME, PolicyName="apply-guardrail", PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "bedrock:ApplyGuardrail",
                       "Resource": guardrail_arn}],
    }))
    iam.put_role_policy(RoleName=RUNTIME_ROLE_NAME, PolicyName="invoke-gateway", PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "bedrock-agentcore:InvokeGateway",
                       "Resource": gateway_arn}],
    }))
    iam.put_role_policy(RoleName=RUNTIME_ROLE_NAME, PolicyName="pull-ecr-image", PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
            {"Effect": "Allow",
             "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
             "Resource": ecr_repo_arn},
        ],
    }))
    return role["Role"]["Arn"]
