"""IAM roles for trial_graph's AgentCore Gateway deployment.

    deploy.py
        |
        v
    infra/iam.py
        |
        |-- lambda_role()    execution role for the tools Lambda —
        |                    basic execution + Secrets Manager read
        |                    for the Neo4j credential
        |
        `-- gateway_role()   execution role for the Gateway itself —
                             lets it invoke the tools Lambda

WHAT THIS DOES NOT DO

    - It does not create a role for the agent's own AgentCore Runtime.
      That role needs bedrock:InvokeModel (for the Anthropic-on-Bedrock
      call) and bedrock:ApplyGuardrail, plus permission to call the
      Gateway's own endpoint — kept in infra/agent_role.py, not here,
      since it is a different trust boundary (the agent's own compute,
      not the tool-execution path).
    - It does not scope the Secrets Manager permission to the exact
      secret ARN at role-creation time — the secret is created in the
      same deploy run, so the policy is attached with a wildcard on the
      known name prefix, tightened once the real ARN exists (see
      lambda_role's own comment).
"""
import json
import time

import boto3

iam = boto3.client("iam")

LAMBDA_ROLE_NAME = "trial-graph-tools-lambda-role"
GATEWAY_ROLE_NAME = "trial-graph-gateway-role"

_LAMBDA_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "lambda.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}

_GATEWAY_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}


def _ensure_role(name: str, trust_policy: dict) -> str:
    """Create the role if it doesn't exist yet; return its ARN either way.
    Idempotent, matching every other infra/*.py in this project.
    """
    try:
        role = iam.get_role(RoleName=name)
        return role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"  creating IAM role {name!r}")
    role = iam.create_role(RoleName=name,
                           AssumeRolePolicyDocument=json.dumps(trust_policy))
    # New roles are not immediately usable — IAM's own propagation delay,
    # not a bug in this script. Every caller of _ensure_role hits a real
    # AWS API within seconds, so this wait belongs here once, not
    # repeated at every call site.
    time.sleep(10)
    return role["Role"]["Arn"]


def _attach_inline(role_name: str, policy_name: str, policy_doc: dict) -> None:
    iam.put_role_policy(RoleName=role_name, PolicyName=policy_name,
                        PolicyDocument=json.dumps(policy_doc))


def lambda_role(secret_arn_prefix: str) -> str:
    """Execution role for the tools Lambda: basic execution (CloudWatch
    Logs) plus read access to the Neo4j credential secret.

    Called before the secret itself necessarily exists, so the policy is
    attached scoped to the secret's known NAME prefix rather than its
    exact ARN — Secrets Manager appends a random suffix to every secret's
    real ARN, which does not exist until the secret is actually created.
    Call tighten_secret_policy() once it does.
    """
    arn = _ensure_role(LAMBDA_ROLE_NAME, _LAMBDA_TRUST)
    _attach_inline(LAMBDA_ROLE_NAME, "basic-execution", {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow",
                       "Action": ["logs:CreateLogGroup", "logs:CreateLogStream",
                                 "logs:PutLogEvents"],
                       "Resource": "arn:aws:logs:*:*:*"}],
    })
    _attach_inline(LAMBDA_ROLE_NAME, "read-neo4j-secret", {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "secretsmanager:GetSecretValue",
                       "Resource": f"{secret_arn_prefix}*"}],
    })
    return arn


def tighten_secret_policy(secret_arn: str) -> None:
    """Re-attach read-neo4j-secret scoped to the secret's real, exact ARN
    now that it exists — replacing the name-prefix wildcard lambda_role()
    had to use before the secret was created. Same policy name, so this
    overwrites rather than adds a second statement.
    """
    _attach_inline(LAMBDA_ROLE_NAME, "read-neo4j-secret", {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "secretsmanager:GetSecretValue",
                       "Resource": secret_arn}],
    })


def gateway_role(lambda_arn: str) -> str:
    """Execution role for the Gateway: lets it invoke the tools Lambda on
    the caller's behalf. Scoped to this one function, not lambda:* —
    the Gateway has no reason to invoke anything else.
    """
    arn = _ensure_role(GATEWAY_ROLE_NAME, _GATEWAY_TRUST)
    _attach_inline(GATEWAY_ROLE_NAME, "invoke-tools-lambda", {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "lambda:InvokeFunction",
                       "Resource": lambda_arn}],
    })
    return arn
