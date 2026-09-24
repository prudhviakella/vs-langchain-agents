"""IAM role for the Supervisor's AgentCore Runtime.

Genuinely different permission set from trial_graph/trial_search's own
runtime_iam.py: this role needs bedrock-agentcore:InvokeAgentRuntime,
scoped to BOTH specialists' runtime ARNs — not InvokeGateway, since the
Supervisor has no Gateway of its own. Its one tool calls
trial_graph/trial_search directly through AgentCore Runtime's data
plane (see agent_client.py), a different integration path than how
trial_graph/trial_search reach their own tools via a Gateway.

Confirmed directly against AWS's own IAM permissions reference: a
caller invoking another AgentCore Runtime needs exactly this action,
scoped to that runtime's ARN — the same shape already verified for
trial_graph/trial_search's own InvokeGateway permission, just a
different action name for a different callee type.
"""
import json
import time

import boto3

iam = boto3.client("iam")

RUNTIME_ROLE_NAME = "supervisor-runtime-role"

_RUNTIME_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}


def runtime_role(trial_graph_arn: str, trial_search_arn: str, ecr_repo_arn: str,
                 model_id: str, guardrail_arn: str) -> str:
    try:
        role = iam.get_role(RoleName=RUNTIME_ROLE_NAME)
        return role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"  creating IAM role {RUNTIME_ROLE_NAME!r}")
    role = iam.create_role(RoleName=RUNTIME_ROLE_NAME,
                           AssumeRolePolicyDocument=json.dumps(_RUNTIME_TRUST))
    time.sleep(10)

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
    iam.put_role_policy(RoleName=RUNTIME_ROLE_NAME, PolicyName="invoke-specialists", PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "bedrock-agentcore:InvokeAgentRuntime",
                       "Resource": [trial_graph_arn, trial_search_arn]}],
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
