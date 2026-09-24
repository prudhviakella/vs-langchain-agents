"""IAM role for trial_search's AgentCore Runtime — same structure and
reasoning as trial_graph/infra/runtime_iam.py. No Secrets Manager access
here either, for the same reason: the agent never touches OpenAI or
Pinecone credentials directly, only the Lambda behind the Gateway does.
"""
import json
import time

import boto3

iam = boto3.client("iam")

RUNTIME_ROLE_NAME = "trial-search-runtime-role"

_RUNTIME_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}


def runtime_role(gateway_arn: str, ecr_repo_arn: str, model_id: str,
                 guardrail_arn: str) -> str:
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
