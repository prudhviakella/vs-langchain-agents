"""IAM execution role for the NLQ AgentCore Runtime.

    ecom-nlq-runtime-role (trusted by bedrock-agentcore.amazonaws.com)
        ├─ read-secrets    the OpenAI and Pinecone key secrets — nothing else
        ├─ invoke-gateway  bedrock-agentcore:InvokeGateway on THE NLQ gateway only
        ├─ pull-ecr-image  the NLQ repository only
        └─ observability   its own log groups, X-Ray, metrics in the bedrock-agentcore namespace

WHAT THIS DOES NOT DO
    No database access: the runtime never touches RDS. The Lambda does, with
    its own role, from inside the VPC.
"""
import json
import time

import boto3

RUNTIME_ROLE_NAME = "ecom-nlq-runtime-role"
iam = boto3.client("iam")


def _account_region() -> tuple[str, str]:
    session = boto3.session.Session()
    return session.client("sts").get_caller_identity()["Account"], session.region_name


def _ensure_role(account: str) -> str:
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}
    try:
        return iam.get_role(RoleName=RUNTIME_ROLE_NAME)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=RUNTIME_ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        print(f"  creating IAM role {RUNTIME_ROLE_NAME!r}")
        time.sleep(10)   # IAM propagation: a brand-new role is not usable at once
        return arn


def runtime_role(*, gateway_arn: str, ecr_repo_arn: str, secret_arns: list[str]) -> str:
    account, region = _account_region()
    arn = _ensure_role(account)
    logs = f"arn:aws:logs:{region}:{account}:log-group:/aws/bedrock-agentcore/runtimes/*"
    policies = {
        "read-secrets": [{"Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": secret_arns}],
        "invoke-gateway": [{"Effect": "Allow", "Action": "bedrock-agentcore:InvokeGateway", "Resource": gateway_arn}],
        "pull-ecr-image": [
            {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
            {"Effect": "Allow", "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"], "Resource": ecr_repo_arn}],
        "observability": [
            {"Effect": "Allow", "Resource": logs, "Action": ["logs:CreateLogGroup", "logs:DescribeLogStreams"]},
            {"Effect": "Allow", "Resource": f"{logs}:log-stream:*", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"]},
            {"Effect": "Allow", "Resource": f"arn:aws:logs:{region}:{account}:log-group:*", "Action": "logs:DescribeLogGroups"},
            {"Effect": "Allow", "Resource": "*", "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords",
                                                           "xray:GetSamplingRules", "xray:GetSamplingTargets"]},
            {"Effect": "Allow", "Resource": "*", "Action": "cloudwatch:PutMetricData",
             "Condition": {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}}}],
    }
    for name, statements in policies.items():
        iam.put_role_policy(RoleName=RUNTIME_ROLE_NAME, PolicyName=name,
                            PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": statements}))
    return arn
