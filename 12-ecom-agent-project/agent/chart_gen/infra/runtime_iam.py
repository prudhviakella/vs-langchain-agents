"""IAM execution role for the chart_gen AgentCore Runtime.

    ecom-chart-gen-runtime-role (trusted by bedrock-agentcore.amazonaws.com)
        ├─ read-secrets    the OpenAI key secret only
        ├─ pull-ecr-image  the chart_gen repository only
        └─ observability   its own log groups, X-Ray, metrics in the bedrock-agentcore namespace

WHAT THIS DOES NOT DO
    No data access of any kind: chart_gen only sees the table the supervisor sends it.
"""
import json
import time

import boto3

RUNTIME_ROLE_NAME = "ecom-chart-gen-runtime-role"
iam = boto3.client("iam")


def runtime_role(*, ecr_repo_arn: str, secret_arns: list[str]) -> str:
    session = boto3.session.Session()
    account, region = session.client("sts").get_caller_identity()["Account"], session.region_name
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}
    try:
        arn = iam.get_role(RoleName=RUNTIME_ROLE_NAME)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=RUNTIME_ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        print(f"  creating IAM role {RUNTIME_ROLE_NAME!r}")
        time.sleep(10)   # IAM propagation: a brand-new role is not usable at once
    logs = f"arn:aws:logs:{region}:{account}:log-group:/aws/bedrock-agentcore/runtimes/*"
    policies = {
        "read-secrets": [{"Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": secret_arns}],
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
