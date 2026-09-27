"""IAM execution role for trial_graph's AgentCore Runtime.

    runtime_role()
        │
        ├─ trust          bedrock-agentcore.amazonaws.com, this account only
        ├─ read-config    Parameter Store under /trial-agents/trial_graph
        ├─ read-secret    the shared OpenAI secret
        ├─ read-prompts   Prompt Management: the system prompt
        ├─ apply-guardrail
        ├─ invoke-gateway its own Gateway
        ├─ pull-ecr-image
        └─ observability  CloudWatch Logs, X-Ray, CloudWatch metrics

WHY THE PARAMETER PATH IS GRANTED TWICE

GetParametersByPath(Path="/trial-agents/x") is authorized against the BARE
path's ARN — parameter/trial-agents/x — not the wildcard parameter/.../x/*.
Granting only the wildcard is denied. Both are granted: the bare ARN for the
path read, the wildcard for any single GetParameter.

NO bedrock:InvokeModel

The chat model is OpenAI, called over HTTPS with the key from Secrets
Manager. This role no longer calls any Bedrock model; the grant is removed
rather than left in place "in case".

THE ROLE CONVERGES

Every deploy puts every policy below and DELETES any inline policy not in
this set. An earlier version only added — so a permission removed from this
file stayed on the role forever.

WHAT THIS DOES NOT DO

    No database or API credential. The container never holds Neo4j or
    Pinecone credentials — tools run in the Gateway's Lambda, under its own role.
"""
import json
import time

import boto3

iam = boto3.client("iam")

RUNTIME_ROLE_NAME = "trial-graph-runtime-role"


def _account_region() -> tuple[str, str]:
    session = boto3.Session()
    return (session.client("sts").get_caller_identity()["Account"],
            session.region_name or "us-east-1")


def _ensure_role(account: str) -> str:
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Action": "sts:AssumeRole",
        "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}
    try:
        arn = iam.get_role(RoleName=RUNTIME_ROLE_NAME)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=RUNTIME_ROLE_NAME,
                                      PolicyDocument=json.dumps(trust))
        return arn
    except iam.exceptions.NoSuchEntityException:
        pass
    print(f"  creating IAM role {RUNTIME_ROLE_NAME!r}")
    arn = iam.create_role(RoleName=RUNTIME_ROLE_NAME,
                          AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
    time.sleep(10)   # IAM propagation: a brand-new role is not usable at once
    return arn


def _converge(policies: dict[str, list[dict]]) -> None:
    for name, statements in policies.items():
        iam.put_role_policy(RoleName=RUNTIME_ROLE_NAME, PolicyName=name,
                            PolicyDocument=json.dumps({"Version": "2012-10-17",
                                                        "Statement": statements}))
    existing = iam.list_role_policies(RoleName=RUNTIME_ROLE_NAME)["PolicyNames"]
    for stale in set(existing) - set(policies):
        print(f"  removing stale policy {stale!r}")
        iam.delete_role_policy(RoleName=RUNTIME_ROLE_NAME, PolicyName=stale)


def _parameter_arns(region: str, account: str, paths: list[str]) -> list[str]:
    base = f"arn:aws:ssm:{region}:{account}:parameter"
    return [arn for p in paths for arn in (f"{base}{p}", f"{base}{p}/*")]


def _common(account: str, region: str, *, paths: list[str], secret_arn: str,
            prompt_arns: list[str], guardrail_arn: str, ecr_repo_arn: str) -> dict:
    logs = f"arn:aws:logs:{region}:{account}:log-group:/aws/bedrock-agentcore/runtimes/*"
    return {
        "read-config": [{"Effect": "Allow",
                         "Action": ["ssm:GetParametersByPath", "ssm:GetParameter",
                                    "ssm:GetParameters"],
                         "Resource": _parameter_arns(region, account, paths)}],
        "read-secret": [{"Effect": "Allow", "Action": "secretsmanager:GetSecretValue",
                         "Resource": secret_arn}],
        # A pinned read is GetPrompt(promptIdentifier, promptVersion); the
        # versioned resource is "<prompt arn>:<version>".
        "read-prompts": [{"Effect": "Allow", "Action": "bedrock:GetPrompt",
                          "Resource": [a for arn in prompt_arns for a in (arn, f"{arn}:*")]}],
        "apply-guardrail": [{"Effect": "Allow", "Action": "bedrock:ApplyGuardrail",
                             "Resource": [guardrail_arn, f"{guardrail_arn}:*"]}],
        "pull-ecr-image": [
            {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
            {"Effect": "Allow", "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
             "Resource": ecr_repo_arn}],
        "observability": [
            {"Effect": "Allow", "Resource": logs,
             "Action": ["logs:CreateLogGroup", "logs:DescribeLogStreams"]},
            {"Effect": "Allow", "Resource": f"{logs}:log-stream:*",
             "Action": ["logs:CreateLogStream", "logs:PutLogEvents"]},
            {"Effect": "Allow", "Resource": f"arn:aws:logs:{region}:{account}:log-group:*",
             "Action": "logs:DescribeLogGroups"},
            {"Effect": "Allow", "Resource": "*",
             "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords",
                        "xray:GetSamplingRules", "xray:GetSamplingTargets"]},
            {"Effect": "Allow", "Resource": "*", "Action": "cloudwatch:PutMetricData",
             "Condition": {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}}}],
    }


def runtime_role(*, gateway_arn: str, ecr_repo_arn: str, guardrail_arn: str,
                 prompt_arns: list[str], secret_arn: str, param_prefix: str) -> str:
    account, region = _account_region()
    arn = _ensure_role(account)
    policies = _common(account, region, paths=[param_prefix], secret_arn=secret_arn,
                       prompt_arns=prompt_arns, guardrail_arn=guardrail_arn,
                       ecr_repo_arn=ecr_repo_arn)
    policies["invoke-gateway"] = [{"Effect": "Allow",
                                   "Action": "bedrock-agentcore:InvokeGateway",
                                   "Resource": gateway_arn}]
    _converge(policies)
    return arn
