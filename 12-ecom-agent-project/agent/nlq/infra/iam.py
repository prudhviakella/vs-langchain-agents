"""IAM roles for the NLQ Gateway and its SQL Lambda.

    ecom-nlq-sql-lambda-role   (trusted by lambda.amazonaws.com)
        ├─ basic-execution         logs
        ├─ read-secrets            ONE secret: the read-only DB user's password
        └─ vpc-network-interfaces  the Lambda runs in the VPC beside RDS

    ecom-nlq-gateway-role      (trusted by bedrock-agentcore.amazonaws.com)
        └─ invoke-tools-lambda     lambda:InvokeFunction on ecom-nlq-sql (and its versions)

WHAT THIS DOES NOT DO
    It does not wait until IAM has propagated. A new role gets a 10 s head
    start, but a new role or a changed policy can take up to a minute to be
    usable — and a changed policy on an existing role gets no head start at
    all. The calls that use them (create_function, create_gateway_target)
    wait for IAM themselves, through infra/iam_ready.py.
"""
import json
import time

import boto3

iam = boto3.client("iam")

LAMBDA_ROLE_NAME = "ecom-nlq-sql-lambda-role"
GATEWAY_ROLE_NAME = "ecom-nlq-gateway-role"


def _trust(service: str) -> dict:
    return {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]}


def _ensure_role(name: str, service: str) -> str:
    try:
        return iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass
    print(f"  creating IAM role {name!r}")
    arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(_trust(service)))["Role"]["Arn"]
    time.sleep(10)   # a head start for IAM; iam_ready.py does the real waiting
    return arn


def _put(role: str, name: str, statements: list[dict]) -> None:
    iam.put_role_policy(RoleName=role, PolicyName=name, PolicyDocument=json.dumps(
        {"Version": "2012-10-17", "Statement": statements}))


def _read_secrets(resources: list[str]) -> list[dict]:
    return [{"Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": resources}]


def lambda_role(secret_prefixes: list[str]) -> str:
    arn = _ensure_role(LAMBDA_ROLE_NAME, "lambda.amazonaws.com")
    _put(LAMBDA_ROLE_NAME, "basic-execution", [{
        "Effect": "Allow", "Resource": "arn:aws:logs:*:*:*",
        "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]}])
    _put(LAMBDA_ROLE_NAME, "read-secrets", _read_secrets([f"{p}*" for p in secret_prefixes]))
    # a VPC Lambda creates and removes its own network interfaces in the subnets
    _put(LAMBDA_ROLE_NAME, "vpc-network-interfaces", [{
        "Effect": "Allow", "Resource": "*",
        "Action": ["ec2:CreateNetworkInterface", "ec2:DescribeNetworkInterfaces", "ec2:DeleteNetworkInterface",
                   "ec2:AssignPrivateIpAddresses", "ec2:UnassignPrivateIpAddresses"]}])
    return arn


def tighten_secret_policy(secret_arns: list[str]) -> None:
    _put(LAMBDA_ROLE_NAME, "read-secrets", _read_secrets(secret_arns))


def gateway_role(lambda_arn: str) -> str:
    """The Gateway's role: invoke THIS function only — unqualified, or any version/alias of it."""
    arn = _ensure_role(GATEWAY_ROLE_NAME, "bedrock-agentcore.amazonaws.com")
    _put(GATEWAY_ROLE_NAME, "invoke-tools-lambda", [{
        "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": [lambda_arn, f"{lambda_arn}:*"]}])
    return arn
