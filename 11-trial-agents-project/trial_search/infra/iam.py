"""IAM roles for trial_search's Gateway and tools Lambda.

    lambda_role()            logs + read on THREE secrets:
                               trial-search/openai     embed the query
                               trial-search/pinecone   search + fetch text
                               trial-graph/neo4j       NEXT traversal (Case A)
    tighten_secret_policy()  re-scopes the secret read to exact ARNs once
                             they exist (Secrets Manager appends a random
                             suffix, so the ARN is unknown until creation)
    gateway_role()           lets the Gateway invoke this one Lambda

WHY THE NEO4J SECRET IS trial_graph's, NOT A COPY

One database, one credential. A second secret holding the same password
drifts the first time either one is rotated. trial_graph/deploy.py owns
it; this role is only granted read on it.

WHAT THIS DOES NOT DO

    No S3 access. Chunk text comes from Pinecone metadata (written by
    chunking._finalise), not from any file in S3.
"""
import json
import time

import boto3

iam = boto3.client("iam")

LAMBDA_ROLE_NAME = "trial-search-tools-lambda-role"
GATEWAY_ROLE_NAME = "trial-search-gateway-role"


def _trust(service: str) -> dict:
    return {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]}


def _ensure_role(name: str, service: str) -> str:
    try:
        return iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass
    print(f"  creating IAM role {name!r}")
    arn = iam.create_role(RoleName=name,
                          AssumeRolePolicyDocument=json.dumps(_trust(service)))["Role"]["Arn"]
    time.sleep(10)   # IAM propagation: a brand-new role is not usable immediately
    return arn


def _put(role: str, name: str, statements: list[dict]) -> None:
    iam.put_role_policy(RoleName=role, PolicyName=name, PolicyDocument=json.dumps(
        {"Version": "2012-10-17", "Statement": statements}))


def _read_secrets(resources: list[str]) -> list[dict]:
    return [{"Effect": "Allow", "Action": "secretsmanager:GetSecretValue",
             "Resource": resources}]


def lambda_role(secret_prefixes: list[str]) -> str:
    arn = _ensure_role(LAMBDA_ROLE_NAME, "lambda.amazonaws.com")
    _put(LAMBDA_ROLE_NAME, "basic-execution", [{
        "Effect": "Allow", "Resource": "arn:aws:logs:*:*:*",
        "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]}])
    _put(LAMBDA_ROLE_NAME, "read-secrets", _read_secrets([f"{p}*" for p in secret_prefixes]))
    return arn


def tighten_secret_policy(secret_arns: list[str]) -> None:
    _put(LAMBDA_ROLE_NAME, "read-secrets", _read_secrets(secret_arns))


def gateway_role(lambda_arn: str) -> str:
    arn = _ensure_role(GATEWAY_ROLE_NAME, "bedrock-agentcore.amazonaws.com")
    _put(GATEWAY_ROLE_NAME, "invoke-tools-lambda", [{
        "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": lambda_arn}])
    return arn
