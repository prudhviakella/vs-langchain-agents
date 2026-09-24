#!/usr/bin/env python3
"""One-click setup for trial_search.

    python deploy.py --pinecone-index rag-docs

    STEP 1  secrets        trial-search/openai, trial-search/pinecone
                           (placeholders if new); trial-graph/neo4j must
                           already exist — owned by trial_graph/deploy.py
    STEP 2  IAM            Lambda role scoped to those three exact ARNs
    STEP 3  Lambda         three tools, Linux-built dependencies
    STEP 4  Gateway        AWS_IAM auth; target created or UPDATED with
                           all three tool schemas
    STEP 5  Guardrail
    STEP 6  Runtime        ECR + linux/arm64 image + AgentCore Runtime,
                           budgets passed as environment variables
    STEP 7  gateway_config.json   read by supervisor/deploy.py

DEPLOY ORDER

    trial_graph/deploy.py  ->  trial_search/deploy.py  ->  supervisor/deploy.py

trial_search needs trial_graph's Neo4j secret for expand_neighbors (the
NEXT traversal). This script stops with a clear message if it is missing.

WHAT THIS DOES NOT DO

    It does not log in to ECR. Run once per shell before deploying:
      aws ecr get-login-password | docker login --username AWS \
          --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
"""
import argparse
import json
from pathlib import Path

import boto3

from infra import gateway, guardrail, iam, lambda_deploy, runtime_deploy, runtime_iam

OUTPUT_PATH = Path(__file__).parent / "gateway_config.json"
OPENAI_SECRET = "trial-search/openai"
PINECONE_SECRET = "trial-search/pinecone"
NEO4J_SECRET = "trial-graph/neo4j"          # owned by trial_graph
MODEL_ID = "us.anthropic.claude-sonnet-4-6-20251001-v1:0"

# Loop budgets. Passed to the Runtime as environment variables so each
# environment (dev/tst/prd) can tune them without a code change.
BUDGETS = {
    "MAX_SEARCHES_PER_TURN": "5",
    "MAX_NEIGHBOR_CALLS": "3",
    "MAX_TABLE_CALLS": "3",
    "MAX_WINDOW": "10",
    "EXPANSION_TOKEN_BUDGET": "6000",
}

sm = boto3.client("secretsmanager")


def ensure_placeholder(name: str) -> str:
    try:
        return sm.describe_secret(SecretId=name)["ARN"]
    except sm.exceptions.ResourceNotFoundException:
        print(f"  creating placeholder secret {name!r}")
        return sm.create_secret(Name=name, SecretString=json.dumps(
            {"api_key": "replace-me"}))["ARN"]


def require(name: str, owner: str) -> str:
    try:
        return sm.describe_secret(SecretId=name)["ARN"]
    except sm.exceptions.ResourceNotFoundException:
        raise SystemExit(f"secret {name!r} does not exist. Run {owner}/deploy.py first — "
                         "trial_search reads the same Neo4j credential.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinecone-index", default="rag-docs")
    args = ap.parse_args()
    region = boto3.Session().region_name or "us-east-1"

    print("=== STEP 1: secrets ===")
    arns = [ensure_placeholder(OPENAI_SECRET), ensure_placeholder(PINECONE_SECRET),
            require(NEO4J_SECRET, "trial_graph")]

    print("\n=== STEP 2: IAM ===")
    lambda_role_arn = iam.lambda_role([f"arn:aws:secretsmanager:*:*:secret:{n}"
                                       for n in (OPENAI_SECRET, PINECONE_SECRET, NEO4J_SECRET)])
    iam.tighten_secret_policy(arns)

    print("\n=== STEP 3: tools Lambda ===")
    lambda_arn = lambda_deploy.deploy(lambda_role_arn, {
        "OPENAI_SECRET_ID": OPENAI_SECRET, "PINECONE_SECRET_ID": PINECONE_SECRET,
        "NEO4J_SECRET_ID": NEO4J_SECRET, "PINECONE_INDEX": args.pinecone_index})

    print("\n=== STEP 4: Gateway ===")
    gw = gateway.create_gateway(iam.gateway_role(lambda_arn))
    lambda_deploy.allow_gateway_invoke(gw["gatewayArn"])
    gateway.create_lambda_target(gw["gatewayId"], lambda_arn)

    print("\n=== STEP 5: Guardrail ===")
    gr = guardrail.create_guardrail()

    print("\n=== STEP 6: Runtime ===")
    repo_uri, repo_arn = runtime_deploy.ensure_ecr_repo()
    runtime_role_arn = runtime_iam.runtime_role(
        gateway_arn=gw["gatewayArn"], ecr_repo_arn=repo_arn,
        model_id=MODEL_ID, guardrail_arn=gr["guardrailArn"])
    image_uri = runtime_deploy.build_and_push(repo_uri, str(Path(__file__).parent / "agent_code"))
    runtime_arn = runtime_deploy.deploy_runtime(image_uri, runtime_role_arn, {
        "GATEWAY_URL": gw["gatewayUrl"], "AWS_REGION": region,
        "GUARDRAIL_ID": gr["guardrailId"], "GUARDRAIL_VERSION": gr["version"],
        "MODEL_ID": MODEL_ID, **BUDGETS})

    print("\n=== STEP 7: gateway_config.json ===")
    OUTPUT_PATH.write_text(json.dumps({
        "runtime_arn": runtime_arn, "gateway_url": gw["gatewayUrl"],
        "gateway_id": gw["gatewayId"], "gateway_arn": gw["gatewayArn"],
        "guardrail_id": gr["guardrailId"], "guardrail_arn": gr["guardrailArn"],
        "guardrail_version": gr["version"], "model_id": MODEL_ID,
        "tools": [t["name"] for t in gateway.TOOL_SCHEMA], "budgets": BUDGETS,
    }, indent=2))
    print(f"  wrote {OUTPUT_PATH}")

    placeholders = [n for n in (OPENAI_SECRET, PINECONE_SECRET)
                    if json.loads(sm.get_secret_value(SecretId=n)["SecretString"])
                    .get("api_key") == "replace-me"]
    if placeholders:
        print("\nREMINDER — still placeholders:")
        for n in placeholders:
            print(f"  aws secretsmanager put-secret-value --secret-id {n} "
                  "--secret-string '{\"api_key\":\"...\"}'")
    print(f"\ndone. trial_search runtime: {runtime_arn}")


if __name__ == "__main__":
    main()
