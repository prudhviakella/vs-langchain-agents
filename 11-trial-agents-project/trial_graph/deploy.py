#!/usr/bin/env python3
"""One-click setup for trial_graph's AgentCore Gateway infrastructure.

    python deploy.py

    IAM roles                                                    (STEP 1)
        |
        v
    Neo4j credential secret (created once, "replace-me" until you        (STEP 2)
    set it for real — same pattern as the RAG pipeline's own deploy.py)
        |
        v
    tools Lambda (zipped with the neo4j driver, deployed)                (STEP 3)
        |
        v
    AgentCore Gateway + Lambda target (MCP-exposes the three tools)      (STEP 4)
        |
        v
    Bedrock Guardrail                                                    (STEP 5)
        |
        v
    ECR repo + build/push linux/arm64 image + AgentCore Runtime          (STEP 6)
    (the agent's own compute — separate IAM role from the Gateway/
    Lambda above, since it's a different trust boundary)
        |
        v
    writes gateway_config.json — read by anything that needs to call     (STEP 7)
    this agent (the Supervisor) or configure it further

Every step is idempotent, exactly like the RAG pipeline's own deploy.py:
re-running this after a partial failure resumes rather than duplicating
anything, since every create_* call here checks for an existing resource
by name first.

WHAT THIS DOES NOT DO

    - It does not build the Docker image without docker/buildx already
      logged into ECR — run `aws ecr get-login-password | docker login`
      once per session before running this, same as any other buildx
      push workflow.
    - It does not set real Neo4j credentials. Run
      scripts/set_neo4j_secret.py afterward with the real URI/password —
      the Lambda will raise a clear connection error until you do,
      rather than silently succeed against nothing.
"""
import json
from pathlib import Path

import boto3

from infra import gateway, guardrail, iam, lambda_deploy, runtime_deploy, runtime_iam

OUTPUT_PATH = Path(__file__).parent / "gateway_config.json"
SECRET_NAME = "trial-graph/neo4j"
MODEL_ID = "us.anthropic.claude-sonnet-4-6-20251001-v1:0"


def ensure_secret() -> str:
    """Create the Neo4j credential secret if it doesn't exist yet, as a
    clearly-fake placeholder — same "replace-me" pattern as the RAG
    pipeline's own deploy.py, so a forgotten setup step fails loudly at
    connection time instead of silently succeeding against nothing.
    """
    client = boto3.client("secretsmanager")
    try:
        secret = client.describe_secret(SecretId=SECRET_NAME)
        return secret["ARN"]
    except client.exceptions.ResourceNotFoundException:
        pass

    print(f"  creating placeholder secret {SECRET_NAME!r}")
    response = client.create_secret(
        Name=SECRET_NAME,
        SecretString=json.dumps({"uri": "replace-me", "user": "neo4j",
                                 "password": "replace-me"}),
    )
    return response["ARN"]


def main() -> None:
    print("=== STEP 1: IAM roles ===")
    secret_arn_prefix = f"arn:aws:secretsmanager:*:*:secret:{SECRET_NAME}"
    lambda_role_arn = iam.lambda_role(secret_arn_prefix)

    print("\n=== STEP 2: Neo4j credential secret ===")
    secret_arn = ensure_secret()
    iam.tighten_secret_policy(secret_arn)

    print("\n=== STEP 3: tools Lambda ===")
    lambda_arn = lambda_deploy.deploy(lambda_role_arn, SECRET_NAME)

    print("\n=== STEP 4: AgentCore Gateway + Lambda target ===")
    gateway_role_arn = iam.gateway_role(lambda_arn)
    gw = gateway.create_gateway(gateway_role_arn)
    # After create_gateway, not before: the resource policy's SourceArn is
    # the gateway's own arn, which does not exist until it is created.
    lambda_deploy.allow_gateway_invoke(lambda_deploy.FUNCTION_NAME, gw["gatewayArn"])
    gateway.create_lambda_target(gw["gatewayId"], lambda_arn)

    print("\n=== STEP 5: Bedrock Guardrail ===")
    gr = guardrail.create_guardrail()

    print("\n=== STEP 6: ECR + build/push + AgentCore Runtime ===")
    repo_uri, repo_arn = runtime_deploy.ensure_ecr_repo()
    runtime_role_arn = runtime_iam.runtime_role(
        gateway_arn=gw["gatewayArn"], ecr_repo_arn=repo_arn,
        model_id=MODEL_ID, guardrail_arn=gr["guardrailArn"])
    dockerfile_dir = str(Path(__file__).parent / "agent_code")
    image_uri = runtime_deploy.build_and_push(repo_uri, dockerfile_dir)
    env_vars = {
        "GATEWAY_URL": gw["gatewayUrl"],
        "AWS_REGION": boto3.Session().region_name or "us-east-1",
        "GUARDRAIL_ID": gr["guardrailId"],
        "GUARDRAIL_VERSION": gr["version"],
        "MODEL_ID": MODEL_ID,
    }
    runtime_arn = runtime_deploy.deploy_runtime(image_uri, runtime_role_arn, env_vars)

    print("\n=== STEP 7: writing gateway_config.json ===")
    config = {
        "gateway_url": gw["gatewayUrl"],
        "gateway_id": gw["gatewayId"],
        "gateway_arn": gw["gatewayArn"],
        "guardrail_id": gr["guardrailId"],
        "guardrail_arn": gr["guardrailArn"],
        "guardrail_version": gr["version"],
        "neo4j_secret_name": SECRET_NAME,
        "runtime_arn": runtime_arn,
        "model_id": MODEL_ID,
    }
    OUTPUT_PATH.write_text(json.dumps(config, indent=2))
    print(f"  wrote {OUTPUT_PATH}")

    print("\ndone.")
    if json.loads(boto3.client("secretsmanager").get_secret_value(
            SecretId=SECRET_NAME)["SecretString"])["uri"] == "replace-me":
        print("\nREMINDER: the Neo4j secret is still a placeholder. Run:")
        print(f"  aws secretsmanager put-secret-value --secret-id {SECRET_NAME} "
             '--secret-string \'{"uri":"neo4j+s://...","user":"neo4j",'
             '"password":"..."}\'')


if __name__ == "__main__":
    main()
