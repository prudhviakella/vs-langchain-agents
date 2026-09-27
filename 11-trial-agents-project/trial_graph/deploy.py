#!/usr/bin/env python3
"""One-click setup for trial_graph.

    python deploy.py

    STEP 1   secrets      trial-graph/neo4j, trial-agents/openai (placeholders if new)
    STEP 2   Lambda role  read the Neo4j secret
    STEP 3   Lambda       find_entity_by_name, validate_cypher, execute_cypher
    STEP 4   Gateway      AWS_IAM (SigV4); target created or updated
    STEP 5   guardrail    shared; a new version only if the policy changed
    STEP 6   prompt       prompts/system.md -> Prompt Management; version only on change
    STEP 7   parameters   /trial-agents/trial_graph/*
    STEP 8   runtime      role, linux/arm64 image, AgentCore Runtime
    STEP 9   registry     /trial-agents/registry/trial_graph = {arn, description}
    STEP 10  Neo4j index  the fulltext index find_entity_by_name queries
    STEP 11  deployment.json

The runtime's only environment variable is PARAM_PREFIX. Everything else it
reads from AWS at start (see agent_code/trial_graph/config.py), so changing
a limit, the prompt version or the model name needs no rebuild — only a new
container, which the next cold start provides.

Before STEP 1, CloudWatch Transaction Search is enabled if it is not already —
without it, no agent's spans appear in CloudWatch (infra/observability.py).

DEPLOY ORDER:  trial_graph -> trial_search -> supervisor

WHAT THIS DOES NOT DO

    It does not set real credentials. It does not grant permissions or install
    Docker — infra/preflight.py checks both first and stops, naming what is
    missing, before anything is created. It logs in to ECR by itself.
"""
import json
from pathlib import Path

import boto3

from infra import (preflight, observability, config_store, gateway, guardrail, iam, lambda_deploy, prompts,
                   runtime_deploy, runtime_iam)

AGENT = "trial_graph"
HERE = Path(__file__).parent
NEO4J_SECRET = "trial-graph/neo4j"
DESCRIPTION = ("Relationships and registry facts about the 20 trials: sponsors, CROs, "
               "sites and countries, phase and status, conditions, listed outcomes and "
               "registry eligibility fields. Answers by traversing a Neo4j graph and can "
               "return a trial's nctId and its protocol's docId. Holds no protocol text "
               "and no drug or intervention data.")
LIMITS = {"row_cap": 500, "graph_node_cap": 500, "max_repairs": 3}


def main() -> None:
    preflight.run(needs_gateway=True)
    prefix = config_store.prefix(AGENT)

    print("=== observability: CloudWatch Transaction Search (once per account) ===")
    observability.ensure_transaction_search()

    print("=== STEP 1: secrets ===")
    neo4j_arn = config_store.ensure_secret(
        NEO4J_SECRET, {"uri": "replace-me", "user": "neo4j", "password": "replace-me"})
    openai_arn = config_store.ensure_openai_secret()

    print("\n=== STEP 2: Lambda role ===")
    lambda_role_arn = iam.lambda_role(f"arn:aws:secretsmanager:*:*:secret:{NEO4J_SECRET}")
    iam.tighten_secret_policy(neo4j_arn)

    print("\n=== STEP 3: tools Lambda ===")
    lambda_arn = lambda_deploy.deploy(lambda_role_arn, NEO4J_SECRET)

    print("\n=== STEP 4: Gateway ===")
    gw = gateway.create_gateway(iam.gateway_role(lambda_arn))
    # After create_gateway: the permission's SourceArn is the gateway's own arn.
    lambda_deploy.allow_gateway_invoke(lambda_deploy.FUNCTION_NAME, gw["gatewayArn"])
    gateway.create_lambda_target(gw["gatewayId"], lambda_arn)

    print("\n=== STEP 5: guardrail ===")
    gr = guardrail.ensure_guardrail()

    print("\n=== STEP 6: prompt ===")
    prompt = prompts.publish("trial-graph-system", HERE / "prompts" / "system.md",
                             "trial_graph system prompt")

    print("\n=== STEP 7: parameters ===")
    config_store.put_parameters(prefix, {
        "gateway_url": gw["gatewayUrl"], **LIMITS,
        "guardrail_id": gr["id"], "guardrail_version": gr["version"],
        "prompt_id": prompt["id"], "prompt_version": prompt["version"],
        "openai_secret_id": config_store.OPENAI_SECRET})

    print("\n=== STEP 8: runtime ===")
    repo_uri, repo_arn = runtime_deploy.ensure_ecr_repo()
    role_arn = runtime_iam.runtime_role(
        gateway_arn=gw["gatewayArn"], ecr_repo_arn=repo_arn, guardrail_arn=gr["arn"],
        prompt_arns=[prompt["arn"]], secret_arn=openai_arn, param_prefix=prefix)
    image_uri = runtime_deploy.build_and_push(repo_uri, str(HERE / "agent_code"))
    runtime_arn = runtime_deploy.deploy_runtime(image_uri, role_arn, {
        "PARAM_PREFIX": prefix,
        "AWS_REGION": boto3.Session().region_name or "us-east-1"})

    print("\n=== STEP 9: registry ===")
    config_store.register(AGENT, runtime_arn, DESCRIPTION)

    print("\n=== STEP 10: Neo4j fulltext index ===")
    if config_store.placeholders(NEO4J_SECRET):
        print("  skipped — the Neo4j secret is still a placeholder. After setting it:")
        print("    python setup_neo4j.py")
    else:
        import setup_neo4j   # needs the neo4j driver only when it actually runs
        setup_neo4j.ensure_index()

    print("\n=== STEP 11: deployment.json ===")
    (HERE / "deployment.json").write_text(json.dumps({
        "runtime_arn": runtime_arn, "param_prefix": prefix,
        "gateway_url": gw["gatewayUrl"], "gateway_arn": gw["gatewayArn"],
        "guardrail": gr, "prompt": prompt, "limits": LIMITS}, indent=2))

    _remind(NEO4J_SECRET, config_store.OPENAI_SECRET)
    print(f"\ndone. {AGENT} runtime: {runtime_arn}")


def _remind(*secrets: str) -> None:
    for name in secrets:
        unset = config_store.placeholders(name)
        if unset:
            print(f"\nREMINDER: {name} still has placeholder {unset}. The agent refuses "
                  "to start until they are set:")
            print(f"  aws secretsmanager put-secret-value --secret-id {name} "
                  "--secret-string '{...}'")


if __name__ == "__main__":
    main()
