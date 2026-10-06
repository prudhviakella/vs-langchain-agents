#!/usr/bin/env python3
"""One-click setup for trial_search.

    python deploy.py --pinecone-index rag-docs

    STEP 1   secrets      trial-search/pinecone, trial-search/cohere,
                          trial-agents/openai (placeholders if new);
                          trial-graph/neo4j must exist — owned by trial_graph
    STEP 2   Lambda role  read those four secrets
    STEP 3   Lambda       resolve_trial, semantic_search, expand_neighbors, expand_table
    STEP 4   Gateway      AWS_IAM (SigV4); target created or updated
    STEP 5   guardrail    shared; a new version only if the policy changed
    STEP 6   prompt       prompts/system.md -> Prompt Management; version only on change
    STEP 7   parameters   /trial-agents/trial_search/* — the budgets live here, and
                          are rendered into the prompt at start, so the prompt
                          states the limits the middleware actually enforces
    STEP 8   runtime      role, linux/arm64 image, AgentCore Runtime
    STEP 9   registry     /trial-agents/registry/trial_search = {arn, description}
    STEP 10  deployment.json

ONE OPENAI SECRET

The Lambda embeds queries with the same OpenAI key the agent's chat model
uses: trial-agents/openai. An earlier version kept a separate
trial-search/openai secret — two copies of one key, which drift the first
time only one is rotated.

Before STEP 1, CloudWatch Transaction Search is enabled if it is not already —
without it, no agent's spans appear in CloudWatch (infra/observability.py).

DEPLOY ORDER:  trial_graph -> trial_search -> supervisor

resolve_trial reads the Neo4j fulltext index trial_entity_names. It is created
by trial_graph/setup_neo4j.py — run that once before this deploy.
"""
import argparse
import json
from pathlib import Path

import boto3

from infra import (preflight, observability, config_store, gateway, guardrail, iam, lambda_deploy, prompts,
                   runtime_deploy, runtime_iam)

AGENT = "trial_search"
HERE = Path(__file__).parent
PINECONE_SECRET = "trial-search/pinecone"
# Cohere Rerank, called by the Lambda's semantic_search. Until the real key is
# set, searches still work, in vector order, marked reranked=false.
COHERE_SECRET = "trial-search/cohere"
RERANK_POOL = 40
NEO4J_SECRET = "trial-graph/neo4j"          # owned by trial_graph
# Read by the supervisor's routing model (rendered into its AVAILABLE AGENTS).
# It describes a capability, not the corpus: no trial names or counts, so it
# never goes stale when protocols are added.
DESCRIPTION = ("What the trial protocols actually say: eligibility wording, study "
               "design, endpoint definitions, dosing, safety and adverse event sections, "
               "and the values inside protocol tables. Answers by retrieving passages. "
               "Resolves a trial named in the question — NCT number, acronym, title "
               "words, drug or condition — to its protocol itself, through the "
               "registry graph's name index. Does not answer sponsor, site, phase or "
               "other registry questions.")
BUDGETS = {"max_resolve_calls": 3, "max_searches_per_turn": 5, "max_neighbor_calls": 3, "max_table_calls": 3,
           "max_window": 10, "expansion_token_budget": 6000}


def _require(name: str, owner: str) -> str:
    try:
        return boto3.client("secretsmanager").describe_secret(SecretId=name)["ARN"]
    except boto3.client("secretsmanager").exceptions.ResourceNotFoundException:
        raise SystemExit(f"secret {name!r} does not exist. Run {owner}/deploy.py first — "
                         "expand_neighbors reads the same Neo4j credential.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinecone-index", default="rag-docs")
    args = ap.parse_args()
    preflight.run(needs_gateway=True)
    prefix = config_store.prefix(AGENT)

    print("=== observability: CloudWatch Transaction Search (once per account) ===")
    observability.ensure_transaction_search()

    print("=== STEP 1: secrets ===")
    openai_arn = config_store.ensure_openai_secret()
    pinecone_arn = config_store.ensure_secret(PINECONE_SECRET, {"api_key": "replace-me"})
    cohere_arn = config_store.ensure_secret(COHERE_SECRET, {"api_key": "replace-me",
                                                            "model": "rerank-v3.5"})
    neo4j_arn = _require(NEO4J_SECRET, "trial_graph")

    print("\n=== STEP 2: Lambda role ===")
    lambda_role_arn = iam.lambda_role([f"arn:aws:secretsmanager:*:*:secret:{n}"
                                       for n in (config_store.OPENAI_SECRET, PINECONE_SECRET,
                                                 COHERE_SECRET, NEO4J_SECRET)])
    iam.tighten_secret_policy([openai_arn, pinecone_arn, cohere_arn, neo4j_arn])

    print("\n=== STEP 3: tools Lambda ===")
    lambda_arn = lambda_deploy.deploy(lambda_role_arn, {
        "OPENAI_SECRET_ID": config_store.OPENAI_SECRET, "PINECONE_SECRET_ID": PINECONE_SECRET,
        "NEO4J_SECRET_ID": NEO4J_SECRET, "PINECONE_INDEX": args.pinecone_index,
        "COHERE_SECRET_ID": COHERE_SECRET, "RERANK_POOL": str(RERANK_POOL)})

    print("\n=== STEP 4: Gateway ===")
    gw = gateway.create_gateway(iam.gateway_role(lambda_arn))
    lambda_deploy.allow_gateway_invoke(gw["gatewayArn"])
    gateway.create_lambda_target(gw["gatewayId"], lambda_arn)

    print("\n=== STEP 5: guardrail ===")
    gr = guardrail.ensure_guardrail()

    print("\n=== STEP 6: prompt ===")
    prompt = prompts.publish("trial-search-system", HERE / "prompts" / "system.md",
                             "trial_search system prompt")

    print("\n=== STEP 7: parameters ===")
    config_store.put_parameters(prefix, {
        "gateway_url": gw["gatewayUrl"], **BUDGETS,
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

    print("\n=== STEP 10: deployment.json ===")
    (HERE / "deployment.json").write_text(json.dumps({
        "runtime_arn": runtime_arn, "param_prefix": prefix,
        "gateway_url": gw["gatewayUrl"], "gateway_arn": gw["gatewayArn"],
        "guardrail": gr, "prompt": prompt, "budgets": BUDGETS}, indent=2))

    for name in (config_store.OPENAI_SECRET, PINECONE_SECRET, COHERE_SECRET):
        unset = config_store.placeholders(name)
        if unset:
            print(f"\nREMINDER: {name} still has placeholder {unset}.")
    print(f"\ndone. {AGENT} runtime: {runtime_arn}")


if __name__ == "__main__":
    main()
