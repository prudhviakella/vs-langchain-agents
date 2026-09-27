#!/usr/bin/env python3
"""One-click setup for the Supervisor.

    python deploy.py

    STEP 1  registry     read /trial-agents/registry/* — every specialist that has
                         deployed. Stops here if there are none.
    STEP 2  secret       trial-agents/openai (placeholder if new)
    STEP 3  guardrail    shared; a new version only if the policy changed
    STEP 4  prompts      prompts/system.md and prompts/compose.md
                         -> Prompt Management; a version only on change
    STEP 5  parameters   /trial-agents/supervisor/*
    STEP 6  runtime      role scoped to exactly the registered specialists,
                         linux/arm64 image, AgentCore Runtime
    STEP 7  deployment.json

WHY THE SUPERVISOR IS DEPLOYED LAST, AND AGAIN WHEN A SPECIALIST IS ADDED

It learns its specialists from the registry at container start, so a new
specialist needs no code change. But its IAM role is scoped to the ARNs
registered at deploy time — so redeploy the supervisor after adding one,
and its role is re-scoped to include it.

DEPLOY ORDER:  trial_graph -> trial_search -> supervisor

WHAT THIS DOES NOT DO

    It creates no Gateway or Lambda. Its one tool, call_agent, invokes the
    specialists' runtimes directly (agent_code/supervisor/agent_client.py).
"""
import json
from pathlib import Path

import boto3

from infra import config_store, guardrail, prompts, runtime_deploy, runtime_iam

AGENT = "supervisor"
HERE = Path(__file__).parent
LIMITS = {"max_agent_calls_per_turn": 6}


def main() -> None:
    prefix = config_store.prefix(AGENT)

    print("=== STEP 1: registry ===")
    specialists = config_store.read_registry()
    if not specialists:
        raise SystemExit(f"nothing registered under {config_store.REGISTRY_PATH}. Deploy "
                         "trial_graph and trial_search first — each registers itself.")
    for name, spec in sorted(specialists.items()):
        print(f"  {name:14} {spec['arn']}")

    print("\n=== STEP 2: secret ===")
    openai_arn = config_store.ensure_openai_secret()

    print("\n=== STEP 3: guardrail ===")
    gr = guardrail.ensure_guardrail()

    print("\n=== STEP 4: prompts ===")
    system = prompts.publish("supervisor-system", HERE / "prompts" / "system.md",
                             "supervisor system prompt")
    compose = prompts.publish("supervisor-compose", HERE / "prompts" / "compose.md",
                              "supervisor composer instruction")

    print("\n=== STEP 5: parameters ===")
    config_store.put_parameters(prefix, {
        **LIMITS, "registry_path": config_store.REGISTRY_PATH,
        "guardrail_id": gr["id"], "guardrail_version": gr["version"],
        "prompt_id": system["id"], "prompt_version": system["version"],
        "compose_prompt_id": compose["id"], "compose_prompt_version": compose["version"],
        "openai_secret_id": config_store.OPENAI_SECRET})

    print("\n=== STEP 6: runtime ===")
    repo_uri, repo_arn = runtime_deploy.ensure_ecr_repo()
    role_arn = runtime_iam.runtime_role(
        specialist_arns=[s["arn"] for s in specialists.values()], ecr_repo_arn=repo_arn,
        guardrail_arn=gr["arn"], prompt_arns=[system["arn"], compose["arn"]],
        secret_arn=openai_arn, param_prefix=prefix,
        registry_path=config_store.REGISTRY_PATH)
    image_uri = runtime_deploy.build_and_push(repo_uri, str(HERE / "agent_code"))
    runtime_arn = runtime_deploy.deploy_runtime(image_uri, role_arn, {
        "PARAM_PREFIX": prefix,
        "AWS_REGION": boto3.Session().region_name or "us-east-1"})

    print("\n=== STEP 7: deployment.json ===")
    (HERE / "deployment.json").write_text(json.dumps({
        "runtime_arn": runtime_arn, "param_prefix": prefix,
        "specialists": specialists, "guardrail": gr,
        "prompts": {"system": system, "compose": compose}, "limits": LIMITS}, indent=2))

    unset = config_store.placeholders(config_store.OPENAI_SECRET)
    if unset:
        print(f"\nREMINDER: {config_store.OPENAI_SECRET} still has placeholder {unset}.")
    print(f"\ndone. supervisor runtime: {runtime_arn}")


if __name__ == "__main__":
    main()
