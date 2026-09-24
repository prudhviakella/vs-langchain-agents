#!/usr/bin/env python3
"""One-click setup for the Supervisor's AgentCore Runtime.

    python deploy.py

Run this AFTER trial_graph/deploy.py and trial_search/deploy.py have
both completed — this script reads their runtime_arn straight out of
their own gateway_config.json files (sibling folders, ../trial_graph/
and ../trial_search/) rather than asking for them again on the command
line. If either file is missing or has no runtime_arn yet, this fails
immediately with a clear message telling you which one to run first,
rather than deploying a Supervisor that can't reach its own specialists.

    read ../trial_graph/gateway_config.json,                    (STEP 1)
    ../trial_search/gateway_config.json for their runtime_arn
        |
        v
    Bedrock Guardrail                                            (STEP 2)
        |
        v
    IAM role — bedrock-agentcore:InvokeAgentRuntime on BOTH       (STEP 3)
    specialist ARNs (not InvokeGateway — the Supervisor has no
    Gateway of its own; see infra/runtime_iam.py)
        |
        v
    ECR + build/push linux/arm64 image + AgentCore Runtime        (STEP 4)
        |
        v
    writes supervisor_config.json                                (STEP 5)

WHAT THIS DOES NOT DO

    - It does not create a Gateway, a Lambda, or any tool-execution
      infrastructure. The Supervisor's one tool (call_agent) calls
      trial_graph/trial_search directly through AgentCore Runtime's own
      data plane — a genuinely different integration path from how
      trial_graph/trial_search reach their own tools. See
      agent_code/supervisor/agent_client.py for the full reasoning.
"""
import json
from pathlib import Path

import boto3

from infra import guardrail, runtime_deploy, runtime_iam

OUTPUT_PATH = Path(__file__).parent / "supervisor_config.json"
MODEL_ID = "us.anthropic.claude-sonnet-4-6-20251001-v1:0"

TRIAL_GRAPH_CONFIG = Path(__file__).parent.parent / "trial_graph" / "gateway_config.json"
TRIAL_SEARCH_CONFIG = Path(__file__).parent.parent / "trial_search" / "gateway_config.json"


def _read_runtime_arn(config_path: Path, agent_name: str) -> str:
    if not config_path.exists():
        raise SystemExit(
            f"{config_path} does not exist. Run {agent_name}/deploy.py first — "
            "the Supervisor needs its runtime_arn before it can be deployed.")
    config = json.loads(config_path.read_text())
    arn = config.get("runtime_arn")
    if not arn:
        raise SystemExit(
            f"{config_path} has no runtime_arn yet. Re-run {agent_name}/deploy.py "
            "to completion first.")
    return arn


def main() -> None:
    print("=== STEP 1: reading specialist runtime ARNs ===")
    trial_graph_arn = _read_runtime_arn(TRIAL_GRAPH_CONFIG, "trial_graph")
    trial_search_arn = _read_runtime_arn(TRIAL_SEARCH_CONFIG, "trial_search")
    print(f"  trial_graph:  {trial_graph_arn}")
    print(f"  trial_search: {trial_search_arn}")

    print("\n=== STEP 2: Bedrock Guardrail ===")
    gr = guardrail.create_guardrail()

    print("\n=== STEP 3: IAM role ===")
    repo_uri, repo_arn = runtime_deploy.ensure_ecr_repo()
    runtime_role_arn = runtime_iam.runtime_role(
        trial_graph_arn=trial_graph_arn, trial_search_arn=trial_search_arn,
        ecr_repo_arn=repo_arn, model_id=MODEL_ID, guardrail_arn=gr["guardrailArn"])

    print("\n=== STEP 4: ECR + build/push + AgentCore Runtime ===")
    dockerfile_dir = str(Path(__file__).parent / "agent_code")
    image_uri = runtime_deploy.build_and_push(repo_uri, dockerfile_dir)
    env_vars = {
        "TRIAL_GRAPH_RUNTIME_ARN": trial_graph_arn,
        "TRIAL_SEARCH_RUNTIME_ARN": trial_search_arn,
        "AWS_REGION": boto3.Session().region_name or "us-east-1",
        "GUARDRAIL_ID": gr["guardrailId"],
        "GUARDRAIL_VERSION": gr["version"],
        "MODEL_ID": MODEL_ID,
    }
    runtime_arn = runtime_deploy.deploy_runtime(image_uri, runtime_role_arn, env_vars)

    print("\n=== STEP 5: writing supervisor_config.json ===")
    config = {
        "runtime_arn": runtime_arn,
        "guardrail_id": gr["guardrailId"], "guardrail_arn": gr["guardrailArn"],
        "guardrail_version": gr["version"],
        "trial_graph_runtime_arn": trial_graph_arn,
        "trial_search_runtime_arn": trial_search_arn,
        "model_id": MODEL_ID,
    }
    OUTPUT_PATH.write_text(json.dumps(config, indent=2))
    print(f"  wrote {OUTPUT_PATH}")

    print(f"\ndone. Supervisor runtime: {runtime_arn}")


if __name__ == "__main__":
    main()
