"""Step 3 — deploy the four agents to AgentCore Runtime.

    python 3_deploy_agents.py
    python 3_deploy_agents.py --only vdb

Each agent becomes its own runtime: its own container, its own scaling, its own
session isolation. That is the whole reason for A2A — four deployments that only
agree on a message format.

WHAT CHANGES FROM LOCAL, AND WHAT DOES NOT

Not the agents. `execute()` is identical, the cards are identical, the shape
contract is identical. What changes is one line of configuration: an endpoint
that was `http://127.0.0.1:9202` becomes a runtime ARN.

That is the test of whether the local module taught the right thing. If porting
required rewriting the agents, the local version was teaching a toy.

INVOCATION IS SIGV4

Runtime is invoked through the AgentCore data plane, signed with SigV4 from the
caller's IAM credentials. No Cognito, no token exchange for agent-to-agent.

The consequence is worth stating plainly: SigV4 authenticates an **IAM
principal**, not a person. The supervisor calls specialists as itself, so the
specialists cannot apply per-user authorisation — they do not know who asked.
Fine when every user sees the same data. A real limit if they should not, and
the fix then is JWT propagation, not SigV4.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import boto3

import config

LOCAL = Path(__file__).resolve().parent.parent / "local"


def deploy(agent: str) -> str:
    """Configure and launch one agent with the AgentCore toolkit.

    Shelling out rather than calling a Python API, because the CLI is what the
    documentation describes and what a student will find when they search. A
    wrapper that hides it makes the AWS docs harder to follow, not easier.
    """
    entry = LOCAL / "agents" / config.AGENTS[agent]["entry"]
    name = config.runtime_name(agent)

    print(f"\n── {agent}")
    subprocess.run(
        ["agentcore", "configure", "-e", str(entry), "--name", name,
         "--region", config.REGION, "--non-interactive"],
        cwd=LOCAL, check=True)

    result = subprocess.run(["agentcore", "launch", "--name", name],
                            cwd=LOCAL, check=True, capture_output=True, text=True)
    print(result.stdout[-400:])

    # The ARN is printed by launch and is also on the runtime itself. Reading it
    # back is more reliable than parsing stdout, which changes between versions.
    control = boto3.client("bedrock-agentcore-control", region_name=config.REGION)
    for runtime in control.list_agent_runtimes()["agentRuntimes"]:
        if runtime["agentRuntimeName"] == name:
            return runtime["agentRuntimeArn"]
    raise SystemExit(f"{name} deployed but not found in list_agent_runtimes")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy agents to AgentCore Runtime")
    ap.add_argument("--only", choices=list(config.AGENTS),
                    help="deploy one agent")
    args = ap.parse_args()

    state = json.loads(Path(config.STATE_FILE).read_text()) if Path(config.STATE_FILE).exists() else {}
    state.setdefault("runtimes", {})

    # Specialists first, supervisor last. The supervisor needs their ARNs in its
    # environment, so deploying it first would give it an empty fleet — and it
    # would start, and answer every question with "no agents reachable", which
    # looks like a routing bug rather than a deployment order problem.
    order = [args.only] if args.only else ["vdb", "cypher", "chart", "supervisor"]

    for agent in order:
        state["runtimes"][agent] = deploy(agent)
        Path(config.STATE_FILE).write_text(json.dumps(state, indent=2))
        print(f"  {agent}: {state['runtimes'][agent]}")

    print("\ndeployed:")
    for agent, arn in state["runtimes"].items():
        print(f"  {agent:<11} {arn}")
    print("\n  next: python 4_invoke.py \"which sponsors run more than one trial?\"")

    # The supervisor reads these to find its specialists. Locally it was a port
    # table in config.py; here it is environment variables on the runtime. The
    # agent code did not change — only where the endpoints come from.
    print("\nset these on the supervisor runtime:")
    for agent, arn in state["runtimes"].items():
        if agent != "supervisor":
            print(f"  {agent.upper()}_ENDPOINT={arn}")


if __name__ == "__main__":
    main()
