"""What is deployed right now.

    python status.py

Reads AWS rather than deployed.json, because the file records what a script
intended and the account records what exists. They differ after a failed deploy,
and it is the difference that is worth seeing.
"""

import json
from pathlib import Path

import boto3

import config


def main() -> None:
    control = boto3.client("bedrock-agentcore-control", region_name=config.REGION)
    lam = boto3.client("lambda", region_name=config.REGION)

    recorded = (json.loads(Path(config.STATE_FILE).read_text())
                if Path(config.STATE_FILE).exists() else {})

    print(f"{'agent':<12}{'recorded':<10}{'in aws':<10}status")
    print("-" * 52)
    try:
        live = {r["agentRuntimeName"]: r
                for r in control.list_agent_runtimes()["agentRuntimes"]}
    except Exception as exc:
        print(f"  could not list runtimes: {exc}")
        live = {}

    for agent in config.AGENTS:
        name = config.runtime_name(agent)
        in_file = agent in recorded.get("runtimes", {})
        in_aws = name in live
        status = live.get(name, {}).get("status", "-")
        print(f"{agent:<12}{str(in_file):<10}{str(in_aws):<10}{status}")

    print()
    try:
        fn = lam.get_function(FunctionName=config.LAMBDA_NAME)["Configuration"]
        print(f"lambda    {config.LAMBDA_NAME}  {fn['State']}  "
              f"{fn['CodeSize']:,} bytes")
    except Exception:
        print(f"lambda    {config.LAMBDA_NAME}  not deployed")

    if recorded.get("gateway_id"):
        print(f"gateway   {recorded['gateway_id']}")
        print(f"          {recorded.get('gateway_url', '')}")
    else:
        print("gateway   not deployed")


if __name__ == "__main__":
    main()
