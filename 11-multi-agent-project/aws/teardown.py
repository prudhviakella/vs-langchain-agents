"""Delete everything the deploy scripts created.

    python teardown.py            prompts first
    python teardown.py --yes

Order matters. The Gateway target references the Lambda and the Gateway
references the target, so deleting the Lambda first leaves a Gateway pointing at
nothing — which fails at invoke time rather than at delete time.
"""

import argparse
import json
from pathlib import Path

import boto3

import config


def attempt(label: str, fn, **kwargs) -> None:
    """Delete one thing, reporting rather than raising if it is already gone.

    Teardown has to be resumable. A partial deploy, or a second teardown after
    an interrupted first, leaves some resources missing — and stopping on the
    first NotFound strands everything after it, which is exactly the resources
    that cost money.
    """
    try:
        fn(**kwargs)
        print(f"  deleted {label}")
    except Exception as exc:
        print(f"  skipped {label} ({type(exc).__name__})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete every deployed resource")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if not Path(config.STATE_FILE).exists():
        raise SystemExit("nothing to tear down — no deployed.json")
    state = json.loads(Path(config.STATE_FILE).read_text())

    print(f"about to delete every {config.PROJECT} resource in {config.REGION}")
    if not args.yes and input(f"type the project name to confirm: ") != config.PROJECT:
        raise SystemExit("aborted")

    control = boto3.client("bedrock-agentcore-control", region_name=config.REGION)
    lam = boto3.client("lambda", region_name=config.REGION)
    iam = boto3.client("iam", region_name=config.REGION)

    print("\n── runtimes")
    for agent, arn in state.get("runtimes", {}).items():
        attempt(agent, control.delete_agent_runtime, agentRuntimeId=arn.split("/")[-1])

    print("\n── gateway")
    gateway_id = state.get("gateway_id")
    if gateway_id:
        # Targets before the gateway. A gateway with a live target cannot be
        # deleted, and the error names the gateway rather than the target.
        try:
            for target in control.list_gateway_targets(
                    gatewayIdentifier=gateway_id)["items"]:
                attempt(f"target {target['name']}", control.delete_gateway_target,
                        gatewayIdentifier=gateway_id,
                        targetId=target["targetId"])
        except Exception as exc:
            print(f"  could not list targets ({type(exc).__name__})")
        attempt(gateway_id, control.delete_gateway, gatewayIdentifier=gateway_id)

    print("\n── lambda")
    attempt(config.LAMBDA_NAME, lam.delete_function,
            FunctionName=config.LAMBDA_NAME)

    print("\n── roles")
    for role in (config.LAMBDA_ROLE, config.GATEWAY_ROLE, config.AGENT_ROLE):
        try:
            # A role cannot be deleted while a policy is attached, and managed
            # and inline policies detach differently.
            for policy in iam.list_attached_role_policies(
                    RoleName=role)["AttachedPolicies"]:
                iam.detach_role_policy(RoleName=role, PolicyArn=policy["PolicyArn"])
            for name in iam.list_role_policies(RoleName=role)["PolicyNames"]:
                iam.delete_role_policy(RoleName=role, PolicyName=name)
        except Exception:
            pass
        attempt(role, iam.delete_role, RoleName=role)

    # Cognito, if step 2 created one. Not deleted automatically because a user
    # pool may be shared with something else — say what to remove rather than
    # removing it.
    if state.get("cognito"):
        print(f"\n  Cognito user pool was created by step 2 and is NOT deleted here:")
        print(f"    {json.dumps(state['cognito'])[:160]}")

    Path(config.STATE_FILE).unlink()
    print("\nteardown complete")


if __name__ == "__main__":
    main()
