"""Start one ingestion run and, optionally, wait for it.

    python aws/run.py                    start and poll until finished
    python aws/run.py --no-wait          start and exit
    python aws/run.py --only large       run a single tier

One execution processes the whole corpus: the state machine fans out into three
lanes internally, so there is nothing to loop over here.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import boto3

import config

sfn = boto3.client("stepfunctions", region_name=config.REGION)
s3 = boto3.client("s3", region_name=config.REGION)
ACCOUNT = boto3.client("sts", region_name=config.REGION).get_caller_identity()["Account"]


def load_manifest() -> dict:
    """The tier-sorted document list written by upload_pdfs.py.

    Read from S3 rather than from a local file so a run can be started from any
    machine, and so what executes is exactly what was uploaded — a stale local copy
    would silently process the wrong set.
    """
    body = s3.get_object(Bucket=config.BUCKET, Key=config.MANIFEST_KEY)["Body"].read()
    return json.loads(body)


def main() -> None:
    """Start one state-machine execution over the uploaded corpus."""
    ap = argparse.ArgumentParser(description="Start an ingestion run")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--only", choices=list(config.TIERS),
                    help="run one tier and leave the others empty")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="start even if the Fargate vCPU quota looks too low")
    args = ap.parse_args()

    # Checked here rather than left to fail inside the state machine. An exceeded
    # vCPU quota surfaces as a generic task failure in a branch, forty minutes in,
    # with nothing in the message about quotas.
    if not args.skip_preflight:
        import preflight
        needed, _ = preflight.required_vcpu()
        quota = preflight.current_quota()
        if quota is not None and quota < needed:
            raise SystemExit(
                f"Fargate vCPU quota is {quota:.0f} but this configuration needs "
                f"{needed}. Run aws/preflight.py for the fix, or pass "
                "--skip-preflight to start anyway."
            )

    manifest = load_manifest()

    # Running a single tier is how you test the pipeline cheaply: the small lane is
    # two short documents and finishes in minutes. The other lanes are emptied
    # rather than removed, because the state machine expects all three keys.
    if args.only:
        manifest = {tier: (items if tier == args.only else [])
                    for tier, items in manifest.items()}

    for tier, items in manifest.items():
        print(f"{tier:<8} {len(items):>3} documents  "
              f"{config.TIERS[tier]['concurrency']} at a time")

    # Execution names must be unique within the account. A UTC timestamp is both
    # unique and sortable, which makes the console history readable.
    arn = f"arn:aws:states:{config.REGION}:{ACCOUNT}:stateMachine:{config.STATE_MACHINE}"
    name = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    execution = sfn.start_execution(stateMachineArn=arn, name=name,
                                    input=json.dumps(manifest))["executionArn"]
    print(f"\nstarted {name}")
    # The console graph is the best observability this pipeline has: three lanes
    # side by side, one box per document, turning green as they finish.
    print(f"console: https://{config.REGION}.console.aws.amazon.com/states/home"
          f"?region={config.REGION}#/v2/executions/details/{execution}")

    if args.no_wait:
        return

    # Poll rather than wait on an event: Step Functions has no blocking wait, and a
    # printed elapsed time is what tells you the run is alive during the twenty
    # minutes when nothing else is happening.
    started = time.time()
    while True:
        status = sfn.describe_execution(executionArn=execution)["status"]
        elapsed = int(time.time() - started)
        print(f"\r{status}  {elapsed // 60}m{elapsed % 60:02d}s", end="", flush=True)
        if status != "RUNNING":
            print()
            break
        time.sleep(15)

    # A SUCCEEDED execution does not mean every document succeeded: the Map states
    # catch per-document failures so one bad PDF cannot stop the other nineteen.
    # The audit table is where per-document outcomes live.
    print("\nrun `python aws/status.py` for per-document results")


if __name__ == "__main__":
    main()
