"""Deploy the supervisor to AgentCore — after NLQ and chart_gen.

    STEP 0  preflight      AWS identity, Docker with linux/arm64; CloudWatch Transaction Search on
    STEP 1  specialists    ../nlq/deployment.json and ../chart_gen/deployment.json -> their runtime ARNs
    STEP 2  OpenAI key     ecom-agents/openai (shared)
    STEP 3  runtime        image (linux/arm64) -> ECR -> role (invoke those two runtimes, read that secret)
                           -> AgentCore runtime ecom_supervisor  (NLQ_ARN, CHART_ARN in its environment)
    STEP 4  deployment.json  — the backend reads `runtime_arn` from it

    cd agent/supervisor
    python deploy.py                                   build the image locally (needs Docker)
    python deploy.py --image <you>/ecom-supervisor-agent:1.0        copy a prebuilt image from Docker Hub (NO Docker)
    python deploy.py --publish <you>/ecom-supervisor-agent:1.0      instructor: build + push to Docker Hub, then stop

WHAT THIS DOES NOT DO
    It does not deploy NLQ or chart_gen — it stops if either has not been deployed.
"""
import argparse
import json
import os
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent
ENV_FILE = HERE.parents[1] / "ingestion" / ".env"
OPENAI_SECRET = "ecom-agents/openai"


def _env_file() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _specialist(name: str) -> str:
    path = AGENTS / name / "deployment.json"
    if not path.exists():
        raise SystemExit(f"{name} is not deployed ({path} missing) — run `python deploy.py` in agent/{name} first")
    arn = json.loads(path.read_text(encoding="utf-8"))["runtime_arn"]
    print(f"  {name:10} {arn}")
    return arn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", help="Docker Hub image to deploy, e.g. prudhvi/ecom-supervisor-agent:1.0 (no Docker needed)")
    ap.add_argument("--publish", help="instructor: build linux/arm64, push to this Docker Hub image, and stop")
    args = ap.parse_args()
    _env_file()
    session = boto3.session.Session()
    region = session.region_name or os.environ.get("AWS_REGION", "us-east-1")
    from infra import observability, preflight, runtime_deploy, runtime_iam

    print("=== STEP 0: preflight ===")
    preflight.run(needs_gateway=False, session=session, needs_docker=not args.image)
    observability.ensure_transaction_search()
    if args.publish:
        from infra import image_copy
        image_copy.publish(args.publish, str(HERE / "agent_code"))
        print(f"\npublished {args.publish} — students deploy it with: python deploy.py --image {args.publish}")
        return

    print("\n=== STEP 1: specialists ===")
    nlq_arn, chart_arn = _specialist("nlq"), _specialist("chart_gen")

    print("\n=== STEP 2: OpenAI key ===")
    sm = session.client("secretsmanager")
    try:
        secret_arn = sm.describe_secret(SecretId=OPENAI_SECRET)["ARN"]
        print(f"  {OPENAI_SECRET} exists")
    except sm.exceptions.ResourceNotFoundException:
        raise SystemExit(f"{OPENAI_SECRET} missing — deploy NLQ first (it creates the shared secrets)")

    print("\n=== STEP 3: runtime ===")
    repo_uri, repo_arn = runtime_deploy.ensure_ecr_repo()
    if args.image:
        from infra import image_copy
        image = image_copy.copy(args.image, repo_uri, region)
    else:
        runtime_deploy.ecr_login()
        image = runtime_deploy.build_and_push(repo_uri, str(HERE / "agent_code"))
    role = runtime_iam.runtime_role(ecr_repo_arn=repo_arn, secret_arns=[secret_arn], specialist_arns=[nlq_arn, chart_arn])
    runtime_arn = runtime_deploy.deploy_runtime(image, role, {
        "AWS_REGION": region, "SUPERVISOR_MODEL": os.environ.get("SUPERVISOR_MODEL", "gpt-6-sol"),
        "OPENAI_SECRET_ID": secret_arn, "NLQ_ARN": nlq_arn, "CHART_ARN": chart_arn})

    print("\n=== STEP 4: deployment.json ===")
    (HERE / "deployment.json").write_text(json.dumps({"agent": "supervisor", "runtime_arn": runtime_arn,
                                                      "region": region}, indent=2) + "\n", encoding="utf-8")
    print(f"\ndone. supervisor runtime: {runtime_arn}")


if __name__ == "__main__":
    main()
