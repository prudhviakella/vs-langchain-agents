"""Deploy chart_gen to AgentCore — runtime only: no database, Gateway or Lambda.

    STEP 0  preflight      AWS identity, Docker with linux/arm64; CloudWatch Transaction Search on
    STEP 1  OpenAI key     ecom-agents/openai (shared with NLQ; created from ingestion/.env if missing)
    STEP 2  runtime        image (linux/arm64) -> ECR -> role (that secret only) -> AgentCore runtime ecom_chart_gen
    STEP 3  deployment.json

    cd agent/chart_gen
    python deploy.py                                   build the image locally (needs Docker)
    python deploy.py --image <you>/ecom-chart-gen-agent:1.0        copy a prebuilt image from Docker Hub (NO Docker)
    python deploy.py --publish <you>/ecom-chart-gen-agent:1.0      instructor: build + push to Docker Hub, then stop

SAFE TO RE-RUN: the image is rebuilt; everything else is looked up first.

WHAT THIS DOES NOT DO
    It does not deploy NLQ or the supervisor, and gives chart_gen no data access:
    it charts only the table the supervisor sends.
"""
import argparse
import json
import os
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE.parents[1] / "ingestion" / ".env"
OPENAI_SECRET = "ecom-agents/openai"


def _env_file() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _openai_secret(sm) -> str:
    try:
        arn = sm.describe_secret(SecretId=OPENAI_SECRET)["ARN"]
        print(f"  {OPENAI_SECRET} exists")
        return arn
    except sm.exceptions.ResourceNotFoundException:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set — add it to ingestion/.env")
        print(f"  {OPENAI_SECRET} created from OPENAI_API_KEY")
        return sm.create_secret(Name=OPENAI_SECRET, SecretString=json.dumps({"api_key": key}))["ARN"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", help="Docker Hub image to deploy, e.g. prudhvi/ecom-chart-gen-agent:1.0 (no Docker needed)")
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

    print("\n=== STEP 1: OpenAI key ===")
    secret_arn = _openai_secret(session.client("secretsmanager"))

    print("\n=== STEP 2: runtime ===")
    repo_uri, repo_arn = runtime_deploy.ensure_ecr_repo()
    if args.image:
        from infra import image_copy
        image = image_copy.copy(args.image, repo_uri, region)
    else:
        runtime_deploy.ecr_login()
        image = runtime_deploy.build_and_push(repo_uri, str(HERE / "agent_code"))
    role = runtime_iam.runtime_role(ecr_repo_arn=repo_arn, secret_arns=[secret_arn])
    runtime_arn = runtime_deploy.deploy_runtime(image, role, {
        "AWS_REGION": region, "CHART_MODEL": os.environ.get("CHART_MODEL", "gpt-6-sol"),
        "OPENAI_SECRET_ID": secret_arn})

    print("\n=== STEP 3: deployment.json ===")
    (HERE / "deployment.json").write_text(json.dumps({"agent": "chart_gen", "runtime_arn": runtime_arn,
                                                      "region": region}, indent=2) + "\n", encoding="utf-8")
    print(f"\ndone. chart_gen runtime: {runtime_arn}")


if __name__ == "__main__":
    main()
