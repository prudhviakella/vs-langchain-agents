#!/usr/bin/env python3
"""Provision the entire pipeline in one command.

    python deploy.py

Safe to re-run. Every step checks whether its resource already exists first
— deploy fails part-way often enough in practice (a permissions gap, an
expired session, a service quota) that re-running has to pick up where it
stopped, not error on everything that already succeeded.

ORDER, AND WHY IT IS THIS ORDER

    0. Quota check          confirms the account's EC2 Spot vCPU quota can
                           actually run the largest configured tier BEFORE
                           anything is created — see infra/preflight.py for
                           why a quota that is too low produces no error at
                           all otherwise, just a job stuck in RUNNABLE
    1. IAM roles           needed by nearly everything below
    2. S3 bucket           empty; raw/, docs/, cache/, registry/, error/
                           come into existence on first write
    3. DynamoDB table      + its stream, which the error handler subscribes
                           to later
    4. Secrets Manager     placeholder API keys — real values are a separate,
                           printed command, deliberately not part of this
                           script (see the docstring on infra.secrets)
    5. Parameter Store     tuning defaults, written only where not already
                           set — a re-deploy never resets a hand-tuned value
    6. ECR repositories    two of them; see infra.ecr for why not one
    7. Docker build + push both images — the one step that is NOT pure boto3,
                           because it needs a local Docker daemon. Everything
                           before and after this step is idempotent AWS API
                           calls; this step shells out.
    8. AWS Batch           compute environments, job queues, job definitions,
                           for both stages — needs the image URIs from step 7
                           and the roles from step 1
    9. Lambda functions    all three, packaged and deployed
   10. Triggers            wired LAST, deliberately: this is what makes the
                           pipeline self-starting, and everything it depends
                           on — the classifier's role, the Batch queues, the
                           Parameter Store values it reads at invocation —
                           must already exist before a PDF can land and fire
                           it. Wiring it first would mean a real upload could
                           arrive before the pipeline is actually ready to
                           process it.

WHAT THIS DOES NOT DO

Rotate or set real secret values — printed as a command, not run
automatically, because typing real API keys into a script's own output
defeats a meaningful part of using Secrets Manager at all.

Create a dedicated VPC — see infra/network.py for why the default VPC is
used instead.
"""

import argparse
import subprocess

import boto3

from infra import batch, config, ecr, iam, lambdas, preflight, secrets, storage

sts = boto3.client("sts", region_name=config.REGION)


def step(message: str) -> None:
    print(f"\n{'=' * 70}\n{message}\n{'=' * 70}")


def docker_build_and_push(repo_uri: str, dockerfile_dir: str) -> str:
    """Build for linux/amd64 explicitly and push.

    The explicit platform matters on Apple Silicon: building without it on
    an M-series Mac produces an arm64 image that Batch's EC2 instances
    (x86_64) fail to start with an exec format error — a failure that
    surfaces at job-run time, in CloudWatch, looking nothing like an image
    problem.
    """
    login = subprocess.run(
        ["aws", "ecr", "get-login-password", "--region", config.REGION],
        capture_output=True, text=True, check=True)
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin",
        repo_uri.split("/")[0]],
        input=login.stdout, text=True, check=True)

    subprocess.run(["docker", "build", "--platform", "linux/amd64",
                    "-t", f"{repo_uri}:latest", dockerfile_dir], check=True)
    subprocess.run(["docker", "push", f"{repo_uri}:latest"], check=True)
    return f"{repo_uri}:latest"


def main() -> None:
    ap = argparse.ArgumentParser(description="Provision the entire pipeline")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the EC2 Spot vCPU quota check")
    args = ap.parse_args()

    account_id = sts.get_caller_identity()["Account"]
    bucket = config.bucket_name(account_id)

    print(f"Project: {config.PROJECT}")
    print(f"Region:  {config.REGION}")
    print(f"Account: {account_id}")
    print(f"Bucket:  {bucket}")

    step("0/10  EC2 Spot vCPU quota")
    if args.skip_preflight:
        print("   skipped (--skip-preflight)")
    else:
        # Checked BEFORE anything is created, deliberately. AWS Batch draws
        # from one account-and-region-wide Spot vCPU quota that defaults to
        # 5 on many accounts — see infra/preflight.py for the confirmed
        # quota code and why a quota that is too low produces no error at
        # all, just a job stuck in RUNNABLE forever.
        preflight.check(config.STAGE1_TIERS, config.STAGE2_CPU)

    step("1/10  IAM roles")
    roles = iam.create_roles(account_id)

    step("2/10  S3 bucket")
    storage.create_bucket(bucket)

    step("3/10  DynamoDB audit table")
    stream_arn = storage.create_audit_table()

    step("4/10  Secrets Manager")
    secrets.create_secret()

    step("5/10  Parameter Store")
    secrets.create_parameters()

    step("6/10  ECR repositories")
    repo_uris = ecr.create_repos()

    step("7/10  Build and push container images")
    print("This needs a local Docker daemon and takes several minutes — the "
          "worker image includes Docling's model weights.")
    worker_image = docker_build_and_push(repo_uris["worker"], "worker")
    graph_image = docker_build_and_push(repo_uris["graph_worker"], "graph_worker")

    step("8/10  AWS Batch: compute environments, queues, job definitions")
    batch_result = batch.deploy_all(account_id, worker_image, graph_image, roles)

    step("9/10  Lambda functions")
    lambdas.deploy_all(account_id, bucket, roles, stream_arn)

    step("10/10  Wiring the upload trigger")
    # Already done inside lambdas.deploy_all, called out as its own numbered
    # step because it is the one that matters most: this is the exact
    # moment the pipeline becomes self-starting. Nothing further is called
    # here — this line exists so the console output names the moment
    # explicitly rather than letting it pass silently inside step 9.
    print("   raw/*.pdf uploads now trigger the classifier automatically")

    print("\n" + "=" * 70)
    print("DEPLOY COMPLETE")
    print("=" * 70)
    print(f"""
Next steps:

1. Set the real API keys (placeholders were written in step 4):

   aws secretsmanager put-secret-value --secret-id {config.SECRET_NAME} \\
     --secret-string '{{"OPENAI_API_KEY":"sk-...","PINECONE_API_KEY":"pc-...","NEO4J_PASSWORD":"..."}}'

2. Set your Neo4j URI (defaults to a "replace-me" placeholder — the graph
   worker rejects it with a clear error rather than trying to connect):

   aws ssm put-parameter --name {config.param_name('NEO4J_URI')} \\
     --value "neo4j+s://xxxxxxxx.databases.neo4j.io" --type String --overwrite

3. Upload a PDF to start the pipeline:

   python scripts/upload_pdf.py path/to/document.pdf

4. Check status:

   python scripts/status.py <doc_id>

Bucket:        s3://{bucket}/
Audit table:   {config.AUDIT_TABLE}
Batch queues:  {', '.join(batch_result['queues'].values())}
""")


if __name__ == "__main__":
    main()
