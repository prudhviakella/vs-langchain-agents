"""Step 1 — deploy the tool Lambda.

    python 1_deploy_tools.py

Packages `tools/` and creates one Lambda function backing every tool. The
Gateway turns it into MCP tools in step 2.

One function for several tools, not one each: fewer cold starts, one deployment,
and the Gateway passes the tool name in the invocation context so the handler
can route.
"""

import io
import json
import time
import zipfile
from pathlib import Path

import boto3

import config

iam = boto3.client("iam", region_name=config.REGION)
lam = boto3.client("lambda", region_name=config.REGION)

TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "lambda.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}


def ensure_role() -> str:
    """The Lambda execution role.

    Only CloudWatch Logs. These tools call a public API and nothing in the
    account — so anything more is permission the function does not need, and
    permissions granted "just in case" are the ones nobody removes.
    """
    try:
        role = iam.get_role(RoleName=config.LAMBDA_ROLE)["Role"]
        print(f"  role exists: {config.LAMBDA_ROLE}")
        return role["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    role = iam.create_role(
        RoleName=config.LAMBDA_ROLE,
        AssumeRolePolicyDocument=json.dumps(TRUST),
        Description="Execution role for the trial tool Lambda")["Role"]
    iam.attach_role_policy(
        RoleName=config.LAMBDA_ROLE,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    print(f"  created role: {config.LAMBDA_ROLE}")
    # IAM is eventually consistent. Creating a Lambda with a role that exists
    # but has not propagated fails with an unhelpful "cannot be assumed" error,
    # and the usual reaction is to assume the trust policy is wrong.
    print("  waiting 10s for IAM to propagate")
    time.sleep(10)
    return role["Arn"]


def package() -> bytes:
    """Zip the tools directory in memory.

    No dependencies beyond the standard library, deliberately — `urllib` instead
    of `requests`. A dependency here means a build step, a layer, or a container
    image, none of which teach anything about agents.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in Path("tools").glob("*.py"):
            archive.write(path, path.name)
    return buffer.getvalue()


def main() -> None:
    print(f"deploying tools to {config.REGION}\n")
    role_arn = ensure_role()
    code = package()
    print(f"  package: {len(code):,} bytes")

    try:
        lam.get_function(FunctionName=config.LAMBDA_NAME)
        lam.update_function_code(FunctionName=config.LAMBDA_NAME, ZipFile=code)
        # update_function_code returns before the new code is live; publishing a
        # version and waiting is how you know an invoke will hit what you just
        # uploaded rather than the previous build.
        lam.get_waiter("function_updated").wait(FunctionName=config.LAMBDA_NAME)
        print(f"  updated: {config.LAMBDA_NAME}")
    except lam.exceptions.ResourceNotFoundException:
        lam.create_function(
            FunctionName=config.LAMBDA_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="trial_tools.handler",
            Code={"ZipFile": code},
            # 30s covers up to 25 sequential registry calls in enrollment_stats.
            Timeout=30,
            MemorySize=256,
            Description="Tools exposed to agents through AgentCore Gateway")
        lam.get_waiter("function_active").wait(FunctionName=config.LAMBDA_NAME)
        print(f"  created: {config.LAMBDA_NAME}")

    arn = lam.get_function(FunctionName=config.LAMBDA_NAME)["Configuration"]["FunctionArn"]

    state = json.loads(Path(config.STATE_FILE).read_text()) if Path(config.STATE_FILE).exists() else {}
    state["lambda_arn"] = arn
    Path(config.STATE_FILE).write_text(json.dumps(state, indent=2))

    print(f"\n  arn: {arn}")
    print("  next: python 2_create_gateway.py")


if __name__ == "__main__":
    main()
