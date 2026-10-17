"""Package and deploy the NLQ SQL Lambda — in the VPC, beside RDS.

    lambda_tools/handler.py + sql_exec.py  ┐
    agent_code/nlq/guard.py (the ONE guard) ├─► zip root ─► Lambda ecom-nlq-sql (python3.12, x86_64)
    psycopg[binary] for manylinux x86_64   ┘                 VpcConfig: default VPC subnets + Lambda SG

    deploy()                 STEP 1 build the zip
                             STEP 2 create the function, or update its code then its configuration
                                    (both wait for IAM — a just-created role is refused for a while)
    allow_gateway_invoke()   the function's resource policy: THIS gateway may invoke it.
                             deploy.py calls it BEFORE creating the gateway target.

WHY pip RUNS WITH --platform
    psycopg ships compiled code. `pip install --target` on a Mac bundles macOS
    binaries and the Lambda fails at import; manylinux2014_x86_64 / cp312 /
    --only-binary fetches the Linux build Lambda needs.

WHY guard.py COMES FROM THE AGENT
    One guard, two places it runs. A copy kept in lambda_tools would drift.

WHY allow_gateway_invoke RUNS BEFORE THE TARGET
    A resource policy takes effect at once; the Gateway role's policy only
    after IAM propagates. The gateway target is checked when it is created,
    so the resource policy must already be there (see infra/gateway.py).

WHAT THIS DOES NOT DO
    It does not open the database to the Lambda — network.py does (RDS security
    group rule, Secrets Manager endpoint).
"""
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import boto3

from infra import iam_ready

FUNCTION_NAME = "ecom-nlq-sql"
STATEMENT_ID = "allow-ecom-nlq-gateway"
ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "lambda_tools" / "handler.py", ROOT / "lambda_tools" / "sql_exec.py",
           ROOT / "agent_code" / "nlq" / "guard.py"]
PACKAGES = ["psycopg[binary]>=3.1"]
TIMEOUT_S = 30
MEMORY_MB = 512

lambda_client = boto3.client("lambda")


def build_zip() -> bytes:
    build = ROOT / "lambda_tools" / "_build"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(build),
                    "--platform", "manylinux2014_x86_64", "--implementation", "cp",
                    "--python-version", "3.12", "--only-binary=:all:", "--quiet",
                    "--disable-pip-version-check", "--no-warn-conflicts", *PACKAGES], check=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for module in SOURCES:
            zf.write(module, module.name)
        for path in build.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(build))
    shutil.rmtree(build, ignore_errors=True)
    return buf.getvalue()


def deploy(role_arn: str, env: dict, subnet_ids: list[str], security_group_id: str) -> str:
    # STEP 1 — the package
    zip_bytes = build_zip()
    config = dict(Role=role_arn, Environment={"Variables": env}, Timeout=TIMEOUT_S, MemorySize=MEMORY_MB,
                  VpcConfig={"SubnetIds": subnet_ids, "SecurityGroupIds": [security_group_id]})
    what = f"Lambda to use {role_arn.rsplit('/', 1)[-1]}"

    # STEP 2a — first deploy: create
    try:
        lambda_client.get_function(FunctionName=FUNCTION_NAME)
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"  creating function {FUNCTION_NAME!r} (in the VPC — about a minute)")
        arn = iam_ready.until_ready(lambda: lambda_client.create_function(
            FunctionName=FUNCTION_NAME, Runtime="python3.12", Architectures=["x86_64"],
            Handler="handler.lambda_handler", Code={"ZipFile": zip_bytes}, **config)["FunctionArn"],
            iam_ready.LAMBDA, what)
        lambda_client.get_waiter("function_active_v2").wait(FunctionName=FUNCTION_NAME)
        return arn

    # STEP 2b — re-deploy: code first, then configuration (they cannot overlap)
    print(f"  updating function {FUNCTION_NAME!r}")
    lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes)
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    iam_ready.until_ready(lambda: lambda_client.update_function_configuration(
        FunctionName=FUNCTION_NAME, **config), iam_ready.LAMBDA, what)
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    return lambda_client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]["FunctionArn"]


def allow_gateway_invoke(gateway_arn: str) -> None:
    """Resource policy: the AgentCore service may invoke the function, only for THIS gateway (SourceArn).

    A statement left from an older gateway (same id, another SourceArn) is replaced:
    a deploy that recreated the gateway would otherwise keep the old, now useless,
    grant and every target check would fail."""
    try:
        policy = json.loads(lambda_client.get_policy(FunctionName=FUNCTION_NAME)["Policy"])
    except lambda_client.exceptions.ResourceNotFoundException:
        policy = {"Statement": []}
    current = next((s for s in policy.get("Statement", []) if s.get("Sid") == STATEMENT_ID), None)
    if current and gateway_arn in json.dumps(current.get("Condition", {})):
        print(f"  {FUNCTION_NAME} already allows the gateway to invoke it")
        return
    if current:
        lambda_client.remove_permission(FunctionName=FUNCTION_NAME, StatementId=STATEMENT_ID)
    lambda_client.add_permission(FunctionName=FUNCTION_NAME, StatementId=STATEMENT_ID,
                                 Action="lambda:InvokeFunction", Principal="bedrock-agentcore.amazonaws.com",
                                 SourceArn=gateway_arn)
    print(f"  {FUNCTION_NAME} now allows the gateway to invoke it")
