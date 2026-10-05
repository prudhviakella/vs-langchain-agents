"""Package and deploy the NLQ SQL Lambda — in the VPC, beside RDS.

    lambda_tools/handler.py + sql_exec.py  ┐
    agent_code/nlq/guard.py (the ONE guard) ├─► zip root ─► Lambda ecom-nlq-sql (python3.12, x86_64)
    psycopg[binary] for manylinux x86_64   ┘                 VpcConfig: default VPC subnets + Lambda SG

WHY pip RUNS WITH --platform
    psycopg ships compiled code. `pip install --target` on a Mac bundles macOS
    binaries and the Lambda fails at import; manylinux2014_x86_64 / cp312 /
    --only-binary fetches the Linux build Lambda needs.

WHY guard.py COMES FROM THE AGENT
    One guard, two places it runs. A copy kept in lambda_tools would drift.

WHAT THIS DOES NOT DO
    It does not open the database to the Lambda — network.py does (RDS security
    group rule, Secrets Manager endpoint).
"""
import io
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3

FUNCTION_NAME = "ecom-nlq-sql"
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


# What Lambda says while a just-created role or policy has not propagated yet.
_PROPAGATING = ("does not have permissions to call", "cannot be assumed by Lambda")


def _role_ready(call, attempts: int = 12, wait_s: int = 10):
    """Run a Lambda call that depends on the execution role; retry ONLY while IAM is
    still propagating it (a new role, or a policy just put on it — e.g. the
    ec2:CreateNetworkInterface permission a VPC function needs). Lambda checks the
    role when the function is created, often before IAM has finished. Any other
    error raises at once."""
    for attempt in range(attempts):
        try:
            return call()
        except lambda_client.exceptions.InvalidParameterValueException as exc:
            if not any(m in str(exc) for m in _PROPAGATING) or attempt == attempts - 1:
                raise
            print(f"  waiting for the new IAM role to propagate ({(attempt + 1) * wait_s}s)…")
            time.sleep(wait_s)


def deploy(role_arn: str, env: dict, subnet_ids: list[str], security_group_id: str) -> str:
    zip_bytes = build_zip()
    config = dict(Role=role_arn, Environment={"Variables": env}, Timeout=TIMEOUT_S, MemorySize=MEMORY_MB,
                  VpcConfig={"SubnetIds": subnet_ids, "SecurityGroupIds": [security_group_id]})
    try:
        lambda_client.get_function(FunctionName=FUNCTION_NAME)
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"  creating function {FUNCTION_NAME!r} (in the VPC — about a minute)")
        arn = _role_ready(lambda: lambda_client.create_function(
            FunctionName=FUNCTION_NAME, Runtime="python3.12", Architectures=["x86_64"],
            Handler="handler.lambda_handler", Code={"ZipFile": zip_bytes}, **config))["FunctionArn"]
        lambda_client.get_waiter("function_active_v2").wait(FunctionName=FUNCTION_NAME)
        return arn
    print(f"  updating function {FUNCTION_NAME!r}")
    lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes)
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    _role_ready(lambda: lambda_client.update_function_configuration(FunctionName=FUNCTION_NAME, **config))
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    return lambda_client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]["FunctionArn"]


def allow_gateway_invoke(gateway_arn: str) -> None:
    """Only THIS gateway (SourceArn) may invoke the function."""
    try:
        lambda_client.add_permission(FunctionName=FUNCTION_NAME, StatementId="allow-ecom-nlq-gateway",
                                     Action="lambda:InvokeFunction", Principal="bedrock-agentcore.amazonaws.com",
                                     SourceArn=gateway_arn)
    except lambda_client.exceptions.ResourceConflictException:
        pass