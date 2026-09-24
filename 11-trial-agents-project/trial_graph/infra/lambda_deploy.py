"""Package and deploy trial_graph's tools Lambda.

    deploy.py
        |
        v
    infra/lambda_deploy.py
        |
        |-- zip handler.py + the neo4j driver package together
        |   (Lambda's own runtime has no neo4j package pre-installed)
        |
        `-- create, or update, the function — idempotent either way

WHAT THIS DOES NOT DO

    - It does not use a Lambda Layer for the neo4j dependency. A layer is
      the better long-term answer once more than one Lambda in this
      project needs the same package, but for one function it is an
      extra moving part with nothing yet to share it with — the
      dependency is bundled directly into the deployment zip instead.
    - It does not set the Neo4j credentials as plain environment
      variables. They are read from Secrets Manager at cold-start
      (see lambda_tools/handler.py's own driver setup) — environment
      variables on a Lambda are visible to anyone with
      lambda:GetFunctionConfiguration, which is a much wider audience
      than "can read this one secret."
"""
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import boto3

lambda_client = boto3.client("lambda")

FUNCTION_NAME = "trial-graph-tools"
LAMBDA_DIR = Path(__file__).parent.parent / "lambda_tools"


def _build_zip() -> bytes:
    """Bundle handler.py plus the neo4j package into one deployment zip."""
    build_dir = LAMBDA_DIR / "_build"
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--target", str(build_dir), "neo4j>=5.20",
                    "--quiet", "--break-system-packages"], check=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(LAMBDA_DIR / "handler.py", "handler.py")
        for path in build_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(build_dir))
    return buf.getvalue()


def deploy(role_arn: str, secret_id: str) -> str:
    """Create the function if it doesn't exist, otherwise update its code
    and configuration. Returns the function ARN.
    """
    zip_bytes = _build_zip()
    env = {"Variables": {"NEO4J_SECRET_ID": secret_id}}

    try:
        lambda_client.get_function(FunctionName=FUNCTION_NAME)
        exists = True
    except lambda_client.exceptions.ResourceNotFoundException:
        exists = False

    if exists:
        print(f"  updating existing function {FUNCTION_NAME!r}")
        lambda_client.update_function_code(FunctionName=FUNCTION_NAME,
                                           ZipFile=zip_bytes)
        # Code and configuration updates cannot run concurrently — the
        # waiter below is not optional polish, a configuration update
        # issued too soon fails outright with ResourceConflictException.
        lambda_client.get_waiter("function_updated").wait(FunctionName=FUNCTION_NAME)
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME, Environment=env, Role=role_arn)
        response = lambda_client.get_function(FunctionName=FUNCTION_NAME)
        return response["Configuration"]["FunctionArn"]

    print(f"  creating function {FUNCTION_NAME!r}")
    response = lambda_client.create_function(
        FunctionName=FUNCTION_NAME, Runtime="python3.12", Role=role_arn,
        Handler="handler.lambda_handler", Code={"ZipFile": zip_bytes},
        Environment=env, Timeout=30, MemorySize=256,
    )
    lambda_client.get_waiter("function_active").wait(FunctionName=FUNCTION_NAME)
    return response["FunctionArn"]


def allow_gateway_invoke(function_name: str, gateway_arn: str) -> None:
    """Grant the Gateway's own role permission to invoke this function.
    SourceArn is the GATEWAY's arn — the source resource on whose behalf
    the service invokes this function. An earlier version passed the
    gateway ROLE's arn rewritten into an assumed-role form, which is a
    principal, not a source resource: the condition it produces cannot
    match the invoking gateway, so the call is denied.

    Idempotent: AWS returns a specific, catchable error when the same
    statement is added twice, which is treated as success rather than
    re-raised.
    """
    try:
        lambda_client.add_permission(
            FunctionName=function_name, StatementId="allow-agentcore-gateway",
            Action="lambda:InvokeFunction",
            Principal="bedrock-agentcore.amazonaws.com",
            SourceArn=gateway_arn,
        )
    except lambda_client.exceptions.ResourceConflictException:
        pass  # permission already granted — fine
