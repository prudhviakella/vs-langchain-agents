"""Package and deploy trial_search's tools Lambda.

    _build_zip()   handler.py + openai + pinecone + neo4j, built for the
                   LAMBDA's platform, not the deployer's
    deploy()       create, or update code then configuration

WHY pip RUNS WITH --platform

The deploy runs on a developer laptop, often macOS. openai depends on
pydantic-core and jiter, which ship compiled binaries. A plain
`pip install --target` bundles the laptop's binaries, and the Lambda fails
on import with no useful error until the first invocation. Pinning
manylinux2014_x86_64 / cp312 / --only-binary produces the Linux binaries
the python3.12 x86_64 runtime loads — verified by inspecting the .so files
the pinned install produces.

WHAT THIS DOES NOT DO

    No Lambda Layer. One function uses these packages; a layer is worth it
    once a second one does.
"""
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import boto3

lambda_client = boto3.client("lambda")

FUNCTION_NAME = "trial-search-tools"
LAMBDA_DIR = Path(__file__).parent.parent / "lambda_tools"
PACKAGES = ["openai>=1.0", "pinecone>=6.0", "neo4j>=5.20"]


def _build_zip() -> bytes:
    build = LAMBDA_DIR / "_build"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(build),
                    "--platform", "manylinux2014_x86_64", "--implementation", "cp",
                    "--python-version", "3.12", "--only-binary=:all:", "--quiet",
                    *PACKAGES], check=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(LAMBDA_DIR / "handler.py", "handler.py")
        for path in build.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(build))
    shutil.rmtree(build, ignore_errors=True)
    return buf.getvalue()


def deploy(role_arn: str, env: dict) -> str:
    zip_bytes = _build_zip()
    environment = {"Variables": env}
    try:
        lambda_client.get_function(FunctionName=FUNCTION_NAME)
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"  creating function {FUNCTION_NAME!r}")
        response = lambda_client.create_function(
            FunctionName=FUNCTION_NAME, Runtime="python3.12", Architectures=["x86_64"],
            Role=role_arn, Handler="handler.lambda_handler", Code={"ZipFile": zip_bytes},
            Environment=environment, Timeout=30, MemorySize=512)
        lambda_client.get_waiter("function_active_v2").wait(FunctionName=FUNCTION_NAME)
        return response["FunctionArn"]

    # Code and configuration updates cannot overlap: the second fails with
    # ResourceConflictException if issued before the first finishes.
    print(f"  updating function {FUNCTION_NAME!r}")
    lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes)
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    lambda_client.update_function_configuration(
        FunctionName=FUNCTION_NAME, Role=role_arn, Environment=environment,
        Timeout=30, MemorySize=512)
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    return lambda_client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]["FunctionArn"]


def allow_gateway_invoke(gateway_arn: str) -> None:
    """Resource policy: the Gateway service may invoke this function, but
    only on behalf of THIS gateway (SourceArn)."""
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME, StatementId="allow-trial-search-gateway",
            Action="lambda:InvokeFunction", Principal="bedrock-agentcore.amazonaws.com",
            SourceArn=gateway_arn)
    except lambda_client.exceptions.ResourceConflictException:
        pass   # already granted
