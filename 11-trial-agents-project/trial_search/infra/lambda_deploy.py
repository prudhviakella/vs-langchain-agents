"""Package and deploy trial_search's tools Lambda.

    _source_files()  every .py module in lambda_tools/ — handler.py, rerank.py
    _build_zip()     those modules + openai + pinecone + neo4j, built for
                     the LAMBDA's platform, not the deployer's
    deploy()         create, or update code then configuration
    allow_gateway_invoke()   resource policy for this function's Gateway

WHY EVERY MODULE, NOT JUST handler.py

The zip once listed handler.py by name. A second module (rerank.py) would
have been left out, and the Lambda would fail on "import rerank" at its
first invocation in AWS while every local test passed — the tests import
from the source folder, not from the zip. Packaging every module in
lambda_tools/ removes that trap for this module and the next.

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
    once a second one does. Cohere needs no package: rerank.py calls its
    API with the standard library.
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
# openai pinned to the range langchain-openai supports and the tests ran
# against (3.x). Unbounded, a future major release would reach the Lambda
# untested — the build already jumped from 1.x to 3.x once.
PACKAGES = ["openai>=2.45,<4", "pinecone>=6.0", "neo4j>=5.20"]
TIMEOUT_S = 30          # rerank adds one HTTPS call (5 s cap) to a search
MEMORY_MB = 512


def _source_files() -> list[Path]:
    """Every module the Lambda imports, placed at the zip's root."""
    return sorted(p for p in LAMBDA_DIR.glob("*.py") if not p.name.startswith("test_"))


def _build_zip() -> bytes:
    # STEP 1  third-party packages, for Linux x86_64 / Python 3.12
    build = LAMBDA_DIR / "_build"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(build),
                    "--platform", "manylinux2014_x86_64", "--implementation", "cp",
                    "--python-version", "3.12", "--only-binary=:all:", "--quiet",
                    "--disable-pip-version-check", "--no-warn-conflicts",
                    *PACKAGES], check=True)

    # STEP 2  our modules at the root, then the packages
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for module in _source_files():
            zf.write(module, module.name)
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
            Environment=environment, Timeout=TIMEOUT_S, MemorySize=MEMORY_MB)
        lambda_client.get_waiter("function_active_v2").wait(FunctionName=FUNCTION_NAME)
        return response["FunctionArn"]

    # Code and configuration updates cannot overlap: the second fails with
    # ResourceConflictException if issued before the first finishes.
    print(f"  updating function {FUNCTION_NAME!r}")
    lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes)
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    lambda_client.update_function_configuration(
        FunctionName=FUNCTION_NAME, Role=role_arn, Environment=environment,
        Timeout=TIMEOUT_S, MemorySize=MEMORY_MB)
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
