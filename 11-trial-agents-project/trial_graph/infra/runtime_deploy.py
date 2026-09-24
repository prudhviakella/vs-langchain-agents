"""Build trial_graph's container, push to ECR, and create/update its
AgentCore Runtime.

    infra/runtime_deploy.py
        |
        |-- STEP 1: ensure the ECR repository exists
        |-- STEP 2: docker buildx build --platform linux/arm64 --push
        |   (arm64 is a hard AgentCore Runtime requirement — verified
        |   directly against AWS's own docs, not assumed; the platform
        |   flag is not optional polish)
        `-- STEP 3: create_agent_runtime, or update it if it already
            exists — idempotent, same pattern as every other infra/*.py
            in this project

WHAT THIS DOES NOT DO

    - It does not build the image locally with plain `docker build`.
      AgentCore Runtime rejects an x86_64 image outright, and most
      developer machines are x86_64 by default — buildx with an
      explicit --platform is what actually produces an arm64 image
      regardless of the host architecture.
    - It does not handle the Docker login step. `docker buildx build
      --push` needs `aws ecr get-login-password | docker login` to have
      already run in this shell — a one-time setup step, not something
      to silently shell out to on every deploy.
"""
import subprocess
import time

import boto3

ecr = boto3.client("ecr")
runtime_client = boto3.client("bedrock-agentcore-control")

REPO_NAME = "trial-graph-agent"
RUNTIME_NAME = "trial_graph"


def ensure_ecr_repo() -> str:
    try:
        repo = ecr.describe_repositories(repositoryNames=[REPO_NAME])["repositories"][0]
        return repo["repositoryUri"], repo["repositoryArn"]
    except ecr.exceptions.RepositoryNotFoundException:
        pass
    print(f"  creating ECR repository {REPO_NAME!r}")
    repo = ecr.create_repository(repositoryName=REPO_NAME)["repository"]
    return repo["repositoryUri"], repo["repositoryArn"]


def build_and_push(repo_uri: str, dockerfile_dir: str, tag: str = "latest") -> str:
    image_uri = f"{repo_uri}:{tag}"
    print(f"  building and pushing {image_uri} (linux/arm64)")
    subprocess.run([
        "docker", "buildx", "build", "--platform", "linux/arm64",
        "-t", image_uri, "--push", dockerfile_dir,
    ], check=True)
    return image_uri


def deploy_runtime(image_uri: str, role_arn: str, environment_variables: dict) -> str:
    """Create the Runtime if it doesn't exist yet, otherwise update its
    container image and environment. Returns the runtime ARN.

    environment_variables is required, not optional: confirmed directly
    against the real boto3 service model (CreateAgentRuntime's own
    input shape) that environmentVariables is how a Runtime's container
    actually receives anything — without this, the deployed container's
    own config.py would raise KeyError on every os.environ[...] read at
    startup, since nothing else populates the container's environment.
    """
    existing = runtime_client.list_agent_runtimes(maxResults=100).get("agentRuntimes", [])
    match = next((r for r in existing if r["agentRuntimeName"] == RUNTIME_NAME), None)

    if match:
        print(f"  updating existing runtime {RUNTIME_NAME!r}")
        runtime_client.update_agent_runtime(
            agentRuntimeId=match["agentRuntimeId"],
            agentRuntimeArtifact={"containerConfiguration": {"containerUri": image_uri}},
            roleArn=role_arn,
            environmentVariables=environment_variables,
        )
        return match["agentRuntimeArn"]

    print(f"  creating runtime {RUNTIME_NAME!r}")
    response = runtime_client.create_agent_runtime(
        agentRuntimeName=RUNTIME_NAME,
        agentRuntimeArtifact={"containerConfiguration": {"containerUri": image_uri}},
        networkConfiguration={"networkMode": "PUBLIC"},
        roleArn=role_arn,
        environmentVariables=environment_variables,
    )
    # Runtime creation is asynchronous — poll rather than assume READY
    # the instant this call returns, same reasoning as the Gateway's own
    # create_gateway() polling in infra/gateway.py.
    deadline = time.time() + 300
    while True:
        status = runtime_client.get_agent_runtime(
            agentRuntimeId=response["agentRuntimeId"])["status"]
        if status == "READY":
            break
        if status in ("CREATE_FAILED", "UPDATE_FAILED"):
            raise RuntimeError(f"runtime {RUNTIME_NAME!r} entered {status}")
        if time.time() > deadline:
            raise RuntimeError(f"runtime {RUNTIME_NAME!r} not READY after 300s "
                              f"(status={status})")
        time.sleep(10)

    return response["agentRuntimeArn"]
