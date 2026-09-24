"""Build the Supervisor's container, push to ECR, and create/update its
AgentCore Runtime. Identical structure to trial_graph/infra/runtime_deploy.py
— see that file's own docstring for the full reasoning.
"""
import subprocess
import time

import boto3

ecr = boto3.client("ecr")
runtime_client = boto3.client("bedrock-agentcore-control")

REPO_NAME = "supervisor-agent"
RUNTIME_NAME = "supervisor"


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
    container image and environment. Returns the runtime ARN. See
    trial_graph/infra/runtime_deploy.py's own docstring for why
    environment_variables is required, not optional.
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
