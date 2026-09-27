"""Build trial_graph's container, push it to ECR, and create or update its
AgentCore Runtime.

    ensure_ecr_repo()   repository, created once
    build_and_push()    docker buildx --platform linux/arm64 --push
    deploy_runtime()    create_agent_runtime, or update_agent_runtime

THREE SETTINGS THE RUNTIME CANNOT DO WITHOUT

    platform linux/arm64
        AgentCore Runtime runs arm64 only. Most laptops build x86_64 by
        default; buildx with an explicit platform builds arm64 regardless.

    protocolConfiguration = {"serverProtocol": "A2A"}
        The API accepts MCP | HTTP | A2A | AGUI. The container serves the
        A2A contract on port 9000 (serve_a2a's fixed A2A_CONTRACT_PORT).
        An earlier version omitted this field, so the Runtime was never
        told to use the A2A contract and no invocation reached the agent.
        Set on update as well as create, so an existing runtime is fixed
        by redeploying.

    environmentVariables
        The container's config.py reads its settings from the environment
        with os.environ[...]. Nothing else populates it.

WHAT THIS DOES NOT DO

    It does not check Docker or buildx — infra/preflight.py does, before
    anything is created.
"""
import subprocess
import time

import boto3

ecr = boto3.client("ecr")
runtime_client = boto3.client("bedrock-agentcore-control")

REPO_NAME = "trial-graph-agent"
RUNTIME_NAME = "trial_graph"
PROTOCOL = {"serverProtocol": "A2A"}


def ensure_ecr_repo() -> tuple[str, str]:
    try:
        repo = ecr.describe_repositories(repositoryNames=[REPO_NAME])["repositories"][0]
    except ecr.exceptions.RepositoryNotFoundException:
        print(f"  creating ECR repository {REPO_NAME!r}")
        repo = ecr.create_repository(repositoryName=REPO_NAME)["repository"]
    return repo["repositoryUri"], repo["repositoryArn"]


def ecr_login() -> str:
    """docker login to this account's ECR registry. Returns the registry host.

    Runs on every deploy: the token lasts 12 hours, so a login from an earlier
    session fails the push with "no basic auth credentials". The password is
    passed on stdin, never as an argument — arguments are visible to every
    user on the machine through `ps`.
    """
    import base64
    data = ecr.get_authorization_token()["authorizationData"][0]
    user, password = base64.b64decode(data["authorizationToken"]).decode().split(":", 1)
    registry = data["proxyEndpoint"].removeprefix("https://")
    result = subprocess.run(["docker", "login", "--username", user, "--password-stdin", registry],
                            input=password, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"docker login to {registry} failed: {result.stderr.strip()}")
    print(f"  logged in to {registry}")
    return registry


def build_and_push(repo_uri: str, dockerfile_dir: str, tag: str = "latest") -> str:
    ecr_login()
    image_uri = f"{repo_uri}:{tag}"
    print(f"  building and pushing {image_uri} (linux/arm64)")
    subprocess.run(["docker", "buildx", "build", "--platform", "linux/arm64",
                    "-t", image_uri, "--push", dockerfile_dir], check=True)
    return image_uri


def _wait_ready(runtime_id: str) -> None:
    deadline = time.time() + 600
    while (status := runtime_client.get_agent_runtime(
            agentRuntimeId=runtime_id)["status"]) != "READY":
        if status in ("CREATE_FAILED", "UPDATE_FAILED"):
            raise RuntimeError(f"runtime {RUNTIME_NAME!r} entered {status} — see "
                               "/aws/bedrock-agentcore/runtimes/ in CloudWatch Logs")
        if time.time() > deadline:
            raise RuntimeError(f"runtime {RUNTIME_NAME!r} not READY after 600s ({status})")
        time.sleep(10)


def deploy_runtime(image_uri: str, role_arn: str, environment_variables: dict) -> str:
    artifact = {"containerConfiguration": {"containerUri": image_uri}}
    existing = runtime_client.list_agent_runtimes(maxResults=100).get("agentRuntimes", [])
    match = next((r for r in existing if r["agentRuntimeName"] == RUNTIME_NAME), None)

    if match:
        print(f"  updating runtime {RUNTIME_NAME!r}")
        runtime_client.update_agent_runtime(
            agentRuntimeId=match["agentRuntimeId"], agentRuntimeArtifact=artifact,
            roleArn=role_arn, networkConfiguration={"networkMode": "PUBLIC"},
            protocolConfiguration=PROTOCOL, environmentVariables=environment_variables)
        _wait_ready(match["agentRuntimeId"])
        return match["agentRuntimeArn"]

    print(f"  creating runtime {RUNTIME_NAME!r}")
    response = runtime_client.create_agent_runtime(
        agentRuntimeName=RUNTIME_NAME, agentRuntimeArtifact=artifact,
        roleArn=role_arn, networkConfiguration={"networkMode": "PUBLIC"},
        protocolConfiguration=PROTOCOL, environmentVariables=environment_variables)
    _wait_ready(response["agentRuntimeId"])
    return response["agentRuntimeArn"]
