"""Checks that run before a deploy creates anything.

    run(needs_gateway)
        │
        ├─ STEP 1  AWS       credentials and a region (from `aws configure`)
        ├─ STEP 2  Docker    daemon up, buildx present, a builder for linux/arm64
        └─ STEP 3  IAM       every action the deploy calls, simulated for the
                             caller — plus the two Transaction Search actions,
                             but only while Transaction Search is still off
        │
        v
    all fine -> the deploy continues      anything missing -> SystemExit, naming it

WHY CHECK INSTEAD OF DOING

Each check is something the script cannot fix for you. A missing permission
cannot be self-granted; buildx is a local tool; an arm64 builder needs a
privileged container on your machine, which a deploy script should not start
unasked. What the script can do is stop BEFORE creating the first resource,
and say exactly what is missing — rather than fail at step 8 of 11 with a
half-built stack.

THE IAM CHECK IS BEST EFFORT

SimulatePrincipalPolicy evaluates identity policies and permissions
boundaries. It does not see service control policies or resource policies,
so a real call can still be denied after a clean check. If the caller may not
run the simulator at all, the check is skipped with a warning, not a failure.

WHAT THIS DOES NOT DO

    It does not log in to ECR — runtime_deploy.ecr_login does, on every
    deploy, because the token expires after 12 hours.
"""
import subprocess

import boto3

COMMON = [
    "iam:CreateRole", "iam:GetRole", "iam:PutRolePolicy", "iam:DeleteRolePolicy",
    "iam:ListRolePolicies", "iam:UpdateAssumeRolePolicy", "iam:PassRole",
    "ecr:GetAuthorizationToken", "ecr:CreateRepository", "ecr:DescribeRepositories",
    "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
    "ecr:CompleteLayerUpload", "ecr:PutImage",
    "bedrock-agentcore:CreateAgentRuntime", "bedrock-agentcore:UpdateAgentRuntime",
    "bedrock-agentcore:GetAgentRuntime", "bedrock-agentcore:ListAgentRuntimes",
    "bedrock:CreateGuardrail", "bedrock:UpdateGuardrail", "bedrock:GetGuardrail",
    "bedrock:ListGuardrails", "bedrock:CreateGuardrailVersion",
    "bedrock:CreatePrompt", "bedrock:UpdatePrompt", "bedrock:GetPrompt",
    "bedrock:ListPrompts", "bedrock:CreatePromptVersion",
    "ssm:PutParameter", "ssm:GetParametersByPath",
    "secretsmanager:CreateSecret", "secretsmanager:DescribeSecret",
    "secretsmanager:GetSecretValue",
]
GATEWAY = [
    "lambda:CreateFunction", "lambda:UpdateFunctionCode", "lambda:GetFunction",
    "lambda:UpdateFunctionConfiguration", "lambda:AddPermission",
    "bedrock-agentcore:CreateGateway", "bedrock-agentcore:GetGateway",
    "bedrock-agentcore:ListGateways", "bedrock-agentcore:CreateGatewayTarget",
    "bedrock-agentcore:UpdateGatewayTarget", "bedrock-agentcore:ListGatewayTargets",
]
TRANSACTION_SEARCH = ["logs:PutResourcePolicy", "xray:UpdateTraceSegmentDestination"]


def _sh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def check_aws(session) -> tuple[str, str, str]:
    if not session.region_name:
        raise SystemExit("no AWS region configured — set one with `aws configure`")
    try:
        identity = session.client("sts").get_caller_identity()
    except Exception as exc:
        raise SystemExit(f"AWS credentials are not usable: {exc}")
    print(f"  AWS      {identity['Arn']}  region={session.region_name}")
    return identity["Account"], identity["Arn"], session.region_name


def check_docker(sh=_sh) -> None:
    if sh(["docker", "version", "--format", "{{.Server.Version}}"]).returncode != 0:
        raise SystemExit("Docker is not running — start Docker Desktop (or the docker daemon)")
    if sh(["docker", "buildx", "version"]).returncode != 0:
        raise SystemExit("docker buildx is missing — it ships with Docker Desktop; on Linux "
                         "install the docker-buildx-plugin package")
    inspect = sh(["docker", "buildx", "inspect", "--bootstrap"])
    platforms = " ".join(line for line in inspect.stdout.splitlines() if "Platforms" in line)
    if "linux/arm64" not in platforms:
        raise SystemExit(
            "the current buildx builder cannot build linux/arm64, which AgentCore Runtime "
            "requires. Add arm64 emulation once with:\n"
            "  docker run --privileged --rm tonistiigi/binfmt --install arm64\n"
            "(not run automatically: it starts a privileged container on this machine)")
    print("  Docker   daemon up, buildx present, linux/arm64 buildable")


def principal_arn(caller_arn: str, iam) -> str | None:
    """The IAM ARN to simulate. An assumed role (SSO, role profiles) reports an
    STS ARN; the simulator needs the role's own ARN, path included."""
    if ":assumed-role/" in caller_arn:
        role = caller_arn.split(":assumed-role/")[1].split("/")[0]
        return iam.get_role(RoleName=role)["Role"]["Arn"]
    if caller_arn.endswith(":root"):
        return None                                   # root: nothing to simulate
    return caller_arn


def transaction_search_enabled(session) -> bool | None:
    try:
        return session.client("xray").get_trace_segment_destination() \
                      .get("Destination") == "CloudWatchLogs"
    except Exception:
        return None                                    # cannot tell


def check_permissions(session, caller_arn: str, actions: list[str]) -> list[str] | None:
    """Actions not allowed — or None when nothing could be checked, so the
    caller never reports a skipped check as a passed one."""
    iam = session.client("iam")
    try:
        arn = principal_arn(caller_arn, iam)
        if arn is None:
            print("  IAM      root credentials — permission check skipped. Root keys cannot "
                  "be restricted by any policy; prefer an IAM user or SSO role.")
            return None
        denied = []
        for start in range(0, len(actions), 50):
            page = iam.simulate_principal_policy(PolicySourceArn=arn,
                                                 ActionNames=actions[start:start + 50])
            denied += [r["EvalActionName"] for r in page["EvaluationResults"]
                       if r["EvalDecision"] != "allowed"]
    except Exception as exc:
        print(f"  IAM      WARNING: could not simulate permissions ({type(exc).__name__}); "
              "continuing — a missing permission will fail at the call that needs it")
        return None
    return denied


def run(needs_gateway: bool, session=None, sh=_sh, needs_docker: bool = True) -> None:
    """needs_docker=False for `deploy.py --image`: the image is copied from
    Docker Hub into ECR over HTTPS (infra/image_copy.py), so Docker is never
    started and its absence is not an error."""
    print("=== preflight ===")
    session = session or boto3.Session()
    _, caller, _ = check_aws(session)
    if needs_docker:
        check_docker(sh)
    else:
        print("  Docker   not needed — the image is copied from Docker Hub")

    actions = COMMON + (GATEWAY if needs_gateway else [])
    if transaction_search_enabled(session) is not True:
        actions += TRANSACTION_SEARCH + ["xray:GetTraceSegmentDestination"]
    denied = check_permissions(session, caller, actions)
    if denied is None:
        return
    if denied:
        raise SystemExit("the deploying identity lacks these permissions:\n  "
                         + "\n  ".join(sorted(set(denied)))
                         + "\nAsk an administrator to grant them; nothing has been created yet.")
    print(f"  IAM      {len(actions)} actions allowed")
