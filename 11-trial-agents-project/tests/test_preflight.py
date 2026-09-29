"""Preflight and the automated ECR login — each failure stops the deploy
before anything is created, and names what is missing."""
import base64
import importlib.util
import subprocess

import pytest

import fakes

pf = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "preflight", fakes.ROOT / "trial_graph" / "infra" / "preflight.py"))
pf.__spec__.loader.exec_module(pf)


def shell(daemon=True, buildx=True, platforms="linux/amd64, linux/arm64"):
    def sh(args):
        ok = {"version": daemon}.get(args[1], True)
        if args[1:3] == ["buildx", "version"]:
            ok = buildx
        out = f"Name: default\nPlatforms: {platforms}\n" if args[1:3] == ["buildx", "inspect"] else ""
        return subprocess.CompletedProcess(args, 0 if ok else 1, out, "")
    return sh


@pytest.mark.parametrize("kwargs,expected", [
    (dict(daemon=False), "Docker is not running"),
    (dict(buildx=False), "buildx is missing"),
    (dict(platforms="linux/amd64"), "tonistiigi/binfmt --install arm64"),
])
def test_docker_problems_stop_with_the_fix(kwargs, expected):
    with pytest.raises(SystemExit, match=expected):
        pf.check_docker(shell(**kwargs))


def test_docker_ok():
    pf.check_docker(shell())


class IAM:
    def __init__(self, deny=(), simulate_raises=False):
        self.deny, self.raises, self.pages = set(deny), simulate_raises, 0

    def get_role(self, RoleName):
        return {"Role": {"Arn": f"arn:aws:iam::1:role/aws-reserved/sso.amazonaws.com/{RoleName}"}}

    def simulate_principal_policy(self, PolicySourceArn, ActionNames):
        if self.raises:
            raise RuntimeError("AccessDenied")
        assert ":role/" in PolicySourceArn or ":user/" in PolicySourceArn
        self.pages += 1
        return {"EvaluationResults": [
            {"EvalActionName": a, "EvalDecision": "implicitDeny" if a in self.deny else "allowed"}
            for a in ActionNames]}


class XRay:
    def __init__(self, destination): self.destination = destination
    def get_trace_segment_destination(self): return {"Destination": self.destination}


class Session:
    region_name = "us-east-1"

    def __init__(self, iam, xray_destination="XRay",
                 arn="arn:aws:sts::1:assumed-role/AWSReservedSSO_Admin_ab12/prudhvi"):
        self.iam, self.xray, self.arn, self.created = iam, XRay(xray_destination), arn, []

    def client(self, name, **kw):
        if name == "sts":
            return type("S", (), {"get_caller_identity": lambda s: {"Account": "1", "Arn": self.arn}})()
        return {"iam": self.iam, "xray": self.xray}[name]


def test_sso_role_is_simulated_as_its_iam_role_with_path():
    assert pf.principal_arn("arn:aws:sts::1:assumed-role/AWSReservedSSO_Admin_ab12/me", IAM()) \
        == "arn:aws:iam::1:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_Admin_ab12"
    assert pf.principal_arn("arn:aws:iam::1:user/ci", IAM()) == "arn:aws:iam::1:user/ci"
    assert pf.principal_arn("arn:aws:iam::1:root", IAM()) is None


def test_missing_permissions_are_named_before_anything_is_created():
    iam = IAM(deny={"lambda:CreateFunction", "logs:PutResourcePolicy"})
    with pytest.raises(SystemExit) as stop:
        pf.run(needs_gateway=True, session=Session(iam), sh=shell())
    assert "lambda:CreateFunction" in str(stop.value) and "logs:PutResourcePolicy" in str(stop.value)
    assert "nothing has been created" in str(stop.value)


def test_transaction_search_permissions_only_while_it_is_off():
    iam = IAM(deny={"logs:PutResourcePolicy", "xray:UpdateTraceSegmentDestination"})
    pf.run(needs_gateway=True, session=Session(iam, xray_destination="CloudWatchLogs"), sh=shell())
    with pytest.raises(SystemExit, match="xray:UpdateTraceSegmentDestination"):
        pf.run(needs_gateway=True, session=Session(iam, xray_destination="XRay"), sh=shell())


def test_supervisor_is_not_asked_for_gateway_permissions():
    iam = IAM(deny={"lambda:CreateFunction"})
    pf.run(needs_gateway=False, session=Session(iam, "CloudWatchLogs"), sh=shell())


def test_simulator_unavailable_warns_and_continues(capsys):
    pf.run(needs_gateway=True, session=Session(IAM(simulate_raises=True)), sh=shell())
    assert "could not simulate" in capsys.readouterr().out


def test_actions_are_simulated_in_pages_of_50():
    iam = IAM()
    pf.check_permissions(Session(iam), "arn:aws:iam::1:user/ci", [f"s3:A{i}" for i in range(120)])
    assert iam.pages == 3


def test_ecr_login_passes_the_password_on_stdin_only(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")   # the module builds clients at import
    spec = importlib.util.spec_from_file_location(
        "rd", fakes.ROOT / "supervisor" / "infra" / "runtime_deploy.py")
    rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
    token = base64.b64encode(b"AWS:s3cr3t-token").decode()
    rd.ecr = type("E", (), {"get_authorization_token": lambda self: {"authorizationData": [
        {"authorizationToken": token, "proxyEndpoint": "https://1.dkr.ecr.us-east-1.amazonaws.com"}]}})()
    seen = {}

    def fake_run(args, **kw):
        seen.update(args=args, input=kw.get("input"))
        return subprocess.CompletedProcess(args, 0, "Login Succeeded", "")
    monkeypatch.setattr(rd.subprocess, "run", fake_run)
    assert rd.ecr_login() == "1.dkr.ecr.us-east-1.amazonaws.com"
    assert seen["args"] == ["docker", "login", "--username", "AWS", "--password-stdin",
                            "1.dkr.ecr.us-east-1.amazonaws.com"]
    assert "s3cr3t-token" not in " ".join(seen["args"]) and seen["input"] == "s3cr3t-token"


def test_root_never_reports_actions_as_allowed(capsys):
    """Root skips the check; the output must not then claim anything passed."""
    pf.run(needs_gateway=True, session=Session(IAM(), "CloudWatchLogs",
                                               arn="arn:aws:iam::1:root"), sh=shell())
    out = capsys.readouterr().out
    assert "permission check skipped" in out and "allowed" not in out


def test_unavailable_simulator_never_reports_actions_as_allowed(capsys):
    pf.run(needs_gateway=True, session=Session(IAM(simulate_raises=True)), sh=shell())
    assert "allowed" not in capsys.readouterr().out
