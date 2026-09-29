"""Every nested payload the deploy builds, validated by botocore's OWN
ParamValidator — the code that rejected a tool schema's "enum" key at
create_gateway_target, before anything was sent.

test_aws_calls.py checks call sites statically: top-level parameter names
only. That missed a bad key three levels deep in targetConfiguration. This
drives the real infra functions, create path and update path, against a
client that validates every call exactly as botocore does client-side.
"""
import importlib.util
import os

import boto3
import pytest
from botocore.validate import ParamValidator

import fakes

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


class Recording:
    """A boto3 client stand-in: every call is validated against the real
    service model, recorded, then answered from `responses`."""

    def __init__(self, service, responses):
        self.model = boto3.client(service, region_name="us-east-1").meta.service_model
        self.responses, self.calls = responses, []
        self.exceptions = type("E", (), {n: type(n, (Exception,), {}) for n in (
            "ResourceNotFoundException", "ResourceConflictException", "NoSuchEntityException",
            "RepositoryNotFoundException")})

    def __getattr__(self, name):
        op = "".join(p.title() for p in name.split("_"))
        if op not in self.model.operation_names:
            raise AttributeError(name)

        def call(**kwargs):
            report = ParamValidator().validate(kwargs, self.model.operation_model(op).input_shape)
            if report.has_errors():
                raise AssertionError(f"{name}: {report.generate_report()}")
            self.calls.append(name)
            answer = self.responses.get(name, {})
            return answer(**kwargs) if callable(answer) else answer
        return call

    def get_waiter(self, name):
        return type("W", (), {"wait": lambda self, **kw: None})()


def infra(agent, module):
    spec = importlib.util.spec_from_file_location(
        f"{agent}_{module}", fakes.ROOT / agent / "infra" / f"{module}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GATEWAY_OK = {"gatewayId": "g", "gatewayArn": "arn:g", "gatewayUrl": "https://g", "status": "READY"}


@pytest.mark.parametrize("agent", ["trial_graph", "trial_search"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
def test_gateway_and_tool_schema(agent, existing):
    gw = infra(agent, "gateway")
    gw.client = Recording("bedrock-agentcore-control", {
        "list_gateways": {"items": [{"gatewayId": "g", "name": gw.GATEWAY_NAME}] if existing else []},
        "create_gateway": GATEWAY_OK, "get_gateway": GATEWAY_OK,
        "list_gateway_targets": {"items": [{"targetId": "t", "name": gw.TARGET_NAME}] if existing else []},
        "create_gateway_target": {"targetId": "t"}})
    gw.create_gateway("arn:aws:iam::1:role/r")
    gw.create_lambda_target("g", "arn:aws:lambda:us-east-1:1:function:f")
    assert ("update_gateway_target" if existing else "create_gateway_target") in gw.client.calls


@pytest.mark.parametrize("agent", ["trial_graph", "trial_search", "supervisor"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
def test_guardrail(agent, existing):
    g = infra(agent, "guardrail")
    draft_policy = {"contentPolicy": {"filters": []}, "status": "READY"}   # differs -> update
    g.bedrock = Recording("bedrock", {
        "list_guardrails": lambda **kw: {"guardrails": (
            [{"id": "gr", "arn": "arn:gr", "name": g.NAME, "version": "DRAFT"}]
            if existing and "guardrailIdentifier" not in kw else [])},
        "create_guardrail": {"guardrailId": "gr", "guardrailArn": "arn:gr"},
        "get_guardrail": draft_policy, "create_guardrail_version": {"version": "1"}})
    assert g.ensure_guardrail()["version"] == "1"
    assert ("update_guardrail" in g.bedrock.calls) == existing


@pytest.mark.parametrize("agent,files", [("trial_graph", ["system.md"]),
                                         ("trial_search", ["system.md"]),
                                         ("supervisor", ["system.md", "compose.md"])])
def test_real_prompt_files_publish(agent, files):
    p = infra(agent, "prompts")
    for name in files:
        p.agent = Recording("bedrock-agent", {
            "list_prompts": {"promptSummaries": []},
            "create_prompt": {"id": "PR", "arn": "arn:pr", "version": "DRAFT"},
            "create_prompt_version": {"version": "1"}})
        out = p.publish(f"{agent}-{name}", fakes.ROOT / agent / "prompts" / name, "d")
        assert out["version"] == "1"


@pytest.mark.parametrize("agent", ["trial_graph", "trial_search", "supervisor"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
def test_runtime(agent, existing):
    rd = infra(agent, "runtime_deploy")
    rd.runtime_client = Recording("bedrock-agentcore-control", {
        "list_agent_runtimes": {"agentRuntimes": [
            {"agentRuntimeName": rd.RUNTIME_NAME, "agentRuntimeId": "r", "agentRuntimeArn": "arn:r"}]
            if existing else []},
        "create_agent_runtime": {"agentRuntimeId": "r", "agentRuntimeArn": "arn:r"},
        "get_agent_runtime": {"status": "READY"}})
    rd.deploy_runtime("1.dkr.ecr.us-east-1.amazonaws.com/x:latest", "arn:aws:iam::1:role/r",
                      {"PARAM_PREFIX": f"/trial-agents/{agent}", "AWS_REGION": "us-east-1"})


@pytest.mark.parametrize("agent", ["trial_graph", "trial_search"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
def test_lambda(agent, existing):
    ld = infra(agent, "lambda_deploy")
    ld._build_zip = lambda: b"zip"

    def get_function(**kw):
        if not existing:
            raise ld.lambda_client.exceptions.ResourceNotFoundException()
        return {"Configuration": {"FunctionArn": "arn:f"}}
    ld.lambda_client = Recording("lambda", {"get_function": get_function,
                                            "create_function": {"FunctionArn": "arn:f"}})
    args = ("arn:aws:iam::1:role/r", "trial-graph/neo4j") if agent == "trial_graph" else \
           ("arn:aws:iam::1:role/r", {"PINECONE_INDEX": "rag-docs"})
    ld.deploy(*args)
