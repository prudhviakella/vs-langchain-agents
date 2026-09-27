"""Every boto3 call in the project, checked against boto3's own service models:
parameter names exist, required parameters are passed.

Found this way, before: runtimeSessionId below its 33-char minimum, list_gateways
items read for fields they do not carry, protocolConfiguration never passed.
This cannot catch IAM, quotas or timing — only calls that cannot succeed.
"""
import ast

import boto3
import pytest

import fakes

# receiver variable -> service. Every client in the project is bound to one of these.
RECEIVERS = {
    "bedrock": "bedrock", "agent": "bedrock-agent", "iam": "iam", "ecr": "ecr",
    "lambda_client": "lambda", "runtime_client": "bedrock-agentcore-control",
    "sm": "secretsmanager", "ssm": "ssm", "xray": "xray", "logs": "logs",
}
# receivers whose service depends on the file
BY_FILE = {"client": {"gateway.py": "bedrock-agentcore-control"}}
CALL_RECEIVERS = {"_get_client": "bedrock-agentcore", "_bedrock": "bedrock-runtime"}


def _pascal(op):
    return "".join(p.title() for p in op.split("_"))


def call_sites():
    for path in sorted(fakes.ROOT.rglob("*.py")):
        if "tests" in path.parts or "_build" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            recv = node.func.value
            service = None
            if isinstance(recv, ast.Name):
                service = RECEIVERS.get(recv.id) or BY_FILE.get(recv.id, {}).get(path.name)
            elif isinstance(recv, ast.Call) and isinstance(recv.func, ast.Name):
                service = CALL_RECEIVERS.get(recv.func.id)
            elif isinstance(recv, ast.BoolOp):          # (client or _bedrock()).apply_guardrail
                service = "bedrock-runtime" if node.func.attr == "apply_guardrail" else None
            if service:
                yield path, node, service


SITES = list(call_sites())


def test_sites_found():
    assert len(SITES) > 60


@pytest.mark.parametrize("path,node,service", SITES,
                         ids=[f"{p.parent.parent.name}/{p.name}:{n.lineno}:{n.func.attr}"
                              for p, n, _ in SITES])
def test_call_matches_service_model(path, node, service):
    model = boto3.client(service, region_name="us-east-1").meta.service_model
    op = _pascal(node.func.attr)
    if op not in model.operation_names:
        pytest.skip(f"{node.func.attr} is not an API operation (waiter/paginator/helper)")
    shape = model.operation_model(op).input_shape
    if any(k.arg is None for k in node.keywords):
        passed = {k.arg for k in node.keywords if k.arg}
        assert not passed - set(shape.members), f"unknown params {passed - set(shape.members)}"
        return                                           # **kwargs: required may come from it
    passed = {k.arg for k in node.keywords}
    assert not passed - set(shape.members), f"unknown params {passed - set(shape.members)}"
    assert not set(shape.required_members) - passed, \
        f"missing required {set(shape.required_members) - passed}"
