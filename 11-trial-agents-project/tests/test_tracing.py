"""Tracing, proved with OpenTelemetry's own SDK: real trace ids, real parent links.

    supervisor span ──traceParent──► invoke_agent_runtime ──► specialist execute()
         same trace_id, and the specialist's root span is the CHILD of the
         supervisor's call_agent span — one trace for the whole fan-out.
"""
import ast
import asyncio
import json
import importlib
import importlib.util
import sys

import pytest

import fakes
from fakes import ROOT, otel

TRACE_ID = "6a6570e21d257bbf04c8d6927448c208"
PARENT = "9f3f20ec8799fab3"
TRACEPARENT = f"00-{TRACE_ID}-{PARENT}-01"
AGENTS = ("trial_graph", "trial_search", "supervisor")


def tracing(agent):
    return importlib.import_module(f"{agent}.tracing")


def hex_trace(span):
    return format(span.context.trace_id, "032x")


@pytest.mark.parametrize("agent", AGENTS)
def test_inbound_traceparent_becomes_the_trace_of_new_spans(agent):
    exporter, t = otel(), tracing(agent)
    token = t.attach_context(t.extract_parent_context({"Traceparent": TRACEPARENT}))
    try:
        with t.span(f"{agent}.probe", answer=42):
            pass
    finally:
        t.detach_context(token)
    (sp,) = exporter.get_finished_spans()
    assert hex_trace(sp) == TRACE_ID
    assert format(sp.parent.span_id, "016x") == PARENT
    assert sp.attributes[f"{agent}.answer"] == 42


@pytest.mark.parametrize("headers", [None, {}, {"x-amzn-bedrock-agentcore-session-id": "a"},
                                     {"traceparent": "not-a-traceparent"}])
def test_missing_or_bad_headers_never_raise(headers):
    t = tracing("trial_graph")
    t.detach_context(t.attach_context(t.extract_parent_context(headers)))


def test_errors_are_recorded_and_still_raised():
    exporter, t = otel(), tracing("trial_search")
    with pytest.raises(ValueError):
        with t.span("trial_search.boom"):
            raise ValueError("x")
    (sp,) = exporter.get_finished_spans()
    assert sp.status.status_code.name == "ERROR" and sp.events[0].name == "exception"


def test_supervisor_call_carries_its_span_as_the_specialists_parent():
    """The fan-out is ONE trace: the traceParent the supervisor passes to
    invoke_agent_runtime, extracted on the other side, parents the specialist."""
    exporter = otel()
    from supervisor import agent_client
    sent = {}

    class Client:
        def invoke_agent_runtime(self, **kw):
            sent.update(kw)
            import io
            return {"response": io.BytesIO(
                b'{"jsonrpc":"2.0","id":"1","result":{"parts":[{"text":"{\\"result_shape\\":\\"empty\\"}"}]}}')}
    agent_client._client = Client()

    sup, spec = tracing("supervisor"), tracing("trial_graph")
    with sup.span("call_agent trial_graph") as parent:
        agent_client.call_specialist("arn:x", "trial_graph", "q", "c" * 36)

    # Regression: trace context must NOT be passed as invoke_agent_runtime
    # parameters. They become SigV4-signed headers, and in AWS every
    # specialist call then failed "The request signature we calculated does
    # not match the signature you provided".
    assert not {"traceParent", "traceState", "baggage"} & set(sent)
    metadata = json.loads(sent["payload"])["params"]["message"]["metadata"]
    assert metadata["traceparent"].split("-")[1] == hex_trace(parent)

    # the specialist side, from the message metadata it receives
    token = spec.attach_context(spec.extract_parent_context(metadata))
    try:
        with spec.agent_span("trial_graph", conversation_id="c" * 36):
            pass
    finally:
        spec.detach_context(token)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    child, root = spans["invoke_agent trial_graph"], spans["call_agent trial_graph"]
    assert hex_trace(child) == hex_trace(root)
    assert child.parent.span_id == root.context.span_id


def load_main(agent):
    path = str(ROOT / agent / "agent_code")
    if path not in sys.path:
        sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(f"{agent}_main", f"{path}/main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("carrier", ["header", "message_metadata"])
@pytest.mark.parametrize("agent,executor", [("trial_graph", "TrialGraphExecutor"),
                                            ("trial_search", "TrialSearchExecutor"),
                                            ("supervisor", "SupervisorExecutor")])
def test_executor_joins_the_callers_trace(agent, executor, carrier):
    """The real execute(): headers captured by bedrock_agentcore -> attached ->
    the agent's root span belongs to the caller's trace."""
    exporter = otel()
    main = load_main(agent)
    schemas = importlib.import_module(f"{agent}.schemas")
    if agent == "supervisor":
        response = schemas.SupervisorResponse(decision=schemas.SupervisorDecision(answerable=True))
        expected = ("answerable", True)
    else:
        response_type = next(getattr(schemas, n) for n in dir(schemas)
                             if n.endswith("Response") and n.startswith("Trial"))
        response = response_type(result_shape="empty")
        expected = ("result_shape", "empty")

    async def orchestrate(question, *context_id, **history_and_user):
        return response
    main.orchestrate = orchestrate
    if carrier == "header":
        main.BedrockAgentCoreContext.set_request_headers({"traceparent": TRACEPARENT})
        message = None
    else:                                    # how the supervisor sends it
        main.BedrockAgentCoreContext.set_request_headers({})
        message = type("M", (), {"metadata": {"traceparent": TRACEPARENT}})()

    class Queue:
        async def enqueue_event(self, event): self.event = event
    ctx = type("Ctx", (), {"get_user_input": lambda self: "q", "context_id": "ctx-1",
                           "message": message})()
    asyncio.run(getattr(main, executor)().execute(ctx, Queue()))

    root = next(s for s in exporter.get_finished_spans() if s.name == f"invoke_agent {agent}")
    # classified and labelled the way AgentCore Observability reads spans
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.agent.name"] == agent
    assert root.attributes["gen_ai.conversation.id"] == "ctx-1"
    assert root.attributes["openinference.span.kind"] == "AGENT"
    assert hex_trace(root) == TRACE_ID
    assert root.attributes[f"{agent}.{expected[0]}"] == expected[1]


def test_guardrail_checks_are_spans():
    exporter = otel()
    from trial_search import guardrail
    guardrail.check("hello", "INPUT", "g", "1", client=fakes.GuardrailClient())
    (sp,) = exporter.get_finished_spans()
    assert sp.name == "guardrail.input" and sp.attributes["trial_search.action"] == "NONE"


# ── the wiring that makes spans leave the container ─────────────────────
@pytest.mark.parametrize("agent", AGENTS)
def test_container_exports_through_the_aws_distro(agent):
    docker = (ROOT / agent / "agent_code" / "Dockerfile").read_text()
    deps = (ROOT / agent / "agent_code" / "pyproject.toml").read_text()
    assert 'CMD ["opentelemetry-instrument", "python", "main.py"]' in docker
    assert f"OTEL_SERVICE_NAME={agent}" in docker
    assert "aws-opentelemetry-distro" in deps and "openinference-instrumentation-langchain" in deps


@pytest.mark.parametrize("agent", AGENTS)
def test_instrumentor_runs_first_and_is_guarded(agent):
    tree = ast.parse((ROOT / agent / "agent_code" / "main.py").read_text())
    body = [n for n in tree.body if not isinstance(n, ast.Expr)]    # skip docstring
    assert isinstance(body[0], ast.Try), "instrumentation must be the first statement, in try"
    assert "LangChainInstrumentor" in ast.unparse(body[0])
    first_langchain = min(i for i, n in enumerate(body)
                          if isinstance(n, (ast.Import, ast.ImportFrom))
                          and f"{agent}." in ast.unparse(n))
    assert first_langchain > 0
