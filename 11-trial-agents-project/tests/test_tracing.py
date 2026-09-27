"""Tracing, proved with OpenTelemetry's own SDK: real trace ids, real parent links.

    supervisor span ──traceParent──► invoke_agent_runtime ──► specialist execute()
         same trace_id, and the specialist's root span is the CHILD of the
         supervisor's call_agent span — one trace for the whole fan-out.
"""
import ast
import asyncio
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
    with sup.span("supervisor.call_agent") as parent:
        agent_client.call_specialist("arn:x", "trial_graph", "q", "c" * 36)
    assert sent["traceParent"].split("-")[1] == hex_trace(parent)

    # the specialist side, from the headers the Runtime delivers
    token = spec.attach_context(spec.extract_parent_context({"traceparent": sent["traceParent"]}))
    try:
        with spec.span("trial_graph.request"):
            pass
    finally:
        spec.detach_context(token)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    child, root = spans["trial_graph.request"], spans["supervisor.call_agent"]
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


@pytest.mark.parametrize("agent,executor", [("trial_graph", "TrialGraphExecutor"),
                                            ("trial_search", "TrialSearchExecutor")])
def test_executor_joins_the_callers_trace(agent, executor):
    """The real execute(): headers captured by bedrock_agentcore -> attached ->
    the agent's root span belongs to the caller's trace."""
    exporter = otel()
    main = load_main(agent)
    schemas = importlib.import_module(f"{agent}.schemas")
    response_type = next(getattr(schemas, n) for n in dir(schemas) if n.endswith("Response"))

    async def orchestrate(question):
        return response_type(result_shape="empty")
    main.orchestrate = orchestrate
    main.BedrockAgentCoreContext.set_request_headers({"traceparent": TRACEPARENT})

    class Queue:
        async def enqueue_event(self, event): self.event = event
    ctx = type("Ctx", (), {"get_user_input": lambda self: "q", "context_id": "ctx-1"})()
    asyncio.run(getattr(main, executor)().execute(ctx, Queue()))

    root = next(s for s in exporter.get_finished_spans() if s.name == f"{agent}.request")
    assert hex_trace(root) == TRACE_ID
    assert root.attributes[f"{agent}.result_shape"] == "empty"


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
