"""The full chain, across real processes:

    supervisor.orchestrate ─ real call_agent ─ real agent_client
        │  the same envelope bytes invoke_agent_runtime would send, over HTTP
        ├─► trial_graph  main.py  (serve_a2a) ─ real loop ─ real Lambda handler
        └─► trial_search main.py  (serve_a2a) ─ real loop ─ real Lambda handler
    ◄── real parse_a2a_response ─ render_decision ─ compose ─ OUTPUT guardrail
"""
import asyncio
import io
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
import requests
from langchain_core.messages import AIMessage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fakes                                                     # noqa: E402
from fakes import Fake, GuardrailClient, call                    # noqa: E402
from supervisor import agent_client, config, core, guardrail    # noqa: E402

PORTS = {"trial_graph": 9301, "trial_search": 9302}
SPAN_FILES = {a: Path(f"/tmp/e2e_spans_{a}.jsonl") for a in PORTS}
for f in SPAN_FILES.values():
    f.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def servers():
    import os
    procs = [subprocess.Popen([sys.executable, str(HERE / "serve_specialist.py"), a, str(p)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              env={**os.environ, "E2E_SPAN_FILE": str(SPAN_FILES[a])})
             for a, p in PORTS.items()]
    try:
        for port in PORTS.values():
            for _ in range(80):
                try:
                    if requests.get(f"http://127.0.0.1:{port}/ping", timeout=1).ok:
                        break
                except requests.RequestException:
                    time.sleep(0.25)
            else:
                pytest.fail(f"server on {port} never started")
        yield procs
    finally:
        for p in procs:
            p.terminate()
        # A server that raised while handling a request still answers the
        # caller with a JSON-RPC error; only its own stderr shows the traceback.
        crashes = []
        for (name, _), p in zip(PORTS.items(), procs):
            _, err = p.communicate(timeout=10)
            if b"Traceback" in err:
                crashes.append(f"{name}:\n{err.decode()[-1500:]}")
        assert not crashes, "specialist server raised:\n" + "\n".join(crashes)


def test_supervisor_to_both_specialists_and_back(servers):
    settings = fakes.settings_for("supervisor")
    url = {spec["arn"]: f"http://127.0.0.1:{PORTS[name]}/"
           for name, spec in settings.specialists.items()}
    invocations = []

    class Transport:            # boto3 bedrock-agentcore stand-in, same call shape
        def invoke_agent_runtime(self, **kw):
            invocations.append(kw)
            # The Runtime turns traceParent / traceState / baggage into the
            # traceparent / tracestate / baggage headers the container receives.
            headers = {"Content-Type": "application/json"}
            for param, header in (("traceParent", "traceparent"),
                                  ("traceState", "tracestate"), ("baggage", "baggage")):
                if kw.get(param):
                    headers[header] = kw[param]
            r = requests.post(url[kw["agentRuntimeArn"]], data=kw["payload"],
                              headers=headers, timeout=60)
            return {"response": io.BytesIO(r.content)}

    agent_client._client = Transport()
    exporter = fakes.otel()
    guardrail._client = GuardrailClient()
    config.use(settings)
    models = iter([
        Fake(messages=iter([
            call("call_agent", agent_name="trial_graph", question="Pfizer trials?",
                 rationale="registry fact"),
            call("call_agent", agent_name="trial_search", question="Neonates enrolled?",
                 rationale="protocol text", observation="one Pfizer trial"),
            call("SupervisorDecision", entities=["NCT02951156"], answerable=True,
                 clarifying_question="", note="")])),
        Fake(messages=iter([AIMessage(content="Pfizer sponsors NCT02951156; neonates are "
                                              "not enrolled.")]))])
    cls = settings.__class__
    real = cls.chat_model
    cls.chat_model = lambda self: next(models)
    try:
        conversation = str(uuid.uuid4())
        response = asyncio.run(core.orchestrate("Pfizer trials, and neonates?", conversation))
    finally:
        cls.chat_model = real

    assert len(invocations) == 2
    for kw in invocations:
        assert 33 <= len(kw["runtimeSessionId"]) <= 256
        assert kw["runtimeSessionId"].startswith(conversation)
        assert json.loads(kw["payload"])["params"]["message"]["contextId"] == conversation
    assert len({kw["runtimeSessionId"] for kw in invocations}) == 2
    assert [(c.agent_name, c.result_shape, c.succeeded) for c in response.calls] == [
        ("trial_graph", "table", True), ("trial_search", "passages", True)]
    assert response.render_target == "chart"
    assert "NCT02951156" in response.composed_answer

    # ── one trace across three processes ─────────────────────────────────
    sup = {s.name: s for s in exporter.get_finished_spans()}
    calls = [s for s in exporter.get_finished_spans() if s.name == "supervisor.call_agent"]
    trace_id = format(sup["supervisor.route"].context.trace_id, "032x")
    for agent, caller in zip(PORTS, calls):
        remote = [json.loads(line) for line in SPAN_FILES[agent].read_text().splitlines()]
        root = next(r for r in remote if r["name"] == f"{agent}.request")
        assert root["trace"] == trace_id, f"{agent} is not in the supervisor's trace"
        assert root["parent"] == format(caller.context.span_id, "016x"), \
            f"{agent}'s root span is not the child of its call_agent span"
        assert any(r["name"].startswith("guardrail.") for r in remote), \
            f"{agent}'s guardrail spans are missing"
