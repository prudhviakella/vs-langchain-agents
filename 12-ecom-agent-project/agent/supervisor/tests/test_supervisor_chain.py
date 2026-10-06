"""The real chain, over real HTTP: supervisor ──A2A stream──► NLQ ──► PostgreSQL
                                              └──A2A stream──► chart_gen

    NLQ and chart_gen run as ACTUAL A2A servers (uvicorn, background threads, local ports);
    the supervisor calls them exactly as in local mode (NLQ_URL / CHART_URL).
    stand-ins: the three agents' MODEL calls only (scripted), and Pinecone (grounding pinned).

    query path (live NLQ progress, SQL as reasoning, real rows in the table, chart attached, tokens) ·
    answer path · clarify path · a failing NLQ still yields a final answer · the supervisor's own A2A wire
"""
import asyncio
import importlib.util
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
import uvicorn
from langchain_core.messages import AIMessageChunk

AGENT = Path(__file__).resolve().parents[2]
for sub in ("nlq", "chart_gen", "supervisor"):
    sys.path.insert(0, str(AGENT / sub / "agent_code"))
sys.path.insert(0, str(AGENT / "nlq" / "tests"))
sys.path.insert(0, str(AGENT / "chart_gen" / "tests"))


def _port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


NLQ_PORT, CHART_PORT = _port(), _port()
os.environ.update(NLQ_DB_MODE="local", NLQ_URL=f"http://127.0.0.1:{NLQ_PORT}/", CHART_URL=f"http://127.0.0.1:{CHART_PORT}/")

pytestmark = pytest.mark.skipif(not os.environ.get("PGHOST"), reason="needs a local PostgreSQL (PG* env)")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


from a2a.server.apps import A2AStarletteApplication  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore  # noqa: E402
from chart_gen import core as chart_core  # noqa: E402
from chart_gen.schemas import ChartDecision  # noqa: E402
from nlq import core as nlq_core, grounding  # noqa: E402
from supervisor import core as sup_core, specialists  # noqa: E402
from supervisor.schemas import RouteDecision  # noqa: E402
from test_chart_gen import StubModel  # noqa: E402
from test_nlq_agent import Fake, call, decide  # noqa: E402

NLQ_MAIN = _load("nlq_main", AGENT / "nlq" / "agent_code" / "main.py")
CHART_MAIN = _load("chart_main", AGENT / "chart_gen" / "agent_code" / "main.py")
SUP_MAIN = _load("sup_main", AGENT / "supervisor" / "agent_code" / "main.py")


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close(); return server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server on {port} did not start")


def _app(main, executor):
    return A2AStarletteApplication(agent_card=main.agent_card(), http_handler=DefaultRequestHandler(executor, InMemoryTaskStore())).build()


CARRIER_SQL = "SELECT carrier, COUNT(*) AS shipments FROM ecom.shipments GROUP BY carrier ORDER BY shipments DESC"
SCRIPT = {}


@pytest.fixture(scope="module", autouse=True)
def servers():
    grounding.retrieve = lambda q: grounding.Grounding("TABLE ecom.shipments", ["nlq_table:ecom.shipments"], ["ecom.shipments"])
    nlq_core.agent = lambda: nlq_core.build(model=Fake(messages=iter(SCRIPT["nlq"])))
    chart_core.chart_model = lambda: StubModel(SCRIPT["chart"])
    a = _serve(_app(NLQ_MAIN, NLQ_MAIN.NLQExecutor()), NLQ_PORT)
    b = _serve(_app(CHART_MAIN, CHART_MAIN.ChartExecutor()), CHART_PORT)
    yield
    a.should_exit = b.should_exit = True


class Route:
    def __init__(self, decision): self.decision = decision
    def with_structured_output(self, schema, method, include_raw): return self
    async def ainvoke(self, messages):
        return {"parsed": self.decision, "raw": NS(usage_metadata={"input_tokens": 500, "output_tokens": 60, "total_tokens": 560})}


class Compose:
    def __init__(self, words): self.words = words
    async def astream(self, messages):
        self.seen = messages
        for w in self.words:
            yield AIMessageChunk(content=w)
        yield AIMessageChunk(content="", usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940})


def run(decision, compose=None):
    events = []

    async def go():
        async for ev in sup_core.orchestrate("which carrier ships the most?", [], "conv-123",
                                             route_model=Route(decision), compose_model=compose or Compose([])):
            events.append(ev)
    asyncio.run(go())
    return events


def test_query_path_end_to_end():
    SCRIPT["nlq"] = [call("execute_sql", sql=CARRIER_SQL), decide(note="counts all shipments")]
    SCRIPT["chart"] = ChartDecision(chart_type="bar", insight="USPS ships the most", figures=[{
        "data": [{"type": "bar", "x": ["USPS", "UPS", "FedEx"], "y": [8051, 8035, 7990]}], "layout": {}}])
    compose = Compose(["**USPS** ", "ships the most: ", "8,051 shipments."])
    events = run(RouteDecision(action="query", questions=["Shipments per carrier"], rationale="Count shipments by carrier."), compose)
    types = [e["type"] for e in events]
    nlq_phases = [e.get("phase") for e in events if e["type"] == "status" and e.get("agent") == "nlq"]
    reasoning = "".join(e["text"] for e in events if e["type"] == "reasoning")
    assert types[0] == "status" and types[-1] == "final" and "token" in types
    assert "grounding" in nlq_phases and "executing" in nlq_phases                 # NLQ's progress, relayed live
    assert "Plan: Count shipments by carrier." in reasoning and "SELECT carrier, COUNT(*)" in reasoning
    assert "Asked the data agent: \u201cShipments per carrier\u201d" in reasoning
    assert "Schema: ecom.shipments" in reasoning and "Query 1:" in reasoning
    assert "\u21b3 5 rows \u00d7 2 columns (carrier, shipments) in " in reasoning
    assert "Data agent done in " in reasoning and "note: counts all shipments" in reasoning
    assert "Chart: bar, 1 figure (" in reasoning and "USPS ships the most" in reasoning
    assert "Writing the answer from 5 rows of evidence" in reasoning
    f = events[-1]
    assert f["text"] == "**USPS** ships the most: 8,051 shipments." and f["kind"] == "answer"
    assert f["artifacts"]["table"]["columns"] == ["carrier", "shipments"] and len(f["artifacts"]["table"]["rows"]) == 5
    assert f["artifacts"]["table"]["rows"][0] == ["USPS", 8051]                          # real rows from PostgreSQL
    assert f["artifacts"]["charts"][0]["figures"] and f["artifact_summary"] == {"charts": 1, "tables": 1, "graphs": 0, "exports": 0}
    assert f["agents_used"] == ["supervisor", "nlq", "chart_gen"] and f["reasoning"] == reasoning
    usage = {u["agent"]: u["usage"] for u in f["usage_by_agent"]}
    assert usage["supervisor"]["total_tokens"] == 560 + 940 and usage["chart_gen"]["total_tokens"] == 1200
    assert "| USPS | 8051 |" in compose.seen[-1].content                                 # the composer saw the evidence
    src = f["sources"]                                                                   # the citation
    assert len(src) == 1 and src[0]["agent"] == "nlq" and src[0]["query_language"] == "SQL"
    assert src[0]["sources"] == ["ecom.shipments"] and src[0]["row_count"] == 5
    assert src[0]["query"].startswith("SELECT carrier") and src[0]["question"] == "Shipments per carrier"


def test_answer_path_needs_no_specialist():
    events = run(RouteDecision(action="answer", rationale="A greeting.", reply="Hi! Ask me about orders, revenue, customers…"))
    assert [e["type"] for e in events] == ["status", "reasoning", "token", "final"]
    assert events[-1]["text"].startswith("Hi!") and events[-1]["agents_used"] == ["supervisor"]


def test_clarify_path():
    events = run(RouteDecision(action="clarify", rationale="Which period?", clarifying_question="Which year?", options=["2023", "2024"]))
    f = events[-1]
    assert f["kind"] == "clarification" and f["clarifying_question"] == "Which year?" and f["options"] == ["2023", "2024"]


def test_a_failing_nlq_still_produces_an_answer(monkeypatch):
    monkeypatch.setattr(specialists.CFG.__class__, "nlq_url", property(lambda self: "http://127.0.0.1:1/"), raising=False)
    events = run(RouteDecision(action="query", questions=["x"], rationale="r"), Compose(["Sorry, the data agent was unavailable."]))
    f = events[-1]
    assert f["type"] == "final" and "data agent failed" in f["reasoning"] and f["artifacts"]["table"] is None
    assert f["sources"] == []                                                            # a failed call is not cited


def test_supervisor_a2a_wire(monkeypatch):
    SCRIPT["nlq"] = [call("execute_sql", sql=CARRIER_SQL), decide()]
    SCRIPT["chart"] = ChartDecision(chart_type="none", insight="-", figures=[])
    monkeypatch.setattr(sup_core, "chat_model", lambda streaming=False: Compose(["Done."]) if streaming else
                        Route(RouteDecision(action="query", questions=["Shipments per carrier"], rationale="Count.")))
    app = _app(SUP_MAIN, SUP_MAIN.SupervisorExecutor())
    rpc = {"jsonrpc": "2.0", "id": "1", "method": "message/stream", "params": {"message": {
        "role": "user", "messageId": "m", "contextId": "conv-wire-000000000000000000000000",
        "parts": [{"kind": "text", "text": json.dumps({"question": "which carrier?", "history": []})}]}}}

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s", timeout=60) as c:
            async with c.stream("POST", "/", json=rpc) as resp:
                return [json.loads(l[5:]) async for l in resp.aiter_lines() if l.startswith("data:")]
    frames = asyncio.run(go())
    envs = []
    for fr in frames:
        r = fr["result"]
        parts = ((r.get("status") or {}).get("message") or {}).get("parts") or (r.get("artifact") or {}).get("parts") or []
        envs += [json.loads(p["text"]) for p in parts if p.get("text")]
    kinds = [e.get("type") for e in envs]
    assert {"status", "reasoning", "token", "final"} <= set(kinds) and kinds[-1] == "final"
    assert frames[-1]["result"]["status"]["state"] == "completed"


def test_a_failed_attempt_is_visible_in_reasoning():
    """NLQ's first query fails; the reasoning shows both queries and the exact error between them."""
    SCRIPT["nlq"] = [call("execute_sql", sql="SELECT carrier, COUNT(*) AS n FROM ecom.shipments GROUP BY revenue"),
                     call("execute_sql", sql=CARRIER_SQL), decide()]
    SCRIPT["chart"] = ChartDecision(chart_type="none", insight="", figures=[])
    reasoning = "".join(e["text"] for e in run(RouteDecision(action="query", questions=["Shipments per carrier"],
                                                             rationale="r"), Compose(["ok"])) if e["type"] == "reasoning")
    assert "Query 1:" in reasoning and "Query 2:" in reasoning
    assert "\u21b3 query 1 failed: column \"revenue\" does not exist — rewriting" in reasoning
    assert reasoning.index("Query 1:") < reasoning.index("failed") < reasoning.index("Query 2:")
    assert "No chart: the chart agent judged that a chart adds nothing here" in reasoning
