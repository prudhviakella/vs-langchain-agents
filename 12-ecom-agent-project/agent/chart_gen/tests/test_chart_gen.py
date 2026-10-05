"""chart_gen: what is enforced in CODE, and the A2A wire.

    sanitize (XSS scrubbed, <br> kept, unknown trace dropped) · too-thin figure dropped -> none ·
    parse failure -> {"error"} · row cap stated · chart_type/figures agree · wire: progress, result ·
    a malformed request still COMPLETES with {"error"}
"""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_code"))

import main as chart_main  # noqa: E402
from a2a.server.apps import A2AStarletteApplication  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore  # noqa: E402
from chart_gen import core  # noqa: E402
from chart_gen.schemas import ChartDecision  # noqa: E402


class StubModel:
    """with_structured_output(...).ainvoke -> {parsed, raw, parsing_error}, like ChatOpenAI's include_raw=True."""
    def __init__(self, decision=None, error=None):
        self.decision, self.error = decision, error

    def with_structured_output(self, schema, method, include_raw):
        assert method == "function_calling" and include_raw is True
        return self

    async def ainvoke(self, messages):
        self.seen = messages
        return {"parsed": self.decision, "parsing_error": self.error,
                "raw": NS(usage_metadata={"input_tokens": 900, "output_tokens": 300, "total_tokens": 1200})}


COLS, ROWS = ["carrier", "on_time_pct"], [["UPS", 41.9], ["USPS", 41.6], ["FedEx", 41.2], ["DHL", 40.9]]


def run(decision=None, error=None, rows=ROWS):
    model = StubModel(decision, error)
    return asyncio.run(core.generate_chart("which carrier is most on time?", COLS, rows, model=model)), model


def test_figures_are_sanitized_in_code():
    d = ChartDecision(chart_type="bar", insight="UPS leads", figures=[{
        "data": [{"type": "bar", "x": ["UPS<img src=x onerror=alert(1)>", "USPS", "FedEx"], "y": [41.9, 41.6, 41.2]},
                 {"type": "evilwidget", "x": [1, 2, 3]}],
        "layout": {"title": {"text": "On time<br>by carrier<script>steal()</script>"}}}])
    r, _ = run(d)
    fig = r["figures"][0]
    assert [t["type"] for t in fig["data"]] == ["bar"]                    # unknown trace type dropped
    assert fig["data"][0]["x"][0] == "UPS"                                  # handler tag gone
    # the <script> TAGS are removed; their inner text stays as inert plain text; <br> is kept
    assert fig["layout"]["title"]["text"] == "On time<br>by carriersteal()"
    assert "<script" not in json.dumps(r) and r["usage"]["total_tokens"] == 1200


def test_a_two_point_line_is_dropped_and_the_type_follows():
    d = ChartDecision(chart_type="line", insight="up", figures=[{
        "data": [{"type": "scatter", "mode": "lines", "x": ["2023", "2024"], "y": [60.4, 61.4]}], "layout": {}}])
    r, _ = run(d)
    assert r["figures"] == [] and r["chart_type"] == "none"


def test_a_parse_failure_returns_an_error_not_an_exception():
    r, _ = run(None, error="bad json")
    assert "structured output failed" in r["error"]


def test_the_row_cap_is_stated_to_the_model():
    many = [[f"p{i}", i] for i in range(450)]
    _, model = run(ChartDecision(chart_type="none", insight="-", figures=[]), rows=many)
    text = model.seen[1].content
    assert "| p299 | 299 |" in text and "| p300 |" not in text and "first 300 of 450 rows" in text


def test_chart_type_and_figures_must_agree():
    with pytest.raises(ValidationError):
        ChartDecision(chart_type="bar", insight="x", figures=[])
    with pytest.raises(ValidationError):
        ChartDecision(chart_type="none", insight="x", figures=[{"data": [{"type": "bar"}], "layout": {}}])


def frames(text: str) -> list[dict]:
    app = A2AStarletteApplication(agent_card=chart_main.agent_card(), http_handler=DefaultRequestHandler(
        chart_main.ChartExecutor(), InMemoryTaskStore())).build()
    rpc = {"jsonrpc": "2.0", "id": "1", "method": "message/stream", "params": {"message": {
        "role": "user", "messageId": "m", "parts": [{"kind": "text", "text": text}]}}}

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://c") as c:
            async with c.stream("POST", "/", json=rpc) as resp:
                return [json.loads(l[5:]) async for l in resp.aiter_lines() if l.startswith("data:")]
    return asyncio.run(go())


def _texts(f):
    r = f["result"]
    parts = ((r.get("status") or {}).get("message") or {}).get("parts") or (r.get("artifact") or {}).get("parts") or []
    return [p["text"] for p in parts if p.get("text")]


def test_wire_progress_then_result(monkeypatch):
    d = ChartDecision(chart_type="bar", insight="UPS leads", figures=[{
        "data": [{"type": "bar", "x": ["UPS", "USPS", "FedEx"], "y": [41.9, 41.6, 41.2]}], "layout": {}}])
    monkeypatch.setattr(core, "chart_model", lambda: StubModel(d))
    fs = frames(json.dumps({"question": "q", "columns": COLS, "rows": ROWS}))
    kinds = [f["result"].get("kind") for f in fs]
    assert "status-update" in kinds and "artifact-update" in kinds and fs[-1]["result"]["status"]["state"] == "completed"
    progress = json.loads(_texts(next(f for f in fs if f["result"].get("kind") == "status-update"))[0])
    assert progress["agent"] == "chart_gen" and progress["phase"] == "charting"
    result = json.loads(_texts(next(f for f in fs if f["result"].get("kind") == "artifact-update"))[0])
    assert result["chart_type"] == "bar" and len(result["figures"]) == 1


def test_a_malformed_request_still_completes_with_an_error():
    fs = frames("not json at all")
    assert fs[-1]["result"]["status"]["state"] == "completed"
    result = json.loads(_texts(next(f for f in fs if f["result"].get("kind") == "artifact-update"))[0])
    assert "error" in result
