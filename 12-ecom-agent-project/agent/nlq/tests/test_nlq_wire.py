"""NLQ over the A2A wire: message/stream must deliver progress frames, then the result.

    httpx (in-process ASGI) ──JSON-RPC message/stream──► A2AStarletteApplication(NLQExecutor)
        frames: task · status-update (progress JSON) × n · artifact-update (nlq_result) · status completed
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_code"))
os.environ.setdefault("NLQ_DB_MODE", "local")

import main as nlq_main  # noqa: E402
from a2a.server.apps import A2AStarletteApplication  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore  # noqa: E402
from nlq import core, grounding  # noqa: E402
from test_nlq_agent import Fake, call, decide  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("PGHOST"), reason="needs a local PostgreSQL (PG* env)")


def frames(question: dict) -> list[dict]:
    app = A2AStarletteApplication(agent_card=nlq_main.agent_card(),
                                  http_handler=DefaultRequestHandler(nlq_main.NLQExecutor(), InMemoryTaskStore())).build()
    rpc = {"jsonrpc": "2.0", "id": "1", "method": "message/stream", "params": {"message": {
        "role": "user", "messageId": "m1", "parts": [{"kind": "text", "text": json.dumps(question)}]}}}

    async def go():
        out = []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://nlq") as client:
            async with client.stream("POST", "/", json=rpc, headers={"Accept": "text/event-stream"}) as resp:
                assert resp.headers["content-type"].startswith("text/event-stream")
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        out.append(json.loads(line[5:]))
        return out
    return asyncio.run(go())


def texts(frame: dict) -> list[str]:
    r = frame.get("result") or {}
    parts = ((r.get("status") or {}).get("message") or {}).get("parts") or (r.get("artifact") or {}).get("parts") or []
    return [p.get("text") for p in parts if p.get("text")]


def test_progress_streams_then_the_result(monkeypatch):
    monkeypatch.setattr(grounding, "retrieve", lambda q: grounding.Grounding("TABLE ecom.orders", ["t"], ["ecom.orders"]))
    script = [call("execute_sql", sql="SELECT COUNT(*) AS n FROM ecom.orders"), decide()]
    monkeypatch.setattr(core, "agent", lambda: core.build(model=Fake(messages=iter(script))))
    fs = frames({"question": "how many orders?", "history": []})
    kinds = [f["result"].get("kind") for f in fs]
    progress = [json.loads(t) for f in fs if f["result"].get("kind") == "status-update" for t in texts(f)]
    phases = [p["phase"] for p in progress]
    assert kinds[0] == "task" and "artifact-update" in kinds
    assert fs[-1]["result"]["status"]["state"] == "completed"
    assert phases[0] == "grounding" and "executing" in phases and "result" in phases
    assert any("SELECT COUNT(*)" in p.get("sql", "") for p in progress)          # the SQL rides the stream
    result = json.loads(next(t for f in fs if f["result"].get("kind") == "artifact-update" for t in texts(f)))
    assert result["rows"] == [[50000]] and result["result_shape"] == "table"


def test_a_failure_fails_the_task_with_the_error(monkeypatch):
    monkeypatch.setattr(grounding, "retrieve", lambda q: (_ for _ in ()).throw(RuntimeError("pinecone down")))
    monkeypatch.setattr(core, "agent", lambda: core.build(model=Fake(messages=iter([decide()]))))
    fs = frames({"question": "x"})
    last = fs[-1]["result"]
    assert last["status"]["state"] == "failed" and "pinecone down" in texts(fs[-1])[0]
