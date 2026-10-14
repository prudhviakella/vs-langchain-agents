"""The backend over real HTTP (FastAPI TestClient), memory store, a stand-in supervisor stream.

    sign-in · a full turn (event order, one chart artifact per figure, saved BEFORE done, the reloaded
    message carries summary · reasoning · citation · table · charts) · the fallback chart · failure ·
    ownership · resume replay · keepalive · feedback
"""
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.update(APP_STORE="memory", SESSION_SECRET="test-key", SUPERVISOR_URL="http://unused", KEEPALIVE_S="0.2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import supervisor, turns  # noqa: E402
from app.main import app  # noqa: E402
from app.store import get_store  # noqa: E402

TABLE = {"columns": ["carrier", "on_time_pct"], "rows": [["OnTrac", 41.6], ["FedEx", 40.5], ["UPS", 39.8]],
         "row_count": 3, "truncated": False, "sql": "SELECT …"}
FIG = lambda name: {"data": [{"type": "bar", "x": ["OnTrac", "FedEx", "UPS"], "y": [41.6, 40.5, 39.8], "name": name}],
                    "layout": {"title": {"text": name}}}


def final(**over):
    f = {"type": "final", "kind": "answer", "text": "", "reasoning": "Plan: compare carriers.\nSQL (nlq): …\n",
         "trace_id": "t1", "entities": [{"eid": "carrier:OnTrac", "type": "carrier", "name": "OnTrac"}],
         "sources": [{"agent": "nlq", "query": "SELECT …", "sources": ["ecom.shipments"], "row_count": 3}],
         "artifacts": {"table": TABLE, "charts": [{"chart_type": "bar", "insight": "OnTrac leads",
                                                   "figures": [FIG("one"), FIG("two")]}]},
         "agents_used": ["supervisor", "nlq", "chart_gen"], "usage_by_agent": [], "options": []}
    f.update(over)
    return f


def script(*envs, delay=0.0, fail=None):
    async def fake(payload):
        for e in envs:
            if delay:
                await asyncio.sleep(delay)
            yield e
        if fail:
            raise supervisor.SupervisorError(fail)
    return fake


@pytest.fixture(autouse=True)
def fresh():
    get_store.cache_clear()
    turns.TURNS.clear()
    yield


def client(user="prudhvi"):
    c = TestClient(app)
    r = c.post("/api/v1/auth/login", json={"username": user, "password": "longenough1", "first_name": "P", "last_name": "A"})
    assert r.status_code == 200
    return c


def events(resp) -> list[tuple[str, dict]]:
    out, ev = [], None
    for line in resp.iter_lines():
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            out.append((ev, json.loads(line[5:])))
    return out


def ask(c, text, conversation_id=None):
    with c.stream("POST", "/api/v1/chat/message", json={"text": text, "conversation_id": conversation_id}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        return events(r)


def test_sign_in_flow():
    c = TestClient(app)
    r = c.post("/api/v1/auth/login", json={"username": "newbie", "password": "longenough1"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "new_user"
    r = c.post("/api/v1/auth/login", json={"username": "newbie", "password": "longenough1", "first_name": "N", "last_name": "B"})
    assert r.status_code == 200 and "ecom_session" in r.cookies and r.json()["short_name"] == "NB"
    assert c.get("/api/v1/me").json()["user_id"] == "newbie"
    assert TestClient(app).post("/api/v1/auth/login", json={"username": "newbie", "password": "wrongpass1"}).status_code == 401
    assert TestClient(app).get("/api/v1/me").status_code == 401


def test_a_full_turn(monkeypatch):
    monkeypatch.setattr(supervisor, "stream_turn", script(
        {"type": "status", "agent": "supervisor", "phase": "planning"},
        {"type": "reasoning", "text": "Plan: compare carriers.\n"},
        {"type": "status", "agent": "nlq", "phase": "executing", "detail": "running the query"},
        {"type": "token", "text": "**OnTrac** "}, {"type": "token", "text": "leads at 41.6%."}, final()))
    c = client()
    ev = ask(c, "Which carrier is most on time?")
    kinds = [k for k, _ in ev]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert kinds.index("token") < kinds.index("artifact") < kinds.index("done")
    arts = [d for k, d in ev if k == "artifact"]
    assert [a["kind"] for a in arts] == ["table", "chart", "chart"]                     # one artifact per figure
    assert arts[1]["data"]["spec"]["data"][0]["y"] == [41.6, 40.5, 39.8]
    conv_id = ev[0][1]["conversation_id"]
    msgs = c.get(f"/api/v1/conversations/{conv_id}/messages").json()["items"]
    a = msgs[-1]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and a["status"] == "complete"
    assert a["text"] == "**OnTrac** leads at 41.6%."                                     # summary
    assert a["reasoning"].startswith("Plan:")                                            # reasoning
    assert a["sources"][0]["sources"] == ["ecom.shipments"]                              # citation
    assert a["artifacts"]["table"]["rows"][0] == ["OnTrac", 41.6]                        # table
    assert len(a["artifacts"]["charts"]) == 2 and a["artifacts"]["charts"][0]["kind"] == "chart"   # charts
    conv = c.get("/api/v1/conversations").json()["conversations"][0]
    assert conv["conversation_id"] == conv_id and conv["title"] == "Which carrier is most on time?"


def test_saved_before_done(monkeypatch):
    seen = {}

    async def fake(payload):
        yield final()
    monkeypatch.setattr(supervisor, "stream_turn", fake)
    original = turns.Turn.emit

    def spy(self, event, data):
        if event == "done":
            m = get_store().get_message(self.conversation_id, data["message_id"])
            seen["status_at_done"] = m.status
        original(self, event, data)
    monkeypatch.setattr(turns.Turn, "emit", spy)
    ask(client(), "q")
    assert seen["status_at_done"] == "complete"


def test_fallback_chart_on_an_explicit_request(monkeypatch):
    month_table = {"columns": ["month", "revenue"], "rows": [["2024-01", 5.1], ["2024-02", 4.8], ["2024-03", 5.6]],
                   "row_count": 3, "truncated": False}
    monkeypatch.setattr(supervisor, "stream_turn", script(final(artifacts={"table": month_table, "charts": []})))
    ev = ask(client(), "Plot revenue by month")
    charts = [d["data"] for k, d in ev if k == "artifact" and d["kind"] == "chart"]
    assert len(charts) == 1 and charts[0]["source"] == "fallback" and charts[0]["chart_type"] == "line"
    monkeypatch.setattr(supervisor, "stream_turn", script(final(artifacts={"table": month_table, "charts": []})))
    ev = ask(client(), "What was revenue by month?")                                  # no explicit ask: no fallback
    assert not [d for k, d in ev if k == "artifact" and d["kind"] == "chart"]


def test_failure_is_saved_and_reported(monkeypatch):
    monkeypatch.setattr(supervisor, "stream_turn", script({"type": "token", "text": "Partial"}, fail="nlq timed out"))
    c = client()
    ev = ask(c, "q")
    assert ev[-1][0] == "error" and "nlq timed out" in ev[-1][1]["detail"]
    a = c.get(f"/api/v1/conversations/{ev[0][1]['conversation_id']}/messages").json()["items"][-1]
    assert a["status"] == "failed" and a["text"] == "Partial" and "nlq timed out" in a["reasoning"]


def test_ownership(monkeypatch):
    monkeypatch.setattr(supervisor, "stream_turn", script(final()))
    ev = ask(client("owner1"), "q")
    conv_id, inter = ev[0][1]["conversation_id"], ev[0][1]["interaction_id"]
    other = client("intruder")
    for path in (f"/api/v1/conversations/{conv_id}", f"/api/v1/conversations/{conv_id}/messages",
                 f"/api/v1/chat/stream/{inter}"):
        assert other.get(path).status_code == 404
    r = other.post("/api/v1/chat/message", json={"text": "x", "conversation_id": conv_id})
    assert r.status_code == 404


def test_resume_replays_the_whole_turn(monkeypatch):
    monkeypatch.setattr(supervisor, "stream_turn", script({"type": "reasoning", "text": "Plan.\n"},
                                                          {"type": "token", "text": "A"}, final()))
    c = client()
    first = ask(c, "q")
    with c.stream("GET", f"/api/v1/chat/stream/{first[0][1]['interaction_id']}") as r:
        again = events(r)
    assert [k for k, _ in again] == [k for k, _ in first]


def test_keepalive_during_a_slow_turn(monkeypatch):
    monkeypatch.setattr(supervisor, "stream_turn", script({"type": "token", "text": "slow"}, final(), delay=0.5))
    c = client()
    with c.stream("POST", "/api/v1/chat/message", json={"text": "q"}) as r:
        lines = list(r.iter_lines())
    assert ": ping" in lines


def test_feedback_is_joined_into_messages(monkeypatch):
    monkeypatch.setattr(supervisor, "stream_turn", script(final()))
    c = client()
    ev = ask(c, "q")
    conv_id, msg_id = ev[0][1]["conversation_id"], ev[-1][1]["message_id"]
    assert c.post("/api/v1/feedback", json={"conversation_id": conv_id, "message_id": msg_id, "vote": "up",
                                            "comment": "spot on"}).status_code == 204
    a = c.get(f"/api/v1/conversations/{conv_id}/messages").json()["items"][-1]
    assert a["feedback"]["likes"] == 1 and a["feedback"]["my_vote"] == "up" and a["feedback"]["comments"][0]["comment"] == "spot on"
