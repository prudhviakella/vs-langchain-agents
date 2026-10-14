"""Admin console: access · usage (tokens, cost from prices, latency, by agent / user) · null cost without a
price · feedback with question + answer · KB browse (stand-in Pinecone) · DynamoDB day / month index entries."""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.update(APP_STORE="memory", SESSION_SECRET="test-key", SUPERVISOR_URL="http://unused", KEEPALIVE_S="0.2",
                  ADMIN_USERS="boss", MODEL_PRICES=json.dumps({"gpt-6-sol": {"input": 2.0, "output": 8.0}}))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import supervisor, turns  # noqa: E402
from app.main import app  # noqa: E402
from app.routers import admin  # noqa: E402
from app.settings import S  # noqa: E402
from app.store import get_store  # noqa: E402


def final(usage):
    return {"type": "final", "kind": "answer", "text": "Answer.", "reasoning": "Plan.\n", "entities": [], "sources": [],
            "artifacts": {"table": None, "charts": []}, "agents_used": ["supervisor", "nlq"], "usage_by_agent": usage}


USAGE = [{"agent": "supervisor", "usage": {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}},
         {"agent": "nlq", "usage": {"input_tokens": 18000, "output_tokens": 1000, "total_tokens": 19000}}]


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    get_store.cache_clear()
    turns.TURNS.clear()

    async def fake(payload):
        yield final(USAGE)
    monkeypatch.setattr(supervisor, "stream_turn", fake)


def client(user):
    c = TestClient(app)
    assert c.post("/api/v1/auth/login", json={"username": user, "password": "longenough1",
                                                "first_name": "F", "last_name": "L"}).status_code == 200
    return c


def ask(c, text="q"):
    with c.stream("POST", "/api/v1/chat/message", json={"text": text}) as r:
        lines = [json.loads(l[5:]) for l in r.iter_lines() if l.startswith("data:")]
    return lines[0]["conversation_id"], lines[-1]["message_id"]


def test_admins_only():
    assert client("analyst1").get("/api/v1/admin/usage").status_code == 403
    assert client("boss").get("/api/v1/admin/usage").status_code == 200


def test_usage_totals_cost_and_breakdowns():
    a, b = client("analyst1"), client("analyst2")
    ask(a); ask(a); ask(b)
    u = client("boss").get("/api/v1/admin/usage?days=7").json()
    t = u["totals"]
    assert t["interactions"] == 3 and t["users"] == 2 and t["total_tokens"] == 3 * 20100 and t["failed"] == 0
    # 3 × (19,000 in × $2/M + 1,100 out × $8/M) = 3 × 0.0468
    assert t["cost_usd"] == pytest.approx(3 * (19000 * 2 / 1e6 + 1100 * 8 / 1e6))
    assert t["avg_latency_ms"] is not None and t["p95_latency_ms"] is not None
    agents = {r["agent"]: r for r in u["by_agent"]}
    assert agents["nlq"]["calls"] == 3 and agents["nlq"]["total_tokens"] == 57000
    assert {r["user_id"]: r["interactions"] for r in u["by_user"]} == {"analyst1": 2, "analyst2": 1}
    assert len(u["by_day"]) == 7 and u["by_day"][-1]["interactions"] == 3


def test_no_price_means_null_cost_not_a_guess(monkeypatch):
    monkeypatch.delitem(S.model_prices, "gpt-6-sol")
    ask(client("analyst1"))
    u = client("boss").get("/api/v1/admin/usage").json()
    assert u["priced"] is False and u["totals"]["cost_usd"] is None and u["by_agent"][0]["cost_usd"] is None
    assert u["totals"]["total_tokens"] == 20100


def test_feedback_hub():
    a = client("analyst1")
    conv, msg = ask(a, "Which carrier is best?")
    a.post("/api/v1/feedback", json={"conversation_id": conv, "message_id": msg, "vote": "down", "comment": "wrong period"})
    boss = client("boss")
    fb = boss.get("/api/v1/admin/feedback").json()
    assert fb["dislikes"] == 1 and fb["comments"] == 1
    item = fb["items"][0]
    assert item["question"] == "Which carrier is best?" and item["answer"] == "Answer." and item["user_id"] == "analyst1"
    assert boss.get("/api/v1/admin/feedback?vote=up").json()["items"] == []


class FakeIndex:
    DATA = {"nlq-schema": {"nlq_table:ecom.orders": {"type": "table", "table": "ecom.orders", "text": "Every order"},
                           "nlq_column:ecom.orders.total_amount": {"type": "column", "table": "ecom.orders",
                                                                   "column": "total_amount", "text": "Merchandise subtotal"}},
            "nlq-examples": {"ex1": {"type": "example", "question": "Revenue by month", "query": "SELECT …"}},
            "common": {"glossary:aov": {"type": "glossary", "term": "Average Order Value", "text": "AOV"}}}

    def list(self, namespace):
        yield NS(vectors=[NS(id=i) for i in self.DATA[namespace]])

    def fetch(self, ids, namespace):
        return NS(vectors={i: NS(metadata=self.DATA[namespace][i]) for i in ids})


def test_kb_browser(monkeypatch):
    monkeypatch.setattr(admin, "_pinecone", lambda: FakeIndex())
    admin._KB.update(at=0.0, records=[])
    boss = client("boss")
    all_ = boss.get("/api/v1/admin/kb").json()
    assert all_["total"] == 4 and all_["counts"] == {"nlq-schema": 2, "nlq-examples": 1, "common": 1}
    titles = [r["title"] for r in boss.get("/api/v1/admin/kb?q=merchandise").json()["records"]]
    assert titles == ["ecom.orders.total_amount"]
    assert [r["title"] for r in boss.get("/api/v1/admin/kb?namespace=nlq-examples").json()["records"]] == ["Revenue by month"]


def test_dynamo_day_and_month_indexes():
    from moto import mock_aws
    from app.models import Message
    os.environ.update(AWS_DEFAULT_REGION="us-east-1", AWS_ACCESS_KEY_ID="t", AWS_SECRET_ACCESS_KEY="t")
    with mock_aws():
        from app.store.dynamo import DynamoStore
        st = DynamoStore("t-admin", "us-east-1")
        now = time.time_ns()
        for i, (status, age_days) in enumerate([("complete", 0), ("failed", 1), ("streaming", 0), ("complete", 40)]):
            st.put_message(Message(message_id=f"m{i}", conversation_id="c", interaction_id="i", role="assistant",
                                   status=status, created_at_ns=now - age_days * 86_400 * 10**9))
        week = st.answers_between(now - 7 * 86_400 * 10**9, now + 1)
        assert sorted(m.message_id for m in week) == ["m0", "m1"]          # finished, in range; not streaming, not old
        st.put_feedback("c", "m0", "u", "up", "good", now, {"question": "q"})
        assert st.feedback_between(now - 10**9, now + 1)[0]["question"] == "q"
