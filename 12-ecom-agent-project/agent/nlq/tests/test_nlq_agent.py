"""NLQ agent end to end — real loop, real middleware, real PostgreSQL (local mode).

    scripted model (Fake) ─► create_agent + Budget/Guard/Grounding middleware ─► tools
                                                                                 └─► sql_exec ─► local Postgres
                                                                                     (as ecom_reader)
    stand-ins: the MODEL (scripted tool calls) and Pinecone (grounding is pinned)

    happy path · guard rejection · SQL error then repair · repair budget · scope after a
    result · a LIMIT 1 peek is not a result · no query at all · progress events

    PGHOST=localhost PGUSER=ecom_reader PGPASSWORD=… PGDATABASE=ecom PGSSLMODE=disable \
      python -m pytest tests -q
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_code"))
os.environ.setdefault("NLQ_DB_MODE", "local")

from nlq import core, grounding  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("PGHOST"), reason="needs a local PostgreSQL (PG* env)")


class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(tool: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": f"{tool}-{id(args)}"}])


def decide(**kw) -> AIMessage:
    return call("ModelDecision", **{"entities": [], "answerable": True, "note": "", "unmet_parts": [], **kw})


@pytest.fixture(autouse=True)
def pinned_grounding(monkeypatch):
    monkeypatch.setattr(grounding, "retrieve", lambda q: grounding.Grounding(
        context_block="TABLE ecom.orders — every order", record_ids=["nlq_table:ecom.orders"], tables=["ecom.orders"]))


def run(script, question="q"):
    graph = core.build(model=Fake(messages=iter(script)))
    events = []

    async def go():
        async for ev in core.answer_stream(question, graph=graph):
            events.append(ev)
    asyncio.run(go())
    return events, events[-1]["result"]


REVENUE = ("SELECT date_trunc('year', order_date) AS year, ROUND(SUM(total_amount)::numeric, 2) AS revenue "
           "FROM ecom.orders GROUP BY 1 ORDER BY 1")


def test_happy_path_rows_come_from_state_not_the_model():
    events, r = run([call("execute_sql", sql=REVENUE), decide(note="revenue = merchandise total")])
    assert r["result_shape"] == "table" and r["row_count"] == 2
    assert r["columns"] == ["year", "revenue"] and r["rows"][0][1] == 60461311.59   # = SUM of the cleaned CSV, exact decimals
    assert r["sql"].endswith("LIMIT 1001")                       # the guarded SQL is what ran
    phases = [e["phase"] for e in events]
    assert phases[:2] == ["grounding", "grounding"] and "executing" in phases and phases[-1] == "done"
    assert any(e.get("sql", "").startswith("SELECT date_trunc") for e in events if e["phase"] == "executing")
    assert r["grounding_record_ids"] == ["nlq_table:ecom.orders"]


def test_guard_rejects_a_write_then_the_model_recovers():
    _, r = run([call("execute_sql", sql="DELETE FROM ecom.orders"),
                call("execute_sql", sql="SELECT COUNT(*) AS n FROM ecom.orders"), decide()])
    assert r["rows"] == [[50000]] and r["repair_count"] == 1


def test_sql_error_is_shown_to_the_model_and_repaired():
    _, r = run([call("execute_sql", sql="SELECT revenue FROM ecom.orders"),
                call("execute_sql", sql="SELECT SUM(total_amount) AS revenue FROM ecom.orders"), decide()])
    assert r["result_shape"] == "table" and r["repair_count"] == 1


def test_repair_budget_stops_the_loop():
    bad = [call("execute_sql", sql=f"SELECT nope{i} FROM ecom.orders") for i in range(4)]
    _, r = run(bad + [decide(answerable=False, note="could not find the column")])
    assert r["result_shape"] == "unanswerable" and r["repair_count"] == 3   # the 4th was refused


def test_scope_after_a_complete_result():
    q = "SELECT COUNT(*) AS n FROM ecom.orders"
    _, r = run([call("execute_sql", sql=q), call("execute_sql", sql=q + " WHERE is_gift"),
                call("execute_sql", sql=q + " WHERE NOT is_gift"), decide()])
    # 1 extra query allowed after a result: the 2nd (gift orders) ran, the 3rd was refused
    assert 0 < r["rows"][0][0] < 50000 and "is_gift" in r["sql"] and "NOT" not in r["sql"]


def test_a_limit_1_peek_does_not_count_as_a_result():
    peek = "SELECT order_id FROM ecom.orders LIMIT 1"
    _, r = run([call("execute_sql", sql=peek), call("execute_sql", sql="SELECT COUNT(*) AS n FROM ecom.orders"),
                call("execute_sql", sql="SELECT COUNT(*) AS n FROM ecom.customers"), decide()])
    assert r["rows"] == [[10000]]      # peek, then 2 real queries: the second was allowed


def test_no_query_means_not_executed():
    _, r = run([decide(note="I know the answer")])
    assert r["result_shape"] == "not_executed" and r["rows"] == []


def test_an_unbounded_result_is_reported_as_truncated():
    """Both guards inject LIMIT row_cap + 1; the extra row proves more exist."""
    _, r = run([call("execute_sql", sql="SELECT order_item_id FROM ecom.order_items"), decide()])
    assert r["row_count"] == 1000 and r["truncated"] is True
    assert "capped at 1000 rows" in r["result_note"]
