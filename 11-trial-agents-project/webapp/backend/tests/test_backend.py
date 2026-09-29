"""The backend, end to end through FastAPI, with a fake supervisor.

    AUTH           new username -> names -> created; wrong password 401;
                   cookie is HttpOnly; forged or expired tokens rejected
    CONVERSATIONS  create, list newest first, search title AND questions,
                   rename, delete (messages + feedback go too), ownership 404
    CHAT           events in order; the answer is saved with agents + query,
                   tools, citations best-first, cost, trace link
    HISTORY        the last 20 messages BEFORE the question; failed answers
                   left out; the username reaches the supervisor
    FAILURES       a supervisor error is streamed and saved, not raised
    DISCONNECT     a browser leaving mid-turn still gets its answer saved
    FEEDBACK       answers only, mine only, resubmit replaces, shown on messages
    AGENTOPS       own rows for users, everyone's for admins
    DYNAMODB       every call passes botocore's own validator
"""
import asyncio
import json
import time

import jwt
import pytest

from app import answers, session, settings, supervisor
from app.store.memory import MemoryStore
from conftest import TRACE, signed_in, supervisor_response


def events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def ask(client, text, conversation_id=None):
    reply = client.post("/api/chat", json={"text": text, "conversation_id": conversation_id})
    assert reply.status_code == 200
    return events(reply.text)


# ── AUTH ─────────────────────────────────────────────────────────────────
def test_new_username_asks_for_names_then_signs_in(fresh_store):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    first = client.post("/api/auth/login", json={"username": "New.User", "password": "longenough1"})
    assert first.status_code == 404 and first.json()["detail"]["code"] == "new_user"

    made = client.post("/api/auth/login", json={"username": "New.User", "password": "longenough1",
                                                "first_name": "New", "last_name": "User"})
    assert made.status_code == 200 and made.json()["username"] == "new.user"
    cookie = made.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert client.get("/api/auth/me").json()["first_name"] == "New"

    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401


def test_existing_user_needs_the_right_password(fresh_store):
    signed_in("prudhvi", "correct-horse-1")
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    assert client.post("/api/auth/login", json={"username": "prudhvi",
                                                "password": "wrong-password"}).status_code == 401
    ok = client.post("/api/auth/login", json={"username": "prudhvi", "password": "correct-horse-1"})
    assert ok.status_code == 200 and ok.json()["first_name"] == "Prudhvi"
    assert fresh_store.get_user("prudhvi")["password_hash"].startswith("pbkdf2_sha256$600000$")


def test_forged_and_expired_tokens_are_rejected(fresh_store):
    now = int(time.time())
    forged = jwt.encode({"sub": "admin.user", "iss": "trial-agents-app",
                         "aud": "trial-agents-session", "exp": now + 60}, "not-the-key",
                        algorithm="HS256")
    expired = jwt.encode({"sub": "prudhvi", "iss": "trial-agents-app",
                          "aud": "trial-agents-session", "exp": now - 1},
                         settings.SESSION_SECRET, algorithm="HS256")
    assert session.verify(forged) is None and session.verify(expired) is None
    assert session.verify(session.mint("prudhvi")[0]) == "prudhvi"


# ── CONVERSATIONS ────────────────────────────────────────────────────────
def test_create_list_search_rename_delete(fresh_store, asked):
    client = signed_in()
    a = client.post("/api/conversations", json={"title": "Oncology"}).json()
    ask(client, "Which trials does Novo Nordisk sponsor?", a["conversation_id"])
    b = client.post("/api/conversations", json={"title": "Glaucoma"}).json()

    listed = client.get("/api/conversations").json()
    assert [c["title"] for c in listed] == ["Glaucoma", "Oncology"]          # newest first
    assert [c["title"] for c in client.get("/api/conversations?q=NOVO").json()] == ["Oncology"]

    client.patch(f"/api/conversations/{b['conversation_id']}", json={"title": "Eye trials"})
    assert client.get("/api/conversations?q=eye").json()[0]["title"] == "Eye trials"

    assert client.delete(f"/api/conversations/{a['conversation_id']}").status_code == 204
    assert [c["title"] for c in client.get("/api/conversations").json()] == ["Eye trials"]
    assert fresh_store.list_messages(a["conversation_id"]) == []


def test_another_users_conversation_is_not_found(fresh_store, asked):
    owner = signed_in("owner.one")
    cid = owner.post("/api/conversations", json={"title": "mine"}).json()["conversation_id"]
    intruder = signed_in("intruder")
    assert intruder.get(f"/api/conversations/{cid}/messages").status_code == 404
    assert intruder.delete(f"/api/conversations/{cid}").status_code == 404
    assert intruder.post("/api/chat", json={"text": "hi", "conversation_id": cid}).status_code == 404
    assert asked == []


# ── CHAT ─────────────────────────────────────────────────────────────────
def test_a_turn_streams_and_saves_the_full_answer(fresh_store, asked):
    client = signed_in()
    stream = ask(client, "What are the exclusion criteria of IMbrave150?")
    assert [name for name, _ in stream] == ["conversation", "answer"]
    cid = stream[0][1]["conversation_id"]

    answer = client.get(f"/api/conversations/{cid}/messages").json()[1]
    d = answer["details"]
    assert answer["role"] == "assistant" and answer["status"] == "complete"
    assert d["agents"][0]["query"].startswith("MATCH")                  # the Cypher that ran
    assert d["agents"][1]["stats"] == {"search_calls": 2}
    assert [c["snippet"] for c in d["citations"]] == ["best", "low", "near"]
    assert d["tools"][0]["tool"] == "recall_facts" and d["memory"][0]["text"] == "Prefers tables."
    assert d["trace_url"].endswith(f"#xray:traces/1-{TRACE[:8]}-{TRACE[8:]}")
    # 2,000 fresh input at $2 + 10,000 cached at $2 (upper bound) + 500 output at $10
    assert d["cost_usd"] == pytest.approx((2000 * 2 + 10000 * 2 + 500 * 10) / 1e6)


def test_history_is_the_last_ten_interactions_before_the_question(fresh_store, asked):
    client = signed_in()
    cid = ask(client, "q0")[0][1]["conversation_id"]
    for i in range(1, 12):
        ask(client, f"q{i}", cid)
    last = asked[-1]
    assert len(last["history"]) == 20                                  # 10 interactions
    assert last["history"][-1] == {"role": "assistant", "text": "Answer to: q10"}
    assert all(m["text"] != "q11" for m in last["history"])            # not the question itself
    assert last["username"] == "prudhvi" and last["conversation_id"] == cid


def test_a_supervisor_failure_is_streamed_saved_and_kept_out_of_history(fresh_store, monkeypatch):
    seen = []

    def failing(question, conversation_id, history, username):
        seen.append(history)
        raise supervisor.SupervisorError("AccessDeniedException: signature mismatch")
    monkeypatch.setattr(supervisor, "ask", failing)
    client = signed_in()
    stream = ask(client, "first")
    assert stream[-1][0] == "error" and "signature mismatch" in stream[-1][1]["message"]["text"]
    cid = stream[0][1]["conversation_id"]
    ask(client, "second", cid)
    assert seen[-1] == [{"role": "user", "text": "first"}]            # the error text left out
    assert client.get("/api/agentops/interactions").json()[0]["status"] == "error"


def test_an_answer_is_saved_even_if_the_browser_leaves(fresh_store, monkeypatch):
    """The stream generator is closed after its first event — as a
    disconnecting client does. The turn task must still save the answer."""
    from app.routers import chat as chat_router
    from app.models import ChatRequest

    def slow(question, conversation_id, history, username):
        time.sleep(0.3)
        return supervisor_response(question)
    monkeypatch.setattr(supervisor, "ask", slow)

    async def leave_early():
        response = await chat_router.chat(ChatRequest(text="leave early"), "prudhvi", fresh_store)
        stream = response.body_iterator
        await stream.__anext__()                      # the "conversation" event
        await stream.aclose()                         # the browser goes away
        await asyncio.gather(*list(chat_router._RUNNING))
    asyncio.run(leave_early())
    [conv] = fresh_store.list_conversations("prudhvi")
    saved = fresh_store.list_messages(conv["conversation_id"])
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[1]["text"] == "Answer to: leave early"


# ── FEEDBACK ─────────────────────────────────────────────────────────────
def test_feedback_on_my_answers_replaces_and_shows(fresh_store, asked):
    client = signed_in()
    stream = ask(client, "exclusion criteria?")
    cid, answer_id = stream[0][1]["conversation_id"], stream[-1][1]["message"]["message_id"]
    question_id = stream[0][1]["user_message"]["message_id"]

    assert client.post("/api/feedback", json={"conversation_id": cid, "message_id": question_id,
                                              "rating": "up"}).status_code == 404
    client.post("/api/feedback", json={"conversation_id": cid, "message_id": answer_id,
                                       "rating": "down", "comment": "missed items"})
    client.post("/api/feedback", json={"conversation_id": cid, "message_id": answer_id,
                                       "rating": "up", "comment": "complete now"})
    mine = client.get("/api/feedback").json()
    assert len(mine) == 1 and mine[0]["rating"] == "up"
    assert mine[0]["question"] == "exclusion criteria?"
    assert client.get(f"/api/conversations/{cid}/messages").json()[1]["feedback"]["rating"] == "up"

    other = signed_in("someone.else")
    assert other.post("/api/feedback", json={"conversation_id": cid, "message_id": answer_id,
                                             "rating": "down"}).status_code == 404


# ── AGENTOPS ─────────────────────────────────────────────────────────────
def test_agentops_scope_mine_versus_everyone(fresh_store, asked):
    ask(signed_in("analyst.one"), "q from one")
    ask(signed_in("analyst.two"), "q from two")
    one = signed_in("analyst.one")
    admin = signed_in("admin.user")

    mine = one.get("/api/agentops/summary").json()
    assert mine["scope"] == "mine" and mine["interactions"] == 1
    everyone = admin.get("/api/agentops/summary").json()
    assert everyone["scope"] == "everyone" and everyone["interactions"] == 2
    assert everyone["agents"] == {"trial_graph": 2, "trial_search": 2}
    assert everyone["tools"] == {"recall_facts": 2}
    assert everyone["total_tokens"] == 25000
    rows = admin.get("/api/agentops/interactions").json()
    assert {r["username"] for r in rows} == {"analyst.one", "analyst.two"}
    assert rows[0]["trace_url"]


# ── SUPERVISOR CALL SHAPE ────────────────────────────────────────────────
def test_envelope_carries_history_and_user_and_parse_reads_both_shapes():
    body = json.loads(supervisor.envelope("q", "c" * 32, [{"role": "user", "text": "x"}], "prudhvi"))
    meta = body["params"]["message"]["metadata"]
    assert meta == {"history": [{"role": "user", "text": "x"}], "user_id": "prudhvi"}
    assert body["params"]["message"]["contextId"] == "c" * 32
    assert 33 <= len(supervisor.session_id("abc")) <= 256

    reply = json.dumps({"composed_answer": "ok"})
    message = {"result": {"parts": [{"kind": "text", "text": reply}]}}
    task = {"result": {"artifacts": [{"parts": [{"kind": "text", "text": reply}]}]}}
    assert supervisor.parse(json.dumps(message).encode())["composed_answer"] == "ok"
    assert supervisor.parse(json.dumps(task).encode())["composed_answer"] == "ok"
    with pytest.raises(supervisor.SupervisorError):
        supervisor.parse(json.dumps({"error": {"code": -1}}).encode())


def test_answer_details_stay_under_the_dynamodb_item_limit():
    huge = supervisor_response("q")
    huge["render_target"] = "chart"
    huge["results"]["trial_graph"]["rows"] = [["x" * 500, "y" * 500] for _ in range(2000)]
    huge["results"]["trial_search"]["passages"] = [
        {"chunk_id": str(i), "doc_id": "d", "text": "t" * 5000, "rerank_score": 0.5}
        for i in range(100)]
    details = answers.details(huge, 1000)
    assert len(json.dumps(details)) <= answers.MAX_DETAILS_BYTES
    assert len(details["citations"]) == answers.MAX_CITATIONS


# ── DYNAMODB: every call through botocore's own validator ───────────────
def test_dynamodb_store_calls_pass_botocore_validation():
    from test_payloads import Recording
    from app.store.dynamodb import DynamoStore, _item

    meta = {"PK": "CONV#c1", "SK": "META", "conversation_id": "c1", "username": "prudhvi",
            "title": "t", "created_at": "2026-09-27T00:00:00.000Z",
            "updated_at": "2026-09-27T00:00:00.000Z", "message_count": 0, "search_text": ""}
    client = Recording("dynamodb", {
        "get_item": {"Item": _item(meta)},
        "query": {"Items": [_item(meta)]},
        "batch_write_item": {"UnprocessedItems": {}}})
    store = DynamoStore(client=client, table="trial-app")
    assert store.create_user({"username": "prudhvi", "first_name": "P", "last_name": "A",
                              "password_hash": "h", "created_at": "t"})
    store.create_conversation("prudhvi", "t")
    store.add_message({"message_id": "m1", "conversation_id": "c1", "role": "assistant",
                       "text": "a", "created_at": "2026-09-27T00:00:01.000Z", "status": "complete",
                       "details": {"cost_usd": 0.0123}, "interaction": {"username": "prudhvi"}})
    store.put_feedback({"conversation_id": "c1", "message_id": "m1", "username": "prudhvi",
                        "rating": "up", "comment": "", "question": "q", "answer_snippet": "a",
                        "created_at": "2026-09-27T00:00:02.000Z"})
    store.list_conversations("prudhvi")
    store.feedback_by_user("prudhvi")
    store.interactions_since("2026-09-01T00:00:00.000Z")
    store.delete_conversation("c1")
    assert {"put_item", "update_item", "query", "batch_write_item"} <= set(client.calls)


def test_memory_store_matches_the_interface():
    from app.store.base import Store
    assert isinstance(MemoryStore(), Store)


# ── history reads only what it sends; searches reach the UI ──────────────
def test_history_read_is_limited_projected_and_oldest_first():
    """One Query, newest first, stopping at the limit, projecting only role,
    text and status — and passing botocore's validator, which rejects the
    reserved words role/text/status unless they go through #names."""
    from test_payloads import Recording
    from app.store.dynamodb import DynamoStore, _item
    seen = {}

    def query(**kwargs):
        seen.update(kwargs)
        return {"Items": [_item({"role": "assistant", "text": "a2", "status": "complete"}),
                          _item({"role": "user", "text": "q2", "status": "complete"})]}
    client = Recording("dynamodb", {"query": query})
    got = DynamoStore(client=client, table="trial-app").recent_messages("c1", 30)

    assert [m["text"] for m in got] == ["q2", "a2"]                   # oldest first
    assert (seen["Limit"], seen["ScanIndexForward"]) == (30, False)
    assert seen["ProjectionExpression"] == "#r, #t, #s"
    assert seen["ExpressionAttributeNames"] == {"#r": "role", "#t": "text", "#s": "status"}
    assert client.calls == ["query"]


def test_history_skips_a_failed_answer_and_still_sends_twenty(fresh_store, asked, monkeypatch):
    client = signed_in()
    cid = ask(client, "q0")[0][1]["conversation_id"]
    for i in range(1, 11):
        ask(client, f"q{i}", cid)
    ok = supervisor.ask
    monkeypatch.setattr(supervisor, "ask", lambda *a: (_ for _ in ()).throw(supervisor.SupervisorError("down")))
    ask(client, "q11", cid)                                           # this answer fails
    monkeypatch.setattr(supervisor, "ask", ok)
    ask(client, "q12", cid)
    history = asked[-1]["history"]
    assert len(history) == 20
    assert all("could not answer" not in m["text"] for m in history)
    assert history[-1] == {"role": "user", "text": "q11"}             # the question stays


def test_each_search_reaches_the_answer_details(fresh_store, asked):
    client = signed_in()
    stream = ask(client, "exclusion criteria of IMbrave150?")
    search_step = stream[-1][1]["message"]["details"]["agents"][1]
    assert search_step["agent"] == "trial_search"
    assert search_step["searches"][0]["query"] == "exclusion criteria"
    assert search_step["searches"][0]["reranked"] is True
