"""Backend test setup: in-memory store, a fixed signing key, a fake supervisor.

The app always stores in DynamoDB; no setting selects anything else. The
tests install MemoryStore with store.use_store() — the only way it is used.

The fake supervisor returns a response in the phase-1 SupervisorResponse
shape — calls, tool_calls, results with a Cypher table and re-ranked
passages, usage, trace id — so answers.py is tested against the real
contract, not a simplified one.
"""
import os
import sys
from pathlib import Path

import pytest

os.environ.update(SESSION_SECRET="test-signing-key",
                  ADMIN_USERS="admin.user", SUPERVISOR_ARN="arn:aws:bedrock-agentcore:us-east-1:1:runtime/sup")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))   # agents' test helpers

from fastapi.testclient import TestClient   # noqa: E402

from app import supervisor   # noqa: E402
from app.main import app   # noqa: E402
from app.store import get_store, use_store   # noqa: E402
from app.store.memory import MemoryStore   # noqa: E402

use_store(MemoryStore())        # no test may reach a real table by accident

TRACE = "6ab8b8452741aafe1a2cd0731ad0c3ea"


def supervisor_response(question: str) -> dict:
    return {
        "composed_answer": f"Answer to: {question}",
        "calls": [{"agent_name": "trial_graph", "question": "Resolve IMbrave150",
                   "rationale": "resolve the trial to its document", "result_shape": "table",
                   "succeeded": True},
                  {"agent_name": "trial_search", "question": "exclusion criteria",
                   "rationale": "protocol text", "result_shape": "passages", "succeeded": True}],
        "tool_calls": [{"tool": "recall_facts", "args": {"query": "format"}, "succeeded": True,
                        "result": "1 semantic memories", "items": 1}],
        "results": {
            "trial_graph": {"result_shape": "table", "columns": ["nctId", "docId"],
                            "rows": [["NCT03434379", "nct03434379-hepatocellular-atezo-bev"]],
                            "cypher": "MATCH (t:Trial {nctId: 'NCT03434379'}) RETURN t"},
            "trial_search": {"result_shape": "passages", "stats": {"search_calls": 2},
                             "searches": [{"query": "exclusion criteria",
                                           "doc_id": "nct03434379-hepatocellular-atezo-bev",
                                           "content_type": None, "top_k": 8, "candidates": 40,
                                           "results": 8, "reranked": True, "succeeded": True}],
                             "passages": [
                                 {"chunk_id": "a", "doc_id": "d", "page": 54, "text": "low",
                                  "headings": ["4.1.2"], "rerank_score": 0.40, "origin": "search"},
                                 {"chunk_id": "b", "doc_id": "d", "page": 56, "text": "best",
                                  "headings": ["4.1.2"], "rerank_score": 0.90, "origin": "search"},
                                 {"chunk_id": "c", "doc_id": "d", "page": 55, "text": "near",
                                  "headings": [], "rerank_score": None, "origin": "neighbor"}]},
            "memory:semantic": {"result_shape": "memory", "kind": "semantic", "query": "format",
                                "items": [{"text": "Prefers tables.", "score": 0.93,
                                           "created_at": "2026-09-20T10:00:00Z"}]}},
        "decision": {"answerable": True, "entities": ["NCT03434379"], "note": "",
                     "resolved_question": "", "from_conversation": False},
        "render_target": "none",
        "usage": {"input_tokens": 12000, "cached_input_tokens": 10000, "output_tokens": 500,
                  "total_tokens": 12500, "llm_calls": 4, "model_id": "gpt-6-sol"},
        "history_turns": 0, "trace_id": TRACE}


@pytest.fixture
def asked(monkeypatch):
    """Records every supervisor call; answers with supervisor_response()."""
    calls = []

    def ask(question, conversation_id, history, username):
        calls.append({"question": question, "conversation_id": conversation_id,
                      "history": history, "username": username})
        return supervisor_response(question)
    monkeypatch.setattr(supervisor, "ask", ask)
    return calls


@pytest.fixture
def fresh_store():
    use_store(MemoryStore())
    yield get_store()
    use_store(MemoryStore())


def signed_in(username: str = "prudhvi", password: str = "correct-horse-1") -> TestClient:
    client = TestClient(app)
    reply = client.post("/api/auth/login", json={"username": username, "password": password,
                                                 "first_name": "Prudhvi", "last_name": "A"})
    assert reply.status_code == 200, reply.text
    return client
