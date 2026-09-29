"""Cohere re-ranking in trial_search's semantic_search.

    RECALL     Pinecone is asked for the pool (40), not top_k
    PRECISION  Cohere's order and scores are applied; top_k kept
    FALLBACK   placeholder key / 429 / 401 / timeout / junk response
               -> vector order, reranked=False, the reason in rerank_note,
               and exactly ONE HTTP call (no retry burning trial quota)
    WIRING     the real urllib request: URL, auth header, JSON body
    PACKAGING  rerank.py is inside the Lambda zip, not just handler.py
    SCHEMA     rerank_score survives the agent's Passage model

Semantic-search tests run on the real Pinecone export (skipped without it).
"""
import io
import json
import os
import sys
import urllib.error

import pytest

import fakes
from fakes import DATA

HANDLER = fakes.load_lambda("trial_search")          # puts lambda_tools on sys.path
rerank_mod = sys.modules["rerank"]
REAL_KEY = {"api_key": "co-test-key", "model": "rerank-v3.5"}


@pytest.fixture(autouse=True)
def fresh_cache():
    rerank_mod._cached = None
    yield
    rerank_mod._cached = None


def passages(n):
    return [{"chunk_id": f"c{i}", "text": f"passage {i}", "headings": ["Eligibility"],
             "score": 1 - i / 100, "origin": "search"} for i in range(n)]


class Poster:
    """Stands in for rerank._post: records bodies, answers or raises."""
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def __call__(self, body, api_key):
        self.calls.append((body, api_key))
        if self.error:
            raise self.error
        return self.answer


def http_error(code):
    return urllib.error.HTTPError(rerank_mod.ENDPOINT, code, "err", {}, io.BytesIO(b"{}"))


# ── PRECISION ────────────────────────────────────────────────────────────
def test_cohere_order_and_scores_are_applied(monkeypatch):
    post = Poster({"results": [{"index": 7, "relevance_score": 0.91},
                               {"index": 2, "relevance_score": 0.55},
                               {"index": 0, "relevance_score": 0.10}]})
    monkeypatch.setattr(rerank_mod, "_post", post)
    out = rerank_mod.rerank("neonates", passages(40), 3, lambda: REAL_KEY)

    assert out["reranked"] is True and out["rerank_note"] == ""
    assert [p["chunk_id"] for p in out["passages"]] == ["c7", "c2", "c0"]
    assert [p["rerank_score"] for p in out["passages"]] == [0.91, 0.55, 0.10]
    body, key = post.calls[0]
    assert key == "co-test-key"
    assert body == {"model": "rerank-v3.5", "query": "neonates", "top_n": 3,
                    "documents": [f"Eligibility\npassage {i}" for i in range(40)]}


def test_out_of_range_indexes_are_ignored(monkeypatch):
    monkeypatch.setattr(rerank_mod, "_post", Poster({"results": [
        {"index": 99, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.8}]}))
    out = rerank_mod.rerank("q", passages(3), 2, lambda: REAL_KEY)
    assert [p["chunk_id"] for p in out["passages"]] == ["c1"] and out["reranked"]


# ── FALLBACK: always vector order, always visible, never a retry ─────────
@pytest.mark.parametrize("error,expect", [
    (http_error(429), "HTTP 429: rate limit"),
    (http_error(401), "HTTP 401: key rejected"),
    (TimeoutError(), "TimeoutError"),
])
def test_failures_fall_back_once_without_retry(monkeypatch, error, expect):
    post = Poster(error=error)
    monkeypatch.setattr(rerank_mod, "_post", post)
    out = rerank_mod.rerank("q", passages(40), 8, lambda: REAL_KEY)
    assert out["reranked"] is False and expect in out["rerank_note"]
    assert [p["chunk_id"] for p in out["passages"]] == [f"c{i}" for i in range(8)]
    assert all("rerank_score" not in p for p in out["passages"])
    assert len(post.calls) == 1, "a retry spends another of the trial key's calls"


def test_placeholder_key_makes_no_call_and_is_reread(monkeypatch):
    post = Poster({"results": [{"index": 1, "relevance_score": 0.9}]})
    monkeypatch.setattr(rerank_mod, "_post", post)
    secret = {"api_key": "replace-me", "model": "rerank-v3.5"}
    out = rerank_mod.rerank("q", passages(5), 2, lambda: dict(secret))
    assert out["reranked"] is False and "replace-me" in out["rerank_note"]
    assert post.calls == []

    secret["api_key"] = "co-real"            # operator sets the key; container stays warm
    out = rerank_mod.rerank("q", passages(5), 2, lambda: dict(secret))
    assert out["reranked"] is True and post.calls[0][1] == "co-real"


def test_unreadable_secret_and_empty_results_fall_back(monkeypatch):
    def boom():
        raise KeyError("COHERE_SECRET_ID")
    assert "unreadable" in rerank_mod.rerank("q", passages(3), 2, boom)["rerank_note"]
    monkeypatch.setattr(rerank_mod, "_post", Poster({"results": []}))
    out = rerank_mod.rerank("q", passages(3), 2, lambda: REAL_KEY)
    assert out["reranked"] is False and len(out["passages"]) == 2


# ── WIRING: the real urllib request that leaves the Lambda ───────────────
def test_real_request_url_headers_and_body(monkeypatch):
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def urlopen(request, timeout):
        seen.update(url=request.full_url, method=request.get_method(), timeout=timeout,
                    auth=request.get_header("Authorization"),
                    body=json.loads(request.data))
        return Response(json.dumps({"results": [{"index": 0, "relevance_score": 0.7}]}).encode())
    monkeypatch.setattr(rerank_mod.urllib.request, "urlopen", urlopen)

    out = rerank_mod.rerank("q", passages(2), 1, lambda: REAL_KEY)
    assert out["reranked"] is True
    assert seen["url"] == "https://api.cohere.com/v2/rerank" and seen["method"] == "POST"
    assert seen["auth"] == "Bearer co-test-key" and seen["timeout"] == rerank_mod.TIMEOUT_S
    assert seen["body"]["model"] == "rerank-v3.5" and seen["body"]["top_n"] == 1


# ── RECALL + PRECISION inside semantic_search, on the real export ────────
@pytest.mark.skipif(not (DATA / "dump.json").exists(), reason="no real data")
def test_semantic_search_recalls_the_pool_then_keeps_cohere_top_k(monkeypatch):
    pine = [v for v in json.loads((DATA / "dump.json").read_text())["vectors"]
            if v["metadata"].get("content_type") == "text"]
    asked = {}

    class Match:
        def __init__(s, v, score): s.id, s.metadata, s.score = v["id"], v["metadata"], score

    class Index:
        def query(self, vector, top_k, include_metadata, filter=None):
            asked["top_k"] = top_k
            return type("Q", (), {"matches": [Match(v, 1 - i / 100)
                                              for i, v in enumerate(pine[:top_k])]})()

    class Embed:
        class embeddings:
            @staticmethod
            def create(model, input):
                return type("R", (), {"data": [type("D", (), {"embedding": [0.0] * 1536})()]})()

    monkeypatch.setattr(HANDLER, "_index", Index())
    monkeypatch.setattr(HANDLER, "_openai", Embed)
    monkeypatch.setattr(HANDLER, "_secret", lambda name: REAL_KEY)
    post = Poster({"results": [{"index": 39 - i, "relevance_score": 0.9 - i / 10}
                               for i in range(8)]})
    monkeypatch.setattr(rerank_mod, "_post", post)

    out = HANDLER.semantic_search({"query": "exclusion criteria", "top_k": 8})
    assert asked["top_k"] == HANDLER.RERANK_POOL == 40          # recall: the pool
    assert out["candidates"] == 40 and out["reranked"] is True
    assert [p["chunk_id"] for p in out["passages"]] == [pine[39 - i]["id"] for i in range(8)]
    assert all(p["text"] and p["rerank_score"] is not None for p in out["passages"])
    assert len(post.calls[0][0]["documents"]) == 40             # one call scores all 40


@pytest.mark.skipif(not (DATA / "dump.json").exists(), reason="no real data")
def test_bad_content_type_spends_no_embedding_call(monkeypatch):
    class Embed:
        class embeddings:
            @staticmethod
            def create(model, input):
                raise AssertionError("embedded a request that was going to be rejected")
    monkeypatch.setattr(HANDLER, "_openai", Embed)
    assert HANDLER.semantic_search({"query": "q", "content_type": "tables"})["error"]


# ── PACKAGING and SCHEMA ─────────────────────────────────────────────────
def test_rerank_module_is_packaged_into_the_lambda_zip(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ts_lambda_deploy", fakes.ROOT / "trial_search" / "infra" / "lambda_deploy.py")
    ld = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ld)
    assert {p.name for p in ld._source_files()} >= {"handler.py", "rerank.py"}


def test_rerank_score_survives_the_agents_passage_model():
    from trial_search.schemas import Passage
    p = Passage(chunk_id="c", doc_id="d", text="t", origin="search", score=0.4, rerank_score=0.93)
    assert p.rerank_score == 0.93
