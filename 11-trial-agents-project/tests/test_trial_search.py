"""trial_search: the three Lambda tools on REAL data, and the loop's budgets.

Data: dump.json (Pinecone export) and graph_dump.json (Neo4j export) from
TRIAL_DATA_DIR. Skipped if absent.
"""
import asyncio
import json
from typing import Optional

import pytest
from pydantic import BaseModel

import fakes
from fakes import DATA, Fake, GuardrailClient, call, mcp_tool

pytestmark = pytest.mark.skipif(not (DATA / "dump.json").exists(), reason="no real data")

SEED = "nct02014597-glaucoma-optokinetic:0441e87ced1c24e4:0"   # position 24
P = "trial-search-tools___"


@pytest.fixture(scope="module")
def lam():
    pine = {v["id"]: v for v in json.loads((DATA / "dump.json").read_text())["vectors"]}
    graph = json.loads((DATA / "graph_dump.json").read_text())
    chunks = {n["export_id"]: n["properties"] for n in graph["nodes"] if n["labels"] == ["Chunk"]}
    nxt = {chunks[r["start"]]["chunkId"]: chunks[r["end"]]["chunkId"]
           for r in graph["relationships"] if r["type"] == "NEXT"}
    prv = {b: a for a, b in nxt.items()}
    props = {p["chunkId"]: p for p in chunks.values()}

    class Session:
        """Mirrors neo4j Session.run(query, parameters=None, **kwargs); returns
        the rows the NEXT traversal is written to return."""
        def __enter__(self): return self
        def __exit__(self, *a): pass

        def run(self, query, parameters=None, **kwargs):
            seed = (parameters or kwargs)["id"]
            window = int(query.split("*1..")[1].split("]")[0])
            if seed not in props:
                return []
            rows = []
            for step in (nxt, prv):
                cur = seed
                for d in range(1, window + 1):
                    cur = step.get(cur)
                    if cur is None:
                        break
                    rows.append({"seed_position": props[seed]["position"], "chunk_id": cur,
                                 "n_tokens": props[cur]["n_tokens"],
                                 "position": props[cur]["position"], "distance": d})
            return sorted(rows, key=lambda r: (r["distance"], r["position"])) or [
                {"seed_position": props[seed]["position"], "chunk_id": None,
                 "n_tokens": None, "position": None, "distance": None}]

    class V:
        def __init__(s, v): s.id, s.values, s.metadata, s.score = v["id"], v["values"], v["metadata"], .5

    class Index:
        def fetch(self, ids):
            return type("F", (), {"vectors": {i: V(pine[i]) for i in ids if i in pine}})()

        def query(self, vector, top_k, include_metadata, filter=None):
            if filter is None:   # semantic: the seed and a table summary
                ids = [SEED, next(k for k, v in pine.items()
                                  if v["metadata"].get("content_type") == "table_summary"
                                  and v["metadata"].get("n_fragments", 0) >= 3)]
                return type("Q", (), {"matches": [V(pine[i]) for i in ids]})()
            conds = filter.get("$and", [filter])
            hits = [V(v) for v in pine.values()
                    if all(v["metadata"].get(k) == c["$eq"] for d in conds for k, c in d.items())]
            return type("Q", (), {"matches": hits[:top_k]})()

    class Embed:
        class embeddings:
            @staticmethod
            def create(model, input):
                return type("R", (), {"data": [type("D", (), {"embedding": [0.0] * 1536})()]})()

    h = fakes.load_lambda("trial_search")
    h._driver = type("D", (), {"session": lambda self: Session()})()
    h._index, h._openai = Index(), Embed
    h.pine, h.props, h.nxt, h.prv = pine, props, nxt, prv
    return h


# ── Lambda tools on real data ────────────────────────────────────────────
def test_neighbors_nearest_first_and_ordered(lam):
    r = lam.expand_neighbors({"chunk_id": SEED, "window": 2, "max_tokens": 5000})
    assert [p["position"] for p in r["passages"]] == [22, 23, 25, 26]
    assert all(p["text"] and p["origin"] == "neighbor" for p in r["passages"])


def test_widening_fetches_only_the_new_ring(lam):
    seen = [p["chunk_id"] for p in lam.expand_neighbors(
        {"chunk_id": SEED, "window": 2, "max_tokens": 5000})["passages"]]
    r = lam.expand_neighbors({"chunk_id": SEED, "window": 5, "max_tokens": 5000,
                              "exclude_ids": seen})
    assert {p["position"] for p in r["passages"]} == {19, 20, 21, 27, 28, 29}


def test_token_budget_stops_nearest_first(lam):
    tight = lam.props[lam.nxt[SEED]]["n_tokens"] + lam.props[lam.prv[SEED]]["n_tokens"]
    r = lam.expand_neighbors({"chunk_id": SEED, "window": 5, "max_tokens": tight})
    assert {p["position"] for p in r["passages"]} == {23, 25}
    assert r["stopped_by"] == "token_budget"


def test_table_rows_from_the_right_document_only(lam):
    summary = next(v for v in lam.pine.values()
                   if v["metadata"].get("content_type") == "table_summary"
                   and v["metadata"].get("n_fragments", 0) >= 3)
    m = summary["metadata"]
    sharing = {v["metadata"]["doc_id"] for v in lam.pine.values()
               if v["metadata"].get("table_id") == m["table_id"]}
    assert len(sharing) > 1, "this table_id is shared across documents"
    r = lam.expand_table({"chunk_id": summary["id"], "max_tokens": 10 ** 6})
    assert len(r["passages"]) == m["n_fragments"]
    assert {p["doc_id"] for p in r["passages"]} == {m["doc_id"]}
    assert "use expand_neighbors" in lam.expand_table(
        {"chunk_id": SEED, "max_tokens": 1})["detail"]


# ── the agent loop's budgets ─────────────────────────────────────────────
class S(BaseModel):
    """Mirrors the Gateway's semantic_search inputSchema (infra/gateway.py). A
    field missing here is silently dropped before the Lambda — doc_id and
    content_type once were, so no scoped search was ever exercised."""
    query: str
    top_k: int = 8
    content_type: str | None = None
    doc_id: str | None = None


class E(BaseModel):
    chunk_id: str
    window: int = 2
    max_tokens: Optional[int] = None
    exclude_ids: Optional[list[str]] = None


def run_loop(lam, script, **overrides):
    from trial_search import core, guardrail
    guardrail._client = GuardrailClient()
    sent = []
    tools = [mcp_tool(P, "semantic_search", lam.lambda_handler, S, sent),
             mcp_tool(P, "expand_neighbors", lam.lambda_handler, E, sent),
             mcp_tool(P, "expand_table", lam.lambda_handler, E, sent)]
    s = fakes.settings_for("trial_search", **overrides)
    result = asyncio.run(core.build_agent(tools, model=Fake(messages=iter(script)), cfg=s)
                         .ainvoke({"messages": [{"role": "user", "content": "q"}]}))
    return result, core.assemble(result, cfg=s), sent


DECIDE = call("ModelDecision", entities=["d"], answerable=True, note="")


def test_budgets_are_enforced_by_rewriting_arguments(lam):
    summary = next(k for k, v in lam.pine.items()
                   if v["metadata"].get("content_type") == "table_summary"
                   and v["metadata"].get("n_fragments", 0) >= 3)
    result, response, sent = run_loop(lam, [
        call(P + "semantic_search", query="q"),
        call(P + "expand_neighbors", chunk_id=SEED, window=2, max_tokens=999999),
        call(P + "expand_neighbors", chunk_id=SEED, window=500),
        call(P + "expand_table", chunk_id=summary),
        call(P + "expand_neighbors", chunk_id=SEED, window=3),
        call(P + "expand_neighbors", chunk_id=SEED, window=3),   # 4th: over the limit
        DECIDE])
    nb = [a for n, a in sent if n == "expand_neighbors"]
    assert len(nb) == 3, "the 4th call must be refused before the Lambda"
    assert nb[0]["max_tokens"] == 6000, "the model's 999999 must be replaced"
    assert nb[1]["window"] == 10, "window 500 must be clamped"
    assert nb[0]["max_tokens"] > nb[1]["max_tokens"] > nb[2]["max_tokens"]
    assert len(nb[0]["exclude_ids"]) < len(nb[1]["exclude_ids"]) < len(nb[2]["exclude_ids"])
    assert (result["neighbor_calls"], result["table_calls"]) == (3, 1)
    assert response.result_shape == "passages"
    assert {p.origin for p in response.passages} == {"search", "neighbor", "table"}
    ids = [p.chunk_id for p in response.passages]
    assert len(ids) == len(set(ids))


def test_tiny_token_budget_holds(lam):
    result, _, _ = run_loop(lam, [call(P + "semantic_search", query="q"),
                                  call(P + "expand_neighbors", chunk_id=SEED, window=5),
                                  call(P + "expand_neighbors", chunk_id=SEED, window=5),
                                  DECIDE], expansion_token_budget=60)
    assert result["expansion_tokens"] <= 60


def test_decision_without_search_is_not_executed(lam):
    _, response, _ = run_loop(lam, [DECIDE])
    assert response.result_shape == "not_executed"


def test_unknown_content_type_is_rejected_not_searched(lam):
    """The Gateway cannot enforce an enum. Without this check, a typo becomes a
    filter matching nothing, reported as 'the corpus does not cover this'."""
    r = lam.semantic_search({"query": "q", "content_type": "tables"})
    assert r["error"] and "table_summary" in r["detail"]
    assert "passages" in lam.semantic_search({"query": "q", "content_type": "formula"})


def test_every_search_is_recorded_as_it_ran(lam):
    """The analyst's Queries tab: each search's query, scope, filter, the recall
    pool and what was kept — taken from the executed call, not from the model.
    A failed search is recorded too, marked as failed."""
    doc = "nct03434379-hepatocellular-atezo-bev"
    _, response, sent = run_loop(lam, [
        call(P + "semantic_search", query="exclusion criteria", doc_id=doc, top_k=8),
        call(P + "semantic_search", query="dose table", content_type="tables"),   # invalid filter
        DECIDE])
    first, second = response.searches
    assert (first.query, first.doc_id, first.top_k, first.succeeded) == ("exclusion criteria", doc, 8, True)
    assert first.results == len([p for p in response.passages if p.origin == "search"]) > 0
    assert first.candidates >= first.results and first.reranked is not None
    assert (second.query, second.content_type, second.succeeded) == ("dose table", "tables", False)
    assert [a["query"] for n, a in sent if n == "semantic_search"] == ["exclusion criteria", "dose table"]


def test_search_fixture_matches_the_gateway_schema():
    """The loop tests feed arguments through S. If S lacks a field the Gateway
    accepts, that field silently vanishes and scoped searches go untested."""
    from test_payloads import infra
    gateway = infra("trial_search", "gateway")
    search = next(t for t in gateway.TOOL_SCHEMA if t["name"] == "semantic_search")
    assert set(S.model_fields) == set(search["inputSchema"]["properties"])
