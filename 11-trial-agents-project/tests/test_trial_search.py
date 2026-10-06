"""trial_search: the four Lambda tools on REAL data, and the loop's budgets.

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

    # Identity, from the same graph export: Trial, Disease, Document and the
    # TARGETS / ABOUT edges resolve_trial's Cypher walks.
    nodes = {n["export_id"]: n for n in graph["nodes"]}
    trials = {i: n["properties"] for i, n in nodes.items() if n["labels"] == ["Trial"]}
    diseases = {i: n["properties"] for i, n in nodes.items() if n["labels"] == ["Disease"]}
    doc_of = {r["end"]: nodes[r["start"]]["properties"]["docId"]
              for r in graph["relationships"] if r["type"] == "ABOUT"}
    targets = [(r["start"], r["end"]) for r in graph["relationships"] if r["type"] == "TARGETS"]

    class Session:
        """Mirrors neo4j Session.run(query, parameters=None, **kwargs); returns
        the rows the NEXT traversal is written to return."""
        def __enter__(self): return self
        def __exit__(self, *a): pass

        def run(self, query, parameters=None, **kwargs):
            if "fulltext.queryNodes" in query:
                return self.resolve(parameters or kwargs)
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

        @staticmethod
        def resolve(params):
            """The rows the _RESOLVE Cypher returns, computed in Python over the
            real graph export. Scoring is token overlap — a stand-in for Lucene;
            what is under test is the walk (Trial | Disease -> Trial -> Document)
            and the shaping, not Lucene's ranking."""
            assert params["index"] == "trial_entity_names" and params["pool"] >= params["limit"]
            words = set(params["search"].replace("\\", "").split())
            def score(*fields):
                return len(words & set(" ".join(str(f or "") for f in fields).lower().split()))
            best: dict = {}
            for i, t in trials.items():
                sc = score(t.get("nctId"), t.get("briefTitle"), t.get("officialTitle"), t.get("acronym"))
                if sc:
                    best.setdefault(i, [0, set()])[0] = max(best.get(i, [0])[0], sc)
            for t_id, d_id in targets:
                sc = score(diseases[d_id].get("name"))
                if sc:
                    row = best.setdefault(t_id, [0, set()])
                    row[0] = max(row[0], sc)
                    row[1].add(diseases[d_id]["name"])
            rows = [{"nct_id": trials[i]["nctId"], "acronym": trials[i].get("acronym"),
                     "title": trials[i]["briefTitle"], "doc_id": doc_of.get(i),
                     "score": float(sc), "matched_conditions": sorted(conds)}
                    for i, (sc, conds) in best.items()]
            return sorted(rows, key=lambda r: -r["score"])[:params["limit"]]

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
    h.pine, h.props, h.nxt, h.prv, h.trials = pine, props, nxt, prv, trials
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


class R(BaseModel):
    """Mirrors the Gateway's resolve_trial inputSchema."""
    name: str


class E(BaseModel):
    chunk_id: str
    window: int = 2
    max_tokens: Optional[int] = None
    exclude_ids: Optional[list[str]] = None


def run_loop(lam, script, **overrides):
    from trial_search import core, guardrail
    guardrail._client = GuardrailClient()
    sent = []
    tools = [mcp_tool(P, "resolve_trial", lam.lambda_handler, R, sent),
             mcp_tool(P, "semantic_search", lam.lambda_handler, S, sent),
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


# ── resolve_trial: identity from the graph, not from the prompt ──────────

def test_resolve_by_acronym_found_only_in_the_registry(lam):
    """An acronym that appears nowhere in its protocol's text is still found:
    resolve_trial reads the registry graph, so it returns the doc_id that
    scopes the search."""
    acronym_trial = next(t for t in lam.trials.values() if t.get("acronym"))
    r = lam.resolve_trial({"name": acronym_trial["acronym"]})
    top = r["candidates"][0]
    assert top["nct_id"] == acronym_trial["nctId"]
    assert top["doc_id"] and top["doc_id"].startswith(acronym_trial["nctId"].lower())


def test_resolve_by_condition_walks_targets(lam):
    r = lam.resolve_trial({"name": "glaucoma"})
    top = r["candidates"][0]
    assert top["doc_id"] == "nct02014597-glaucoma-optokinetic"
    assert "Glaucoma" in top["matched_conditions"]


def test_resolve_returns_every_candidate_for_a_shared_name(lam):
    """'COVID-19 vaccine' fits several trials. All come back; none is picked."""
    r = lam.resolve_trial({"name": "COVID-19 vaccine"})
    assert len({c["nct_id"] for c in r["candidates"]}) >= 3


def test_resolve_escapes_lucene_syntax(lam):
    """Upper-case AND is a Lucene operator and '/' a syntax error. Both must
    reach the index as literal, lower-cased text."""
    assert lam._escape_lucene("Phase 2/3 AND (x)") == "phase 2\\/3 and \\(x\\)"
    assert lam.resolve_trial({"name": "  "})["error"]


def test_resolve_then_scoped_search_gives_entities_from_the_graph(lam):
    """The whole point: the model never writes a doc_id or an NCT number from
    memory. The doc_id comes from resolve_trial, and the final entities are
    computed from resolved trials x passage documents — the model's own
    entities ('d') are ignored once the graph has answered."""
    doc = "nct02014597-glaucoma-optokinetic"
    result, response, sent = run_loop(lam, [
        call(P + "resolve_trial", name="glaucoma"),
        call(P + "semantic_search", query="inclusion criteria", doc_id=doc),
        DECIDE])
    assert [n for n, _ in sent] == ["resolve_trial", "semantic_search"]
    assert result["resolve_calls"] == 1 and response.stats.resolve_calls == 1
    assert response.resolved[0].doc_id == doc
    assert response.entities == ["NCT02014597"]
    assert {p.doc_id for p in response.passages} == {doc}


def test_resolve_budget_is_enforced(lam):
    _, response, sent = run_loop(lam, [
        call(P + "resolve_trial", name="glaucoma"),
        call(P + "resolve_trial", name="covid"),
        call(P + "resolve_trial", name="semaglutide"),
        call(P + "resolve_trial", name="obesity"),          # 4th: over the limit of 3
        call(P + "semantic_search", query="q"),
        DECIDE])
    assert [n for n, _ in sent].count("resolve_trial") == 3
    assert response.stats.resolve_calls == 3


def test_trial_without_a_protocol_is_unanswerable_without_searching(lam):
    """doc_id None means the registry knows the trial but no protocol was
    ingested. The model stops; that is an answer, not a hollow decision."""
    _, response, _ = run_loop(lam, [
        call(P + "resolve_trial", name="glaucoma"),
        call("ModelDecision", entities=[], answerable=False, note="no protocol in the corpus")])
    assert response.result_shape == "unanswerable"


def test_resolve_fixture_matches_the_gateway_schema():
    from test_payloads import infra
    gateway = infra("trial_search", "gateway")
    resolve = next(t for t in gateway.TOOL_SCHEMA if t["name"] == "resolve_trial")
    assert set(R.model_fields) == set(resolve["inputSchema"]["properties"])


def test_no_trial_catalogue_in_the_prompt():
    """The prompt describes HOW to find a trial, never WHICH trials exist. A
    doc_id or NCT number written into it is data that goes stale."""
    import re
    text = (fakes.ROOT / "trial_search" / "prompts" / "system.md").read_text()
    assert not re.search(r"NCT\d{8}", text), "an NCT number is hard-coded in the prompt"
    assert not re.search(r"\bnct\d{8}-", text), "a doc_id is hard-coded in the prompt"


def test_weak_tail_is_dropped(lam, monkeypatch):
    """A candidate far below the best match shares only a common word with the
    name. It is dropped, and the drop is counted, not hidden."""
    rows = [{"nct_id": "NCT1", "title": "a", "doc_id": "d1", "score": 3.0},
            {"nct_id": "NCT2", "title": "b", "doc_id": "d2", "score": 2.0},
            {"nct_id": "NCT3", "title": "c", "doc_id": "d3", "score": 1.0}]
    session = type("S", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None,
                             "run": lambda s, q, parameters=None, **k: rows})()
    monkeypatch.setattr(lam, "_driver", type("D", (), {"session": lambda self: session})())
    r = lam.resolve_trial({"name": "x"})
    assert [c["nct_id"] for c in r["candidates"]] == ["NCT1", "NCT2"]
    assert r["dropped_weak"] == 1
