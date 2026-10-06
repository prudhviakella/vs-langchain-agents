"""trial_search's tool Lambda — four tools: one to find the protocol, three to read it.

    trial_search agent (LangGraph, AgentCore Runtime)
        │  MCP tool call, SigV4-signed
        v
    AgentCore Gateway ──► THIS LAMBDA (dispatch on tool name)
        │
        ├─ resolve_trial      WHICH PROTOCOL. A trial named in the question
        │                     ("IMbrave150", "NCT03434379", "the glaucoma
        │                     trial") -> its nctId, title and doc_id, read
        │                     from the registry graph in Neo4j. Nothing about
        │                     the corpus is written in the prompt.
        │
        ├─ semantic_search    ENTRY. Embed the question, query Pinecone for
        │                     a wide pool (RECALL), then Cohere re-ranks it
        │                     and keeps top_k (PRECISION) — see rerank.py.
        │                     Text comes back on the match itself
        │                     (metadata.text, written by chunking._finalise).
        │
        ├─ expand_neighbors   CASE A. A correct chunk cut at a boundary.
        │                     Neo4j walks NEXT to find neighbour ids,
        │                     nearest first, stopping at a token budget
        │                     using each Chunk node's own n_tokens.
        │                     Pinecone fetch(ids) returns their text.
        │
        └─ expand_table       CASE C. A table_summary matched, but the
                              exact values live in its row fragments.
                              Pinecone fetch(summary) -> filtered query
                              on doc_id AND table_id.

THE CONTRACT BETWEEN THE TWO STORES

Taken from the original graph_rag package (answer.local, chunks.fetch_text):
"The graph decides which chunks; the vector store returns what they say."
Neo4j Chunk nodes carry no text by design. Pinecone carries the text. The
join is chunk_id, and fetching by id is exact.

The same split applies to identity. Which trial a name means, and which
protocol document belongs to it, is a registry fact. The graph holds it:

    (Document {docId})-[:ABOUT]->(Trial {nctId, acronym, briefTitle})
                                       -[:TARGETS]->(Disease {name})

An earlier version copied that mapping into the system prompt as a table of
20 rows. That table had to be edited by hand for every new protocol, cost
prompt tokens on every turn whether a trial was named or not, and stopped
working at a few hundred trials. resolve_trial reads the same facts from
the graph at question time, so a protocol added to the graph is findable
with no prompt change.

WHY resolve_trial LIVES HERE AND NOT ONLY IN trial_graph

trial_graph can also resolve a name, but reaching it means the supervisor
calls a second agent: a full LLM loop, about 20 seconds. This tool is one
indexed Neo4j query from a Lambda that already holds the Neo4j credential
for expand_neighbors. It uses the fulltext index trial_graph created
(trial_graph/setup_neo4j.py), so both agents resolve names the same way.

WHY NEIGHBOURS COME FROM NEO4J AND NOT FROM Pinecone next_id

Pinecone metadata has next_id/prev_id, but walking them is one fetch per
hop: chunk 5 cannot be requested before chunk 4 has been read. Neo4j
returns every id within the window in one query, and widening the window
from 5 to 10 is one changed number, not five more round trips. Verified
against the real graph: 5,764 Chunk nodes, every one has n_tokens, NEXT
out-degree is exactly 1, and no NEXT edge crosses a document boundary.

WHY expand_table FILTERS ON doc_id AS WELL AS table_id

table_id is sha256 of Docling's self_ref ("#/tables/0"), which is
document-local. Verified against the real Pinecone export: 95 of 109
table_id values are shared by more than one document, one of them by 11.
Filtering on table_id alone would return rows from eleven trials' tables.

WHAT THIS DOES NOT DO

    - It does not enforce call budgets across a turn. The Lambda is
      stateless; RetrievalMiddleware in the agent owns every cross-call
      budget and passes the remaining token allowance in as max_tokens.
    - It does not decide whether expansion is needed. The model decides
      that after reading. This file only executes a requested expansion.
    - It does not re-embed anything for expand_table. The summary's own
      stored vector is used as the query vector; the filter already
      narrows the result to exactly that table's fragments.
    - resolve_trial does not pick one trial when a name fits several. It
      returns every candidate with its score; choosing, or asking, is the
      agent's decision.
    - resolve_trial does not answer registry questions (sponsors, sites,
      phases). It returns identity only: nctId, acronym, title, doc_id and
      the conditions a name matched through.
"""
import json
import logging
import os

import boto3

from rerank import rerank
from symbol_fonts import normalize

log = logging.getLogger()
log.setLevel(logging.INFO)

# Must match worker/rag/config.py EMBED_MODEL — the model every vector in
# the index was embedded with. A mismatch returns plausible rankings and
# no error.
EMBED_MODEL = "text-embedding-3-small"

HARD_MAX_WINDOW = 10      # defense in depth; the middleware clamps first
HARD_MAX_TOP_K = 20
# The recall pool semantic_search hands to the re-ranker. Wide enough that the
# right passage is in it; Cohere scores all of them in ONE call, so a larger
# pool costs no extra calls against a trial key's 10 per minute.
RERANK_POOL = int(os.environ.get("RERANK_POOL", "40"))
HARD_MAX_POOL = 100
HARD_MAX_FRAGMENTS = 50   # real max observed in the corpus: 18

# Every content_type in the index. The Gateway cannot enforce an enum, so this
# does: an unknown value would otherwise become a Pinecone filter that matches
# nothing, reported as "the corpus does not cover this" — a confident negative
# produced by a typo.
CONTENT_TYPES = ("text", "table", "table_summary", "figure", "formula")

# resolve_trial. The index is created by trial_graph/setup_neo4j.py over
# Trial(nctId, briefTitle, officialTitle, acronym), Disease(name) and others.
NAME_INDEX = "trial_entity_names"
# How many index hits feed the query. A fulltext query without a limit returns
# EVERY node that shares one token with the name — "trial" is in most titles.
NAME_POOL = 50
RESOLVE_LIMIT = 8         # trials returned; more than this is "too vague"
# A candidate scoring under this fraction of the best one is dropped. The best
# match is the name itself; a trial far below it shares one common word with
# the question ("trial", "study", "1"). Measured on the real graph: the right
# trial led every real name, and the noise sat in a long tail beneath it.
RELATIVE_FLOOR = 0.5

_openai = None
_index = None
_driver = None


# ── clients, created once per warm container ────────────────────────────

def _secret(env_name: str) -> dict:
    client = boto3.client("secretsmanager")
    return json.loads(client.get_secret_value(
        SecretId=os.environ[env_name])["SecretString"])


def _get_openai():
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI(api_key=_secret("OPENAI_SECRET_ID")["api_key"])
    return _openai


def _get_index():
    global _index
    if _index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=_secret("PINECONE_SECRET_ID")["api_key"])
        _index = pc.Index(os.environ.get("PINECONE_INDEX", "rag-docs"))
    return _index


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        s = _secret("NEO4J_SECRET_ID")
        _driver = GraphDatabase.driver(s["uri"], auth=(s.get("user", "neo4j"), s["password"]))
    return _driver


# ── shared shaping ──────────────────────────────────────────────────────

def _passage(chunk_id: str, meta: dict, origin: str, score=None) -> dict:
    """One shape for every tool, so the agent never branches on which
    tool produced a passage. Text and headings pass through
    symbol_fonts.normalize(): "BP \\uf0b3 150" is read as "BP ≥ 150"."""
    return {
        "chunk_id": chunk_id,
        "doc_id": meta.get("doc_id", ""),
        "score": score,
        "text": normalize(meta.get("text", "")),
        "content_type": meta.get("content_type", ""),
        "headings": [normalize(h) for h in meta.get("headings", []) or []],
        "page": meta.get("page"),
        "position": meta.get("position"),
        "table_id": meta.get("table_id", ""),
        "n_fragments": meta.get("n_fragments"),
        "n_tokens": meta.get("n_tokens"),
        "origin": origin,
    }


def _fetch(ids: list[str]) -> dict:
    """id -> Pinecone vector. Batched: Pinecone caps ids per fetch."""
    found = {}
    for start in range(0, len(ids), 100):
        page = _get_index().fetch(ids=ids[start:start + 100])
        found.update(page.vectors)
    return found


# ── tool 1: resolve_trial ───────────────────────────────────────────────

# One index-driven query. Every node it touches is reached from an index hit,
# so its cost follows the number of hits (at most NAME_POOL), not the number
# of trials in the graph.
#
#   index hit is a Trial    -> that trial                (name, NCT, acronym)
#   index hit is a Disease  -> every trial that TARGETS it ("the glaucoma trial")
#   either way              -> the Document ABOUT it, if a protocol was ingested
_RESOLVE = """
CALL db.index.fulltext.queryNodes($index, $search, {limit: $pool})
YIELD node, score
WHERE node:Trial OR node:Disease
OPTIONAL MATCH (node)<-[:TARGETS]-(via:Trial)
WITH CASE WHEN node:Trial THEN node ELSE via END AS t, score,
     CASE WHEN node:Disease THEN node.name END AS condition
WHERE t IS NOT NULL
OPTIONAL MATCH (d:Document)-[:ABOUT]->(t)
WITH t, d, max(score) AS score, collect(DISTINCT condition) AS matched_conditions
RETURN t.nctId AS nct_id, t.acronym AS acronym, t.briefTitle AS title,
       d.docId AS doc_id, score, matched_conditions
ORDER BY score DESC
LIMIT $limit
"""


def _escape_lucene(text: str) -> str:
    """The analyst's words as literal search terms.

    Lower-cased first: Lucene reads upper-case AND / OR / NOT as operators,
    so "atezolizumab AND bevacizumab" would change the query's logic. The
    index analyzer lower-cases anyway, so nothing is lost. Then every
    Lucene special character is escaped — unescaped, "Phase 2/3" is a
    syntax error, not a weak match. Same rule as trial_graph's handler.
    """
    special = '+-&|!(){}[]^"~*?:\\/'
    return "".join(f"\\{c}" if c in special else c for c in text.lower())


def resolve_trial(args: dict) -> dict:
    """A trial's name, as the analyst wrote it -> candidate trials with doc_ids.

    STEP 1  escape the name into a fulltext query. Filler words ("the",
            "a", "that") are removed by the index's english analyzer, which
            queryNodes applies to the query too — not by a list here
    STEP 2  one indexed query: trials matched by name, or through a condition
    STEP 3  drop the tail: candidates under RELATIVE_FLOOR x the best score
    STEP 4  shape candidates; a trial with no protocol keeps doc_id None,
            so the agent can say "no protocol for this trial" instead of
            searching the whole corpus for it
    """
    name = str(args.get("name", "")).strip()
    search = _escape_lucene(name)
    if not search.strip():
        return {"error": True, "detail": "name is empty — pass the trial's name, "
                                         "NCT number, acronym, drug or condition."}

    try:
        with _get_driver().session() as session:
            rows = [dict(r) for r in session.run(_RESOLVE, parameters={
                "index": NAME_INDEX, "search": search,
                "pool": NAME_POOL, "limit": RESOLVE_LIMIT})]
    except Exception as exc:                       # the driver raises many classes
        detail = str(exc)[:300]
        if NAME_INDEX in detail or "fulltext" in detail.lower():
            detail += (f" — the {NAME_INDEX!r} index is created by "
                       "trial_graph/setup_neo4j.py; run it once.")
        return {"error": True, "detail": detail}

    best = max((float(r.get("score") or 0.0) for r in rows), default=0.0)
    kept = [r for r in rows if float(r.get("score") or 0.0) >= RELATIVE_FLOOR * best]

    candidates = [{
        "nct_id": r["nct_id"],
        "acronym": r.get("acronym") or "",
        "title": normalize(r.get("title") or ""),
        "doc_id": r.get("doc_id"),
        "matched_conditions": [c for c in (r.get("matched_conditions") or []) if c],
        "score": round(float(r.get("score") or 0.0), 3),
    } for r in kept if r.get("nct_id")]
    return {"name": name, "candidates": candidates,
            "dropped_weak": len(rows) - len(kept),
            "truncated": len(rows) >= RESOLVE_LIMIT}


# ── tool 2: semantic_search ─────────────────────────────────────────────

def semantic_search(args: dict) -> dict:
    """Recall a wide pool by vector similarity, then keep the top_k that
    Cohere judges most relevant. The result says whether re-ranking ran."""
    top_k = max(1, min(int(args.get("top_k", 8)), HARD_MAX_TOP_K))

    # STEP 1  validate before spending an embedding call on a bad request
    if args.get("content_type") and args["content_type"] not in CONTENT_TYPES:
        return {"error": True,
                "detail": f"content_type {args['content_type']!r} is not valid; use one of "
                          f"{', '.join(CONTENT_TYPES)}, or omit it to search every type."}

    clauses = []
    if args.get("content_type"):
        clauses.append({"content_type": {"$eq": args["content_type"]}})
    if args.get("doc_id"):
        clauses.append({"doc_id": {"$eq": args["doc_id"]}})
    flt = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)

    # STEP 2  RECALL — a pool wider than top_k, in vector order
    embedding = _get_openai().embeddings.create(
        model=EMBED_MODEL, input=args["query"]).data[0].embedding
    pool = min(max(top_k, RERANK_POOL), HARD_MAX_POOL)
    result = _get_index().query(vector=embedding, top_k=pool,
                                include_metadata=True, filter=flt)
    candidates = [_passage(m.id, m.metadata or {}, "search", m.score)
                  for m in result.matches]

    # STEP 3  PRECISION — Cohere keeps the top_k; falls back to vector order
    ranked = rerank(args["query"], candidates, top_k,
                    read_secret=lambda: _secret("COHERE_SECRET_ID"))
    return {**ranked, "candidates": len(candidates)}


# ── tool 3: expand_neighbors (Case A) ───────────────────────────────────

# Path length cannot be a Cypher parameter; it is interpolated only after
# being forced to a bounded int — never a caller's string.
_NEIGHBOURS = """
MATCH (seed:Chunk {chunkId: $id})
OPTIONAL MATCH path = (seed)-[:NEXT*1..%d]-(n:Chunk)
WITH seed, n, min(length(path)) AS distance
RETURN seed.position AS seed_position,
       n.chunkId AS chunk_id, n.n_tokens AS n_tokens,
       n.position AS position, distance
ORDER BY distance, position
"""


def expand_neighbors(args: dict) -> dict:
    """Case A. Nearest-first, stop at the first chunk that would exceed
    max_tokens — not skip to a smaller, farther one, which would leave a
    gap in the text the model reads.
    """
    chunk_id = args["chunk_id"]
    window = max(1, min(int(args.get("window", 2)), HARD_MAX_WINDOW))
    max_tokens = max(0, int(args.get("max_tokens", 0)))
    exclude = set(args.get("exclude_ids") or [])

    # STEP 1 — neighbour ids and sizes from the graph, one query
    with _get_driver().session() as session:
        rows = [dict(r) for r in session.run(_NEIGHBOURS % window, parameters={"id": chunk_id})]
    if not rows:
        return {"error": True, "detail": f"chunk_id {chunk_id!r} is not in the graph"}

    seed_position = rows[0]["seed_position"]
    candidates = [r for r in rows if r["chunk_id"] and r["chunk_id"] not in exclude]

    # STEP 2 — take nearest first until the token allowance runs out
    chosen, used, stopped_by = [], 0, "window"
    for row in candidates:
        cost = int(row["n_tokens"] or 0)
        if used + cost > max_tokens:
            stopped_by = "token_budget"
            break
        chosen.append(row)
        used += cost
    if stopped_by == "window" and len(rows) == 1 and rows[0]["chunk_id"] is None:
        stopped_by = "document_edge"

    # STEP 3 — text for exactly those ids, by id
    vectors = _fetch([r["chunk_id"] for r in chosen])
    passages = [_passage(r["chunk_id"], (vectors[r["chunk_id"]].metadata or {}), "neighbor")
                for r in chosen if r["chunk_id"] in vectors]
    passages.sort(key=lambda p: p["position"] if p["position"] is not None else 0)

    return {"passages": passages, "tokens_used": used, "stopped_by": stopped_by,
            "seed_position": seed_position, "window": window}


# ── tool 4: expand_table (Case C) ───────────────────────────────────────

def expand_table(args: dict) -> dict:
    chunk_id = args["chunk_id"]
    max_tokens = max(0, int(args.get("max_tokens", 0)))
    exclude = set(args.get("exclude_ids") or [])

    # STEP 1 — the summary itself: its table_id, doc_id, n_fragments, vector
    summary = _fetch([chunk_id]).get(chunk_id)
    if summary is None:
        return {"error": True, "detail": f"chunk_id {chunk_id!r} not found"}
    meta = summary.metadata or {}
    if meta.get("content_type") != "table_summary" or not meta.get("table_id"):
        return {"error": True,
                "detail": f"{chunk_id!r} is content_type={meta.get('content_type')!r}, "
                          "not a table_summary. expand_table only works from a summary; "
                          "use expand_neighbors for other chunks."}

    n_fragments = max(1, min(int(meta.get("n_fragments") or 1), HARD_MAX_FRAGMENTS))

    # STEP 2 — exactly this document's copy of this table
    result = _get_index().query(
        vector=list(summary.values), top_k=n_fragments, include_metadata=True,
        filter={"$and": [{"doc_id": {"$eq": meta["doc_id"]}},
                         {"table_id": {"$eq": meta["table_id"]}},
                         {"content_type": {"$eq": "table"}}]})

    fragments = sorted(result.matches, key=lambda m: (m.metadata or {}).get("position", 0))

    # STEP 3 — same token rule as neighbours: in order, stop when it no longer fits
    passages, used, stopped_by = [], 0, "complete"
    for m in fragments:
        if m.id in exclude:
            continue
        cost = int((m.metadata or {}).get("n_tokens") or 0)
        if used + cost > max_tokens:
            stopped_by = "token_budget"
            break
        passages.append(_passage(m.id, m.metadata or {}, "table"))
        used += cost

    return {"passages": passages, "tokens_used": used, "stopped_by": stopped_by,
            "n_fragments": n_fragments}


# ── dispatch ────────────────────────────────────────────────────────────

_TOOLS = {"resolve_trial": resolve_trial,
          "semantic_search": semantic_search,
          "expand_neighbors": expand_neighbors,
          "expand_table": expand_table}


def lambda_handler(event, context):
    cc = getattr(context, "client_context", None)
    name = cc.custom.get("bedrockAgentCoreToolName", "") if cc and cc.custom else ""
    # The Gateway may prefix the target name: "trial-search-tools___expand_table".
    matched = next((t for t in _TOOLS if name.endswith(t)), None)
    if matched is None:
        log.error("unknown tool requested: %r", name)
        return {"error": True, "detail": f"unknown tool: {name}"}

    log.info("dispatching to %s", matched)
    try:
        result = _TOOLS[matched](event)
    except KeyError as exc:
        return {"error": True, "detail": f"missing required argument: {exc}"}
    # The Lambda runtime serializes with plain json.dumps. Guarantee it can.
    return json.loads(json.dumps(result, default=str))
