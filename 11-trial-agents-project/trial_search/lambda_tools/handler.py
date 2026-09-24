"""trial_search's tool Lambda — three tools, one per retrieval case.

    trial_search agent (LangGraph, AgentCore Runtime)
        │  MCP tool call, SigV4-signed
        v
    AgentCore Gateway ──► THIS LAMBDA (dispatch on tool name)
        │
        ├─ semantic_search    ENTRY. Embed the question, query Pinecone.
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
"""
import json
import logging
import os

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

# Must match worker/rag/config.py EMBED_MODEL — the model every vector in
# the index was embedded with. A mismatch returns plausible rankings and
# no error.
EMBED_MODEL = "text-embedding-3-small"

HARD_MAX_WINDOW = 10      # defense in depth; the middleware clamps first
HARD_MAX_TOP_K = 20
HARD_MAX_FRAGMENTS = 50   # real max observed in the corpus: 18

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
    tool produced a passage."""
    return {
        "chunk_id": chunk_id,
        "doc_id": meta.get("doc_id", ""),
        "score": score,
        "text": meta.get("text", ""),
        "content_type": meta.get("content_type", ""),
        "headings": meta.get("headings", []),
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


# ── tool 1: semantic_search ─────────────────────────────────────────────

def semantic_search(args: dict) -> dict:
    top_k = max(1, min(int(args.get("top_k", 8)), HARD_MAX_TOP_K))

    embedding = _get_openai().embeddings.create(
        model=EMBED_MODEL, input=args["query"]).data[0].embedding

    clauses = []
    if args.get("content_type"):
        clauses.append({"content_type": {"$eq": args["content_type"]}})
    if args.get("doc_id"):
        clauses.append({"doc_id": {"$eq": args["doc_id"]}})
    flt = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)

    result = _get_index().query(vector=embedding, top_k=top_k,
                                include_metadata=True, filter=flt)
    return {"passages": [_passage(m.id, m.metadata or {}, "search", m.score)
                         for m in result.matches]}


# ── tool 2: expand_neighbors (Case A) ───────────────────────────────────

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
        rows = [dict(r) for r in session.run(_NEIGHBOURS % window, id=chunk_id)]
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


# ── tool 3: expand_table (Case C) ───────────────────────────────────────

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

_TOOLS = {"semantic_search": semantic_search,
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
        return _TOOLS[matched](event)
    except KeyError as exc:
        return {"error": True, "detail": f"missing required argument: {exc}"}
