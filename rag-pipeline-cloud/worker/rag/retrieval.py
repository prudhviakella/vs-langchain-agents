"""Searching the index and answering from what comes back.

Query flow:

    question
        |
        v
    _query_vector()
        |
        |-- embed the question with the SAME model used at ingestion
        |-- reuse the cache, so a repeated question costs nothing
        |
        v
    index.query()
        |
        |-- dense search over the WHOLE index
        |-- returns ~30 candidates, not 5
        |-- tuned for RECALL: a chunk missed here is lost for good
        |
        v
    pc.inference.rerank()
        |
        |-- a cross-encoder reads the question and each chunk TOGETHER
        |-- tuned for PRECISION
        |-- runs over ~30, never over the index
        |
        v
    with_table_rows()
        |
        |-- did a table SUMMARY match?
        |-- if so, fetch that table's rows by table_id
        |-- exact lookup, no similarity involved
        |
        v
    with_neighbors()
        |
        |-- is a result SHORT or TRUNCATED?
        |-- if so, fetch prev_id/next_id — the structural replacement
        |-- for overlap, paid at query time instead of on every chunk
        |
        v
    assemble()
        |
        |-- fill a token budget, best first
        |-- then restore DOCUMENT order
        |
        v
    answer()
        |
        v
      prose


WHY TWO SEARCH STAGES
---------------------

At ingestion, the embedding model turned each chunk into one vector. It did
that BEFORE your question existed, so it had to decide what to keep without
knowing what you would ask.

The result is a vector that represents the chunk's TOPIC. It cannot represent
"this chunk answers that question", because the question was not there.

A reranker has no such problem: it receives the question and the chunk
together, in full, so a single word can change the score. That is how it
separates "inclusion criteria" from "exclusion criteria" when a vector
comparison cannot.

The catch is cost. A reranker runs ONCE PER QUESTION-CHUNK PAIR. Over 3,000
chunks that is 3,000 model runs for one question. Dense search is one model
run plus a nearest-neighbour lookup.

So: dense search narrows the index to about thirty, and the reranker orders
those thirty properly.


THE CONSEQUENCE WORTH REMEMBERING
---------------------------------

The reranker only reorders what dense search handed it.

If the right chunk is not in those thirty, no amount of reranking will find
it. That makes the candidate pool the ceiling on everything downstream, and
the one retrieval setting genuinely worth tuning.


WHAT THIS FILE DOES NOT DO
--------------------------

    It does NOT parse documents.
    It does NOT chunk.
    It does NOT embed a corpus.

Everything here runs per question, in well under a second. The index was
built by the ingestion modules.
"""

import json
import time

import numpy as np

from .clients import client, pc
from .config import EMBED_MODEL, ENCODING, LLM_MODEL, NAMESPACE, RERANK_MODEL
from .embedding import embed

# How many candidates to retrieve per final result. Larger recovers more at higher
# rerank cost; below roughly 4x the reranker has too little to reorder for the stage
# to be worth its latency.
CANDIDATE_MULTIPLIER = 6

MIN_CANDIDATES = 24

# Metadata every result carries. Split from EXTRA_FIELDS because these are always
# present, while the extras were added later and may be absent on chunks written by
# an older run — reading them with .get() rather than [] keeps that from raising.
METADATA_FIELDS = ("chunk_id", "content_type", "page", "section_id", "doc_date", "text")

EXTRA_FIELDS = ("prev_id", "next_id", "table_id", "image_uri", "truncated")

def _query_vector(text: str) -> np.ndarray:
    """Embed a query with the same model and cache the corpus used.

    Going through `embed()` rather than calling the API directly means a repeated
    query costs nothing, which matters when sweeping configurations over a gold set:
    the same forty questions get embedded once, not once per configuration.
    """
    return embed([text])[0]

def _row(meta: dict, dense: float = 0.0, rerank: float = 0.0) -> dict:
    """Flatten a Pinecone match into the shape every later stage expects.

    One conversion point, so nothing downstream touches the client's object model.
    When the client changes shape, only this moves.
    """
    return {
        **{k: meta[k] for k in METADATA_FIELDS},
        **{k: meta.get(k) for k in EXTRA_FIELDS},
        "position": int(meta["position"]), "n_tokens": int(meta["n_tokens"]),
        "dense": dense, "rerank": rerank,
    }

def retrieve(index, query: str, top_k: int = 5, filters: dict | None = None,
             groups: list[str] | None = None,
             candidates: int | None = None) -> list[dict]:
    """Dense retrieval over a wide pool, then a cross-encoder rerank.

    The two stages answer different questions. Dense retrieval asks "is this in the
    right region of the space", comparing two vectors that were computed
    independently and never saw each other. The reranker asks "does this passage
    answer this question", reading both together — far more accurate, and far too
    slow to run over the whole index.

    So the first stage is tuned for recall and the second for precision, and the
    candidate pool is what connects them: reranking can only reorder what retrieval
    returned, so a chunk missed at this stage is lost for good.
    """
    pool = candidates or max(MIN_CANDIDATES, top_k * CANDIDATE_MULTIPLIER)

    merged = dict(filters or {})
    if groups:
        # Access filtering is applied at query time, not after. Filtering the
        # returned results would mean a user with narrow access gets fewer than
        # top_k, silently, rather than their own best top_k.
        merged["access"] = {"$in": groups}

    matches = index.query(
        vector=_query_vector(query).tolist(), top_k=pool, namespace=NAMESPACE,
        # Values are needed by MMR, which compares results against each other.
        include_metadata=True, include_values=True, filter=merged or None,
    ).matches
    if not matches:
        return []

    # The manifest check at import time covers configuration; this covers the index
    # itself, in case chunks were written by a run using a different model.
    indexed_model = matches[0].metadata.get("embed_model")
    if indexed_model and indexed_model != EMBED_MODEL:
        raise ValueError(f"index built with {indexed_model}, querying with {EMBED_MODEL}")

    ranked = pc.inference.rerank(
        model=RERANK_MODEL, query=query,
        documents=[m.metadata["text"] for m in matches],
        # Twice top_k, so MMR below has something to choose between. Returning
        # exactly top_k would leave diversification nothing to do.
        top_n=min(top_k * 2, len(matches)), return_documents=False,
    )

    results = []
    for entry in ranked.data:
        match = matches[entry.index]
        row = _row(match.metadata, round(match.score, 4), round(entry.score, 4))
        row["embedding"] = match.values
        results.append(row)
    return results


def search(index, query: str, top_k: int = 5, **kwargs) -> list[dict]:
    """Search and rerank. This is what everything below calls."""
    return retrieve(index, query, top_k=top_k, **kwargs)[:top_k]


def with_table_rows(index, results: list[dict], embed_dims: int,
                    max_rows: int = 6) -> list[dict]:
    """Attach the raw fragments of any table whose summary was retrieved.

    The summary is findable because it contains the question's vocabulary; the rows
    are authoritative because they are the table. Sending only the summary means the
    model answers from a paraphrase it cannot check.
    """
    table_ids = {r["table_id"] for r in results
                 if r["content_type"] == "table_summary" and r.get("table_id")}
    if not table_ids:
        return results

    merged = {r["chunk_id"]: r for r in results}
    for table_id in table_ids:
        # A filtered query with a neutral vector. The filter selects the fragments;
        # ranking among them is irrelevant because all of them are wanted. Pinecone
        # has no filter-only query, so a zero vector stands in for one.
        fragments = index.query(
            vector=[0.0] * embed_dims, top_k=max_rows,
            namespace=NAMESPACE, include_metadata=True,
            filter={"table_id": {"$eq": table_id},
                    "content_type": {"$eq": "table"}},
        ).matches
        for match in fragments:
            row = _row(match.metadata)
            row["embedding"] = None
            # setdefault, not assignment: a fragment already retrieved on merit
            # keeps its rerank score rather than being overwritten with zero.
            merged.setdefault(row["chunk_id"], row)
    # Reading order, so a summary is followed by its own rows.
    return sorted(merged.values(), key=lambda r: r["position"])


def with_neighbors(index, results: list[dict], embed_dims: int,
                   min_tokens: int = 100) -> list[dict]:
    """Expand a short result to its adjacent chunks by prev_id/next_id.

    Every record was written with these two ids — see chunking.py's
    _finalise — specifically as the structural replacement for overlap: a
    chunk is context-poor not because it needs padding at ingestion time, but
    because the ONE question that needs its neighbour is rare, so the cost
    should be paid at query time, for that question only, not on every chunk
    in the corpus whether it is ever needed or not.

    Until this function, those two fields were carried on every result (see
    EXTRA_FIELDS) and never read. Written in, never wired up — the exact kind
    of gap this pipeline's own diagnostics are built to catch elsewhere, and
    did not catch here because there is no report for retrieval the way
    there is for ingestion.

    WHEN A NEIGHBOUR IS FETCHED

    Only for a result under `min_tokens` — a chunk long enough to already
    carry a complete thought needs no expansion, and expanding every result
    unconditionally would inflate the context budget for the common case to
    fix the rare one. A chunk marked `truncated` (chunking.py's oversized-atom
    case) is also expanded, since its stored text is known incomplete.

    Same shape as with_table_rows: a filtered, exact lookup by id, not a
    similarity search — a specific chunk_id is either in the index or it is
    not, and asking a vector search to find it would be paying for a rank
    that is not in question.
    """
    ids_needed = {r[cid_field]
                 for r in results
                 for cid_field in ("prev_id", "next_id")
                 if (r["n_tokens"] < min_tokens or r.get("truncated"))
                 and r.get(cid_field)}
    if not ids_needed:
        return results

    merged = {r["chunk_id"]: r for r in results}
    have = ids_needed - merged.keys()
    if not have:
        # Both neighbours already present on their own merit — nothing to
        # fetch, but still worth returning through the same shape as the
        # fetch path so callers do not need two cases.
        return sorted(merged.values(), key=lambda r: r["position"])

    fragments = index.query(
        vector=[0.0] * embed_dims, top_k=len(have),
        namespace=NAMESPACE, include_metadata=True,
        filter={"chunk_id": {"$in": sorted(have)}},
    ).matches
    for match in fragments:
        row = _row(match.metadata)
        row["embedding"] = None
        # setdefault: a neighbour that is ALSO independently a real result
        # keeps its rerank score rather than being overwritten with zero.
        merged.setdefault(row["chunk_id"], row)

    # Reading order, so an expanded chunk sits next to the one that pulled it
    # in rather than wherever position landed it in the merge.
    return sorted(merged.values(), key=lambda r: r["position"])


# Share of the generation model's window reserved for retrieved context. The rest
# covers the system prompt, the question, and the completion — a budget that ignored
# those would produce a prompt that fits and a response that gets truncated.
CONTEXT_BUDGET_TOKENS = 6000

def assemble(results: list[dict], budget: int = CONTEXT_BUDGET_TOKENS) -> tuple[str, dict]:
    """Fill the budget in relevance order, then sort what fits into document order.

    Both halves matter. Filling by relevance means that when the budget runs out,
    what is dropped is what mattered least — not whatever happened to come last.
    Sorting afterwards means the model reads a narrative rather than a ranking.

    Note the `continue` rather than `break`: a single oversized chunk is skipped and
    smaller ones after it still fit, where breaking would discard everything behind
    it.
    """
    chosen, used = [], 0
    for hit in results:
        if used + hit["n_tokens"] > budget:
            continue
        chosen.append(hit)
        used += hit["n_tokens"]
    chosen.sort(key=lambda h: h["position"])
    # The chunk id and page travel with each passage so the model can cite them, and
    # so an answer can be traced back to a specific place in a specific document.
    context = "\n\n".join(f"[{h['chunk_id']} p{h['page']}]\n{h['text']}" for h in chosen)
    return context, {"chunks": len(chosen), "tokens": used,
                     "dropped": len(results) - len(chosen)}

def answer(index, query: str, top_k: int = 5, table_rows: bool = True,
           neighbors: bool = True, embed_dims: int | None = None,
           **kwargs) -> dict:
    """The whole query path: search, rerank, fetch table rows and neighbours,
    fill the budget, generate.

    That is the entire system. Six steps.
    """
    started = time.time()
    results = search(index, query, top_k=top_k, **kwargs)
    if not results:
        return {"answer": "No relevant context found.", "sources": [], "stats": {}}

    if table_rows and embed_dims:
        results = with_table_rows(index, results, embed_dims)
    if neighbors and embed_dims:
        # After with_table_rows, not before: a table fragment pulled in by
        # table_id has no prev_id/next_id of its own worth chasing, and
        # running this first would mean re-sorting twice for no benefit.
        results = with_neighbors(index, results, embed_dims)

    context, stats = assemble(results)
    response = client.chat.completions.create(
        model=LLM_MODEL, temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer from the provided context only. Cite pages as [p12]. State "
             "plainly if the context does not contain the answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    stats["latency_s"] = round(time.time() - started, 2)
    return {"answer": response.choices[0].message.content,
            "sources": [(h["chunk_id"], h["page"], h["content_type"]) for h in results],
            "images": [h["image_uri"] for h in results if h.get("image_uri")],
            "stats": stats}