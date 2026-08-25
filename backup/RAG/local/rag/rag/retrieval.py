"""Searching the index and answering from what comes back.

Five steps, and that is the whole system:

    search      dense retrieval over the index
    rerank      a cross-encoder reorders the candidates
    filter      restrict what search is allowed to look at (passed to search)
    table rows  fetch the rows behind any table summary that matched
    assemble    fill a token budget, then restore document order

Everything here is called from the retrieval notebook. Reading this file is optional
— the notebook shows what each step does and why.
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

EXTRA_FIELDS = ("prev_id", "next_id", "table_id", "image_uri")

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
           embed_dims: int | None = None, **kwargs) -> dict:
    """The whole query path: search, rerank, fetch table rows, fill the budget,
    generate.

    That is the entire system. Five steps.
    """
    started = time.time()
    results = search(index, query, top_k=top_k, **kwargs)
    if not results:
        return {"answer": "No relevant context found.", "sources": [], "stats": {}}

    if table_rows and embed_dims:
        results = with_table_rows(index, results, embed_dims)

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