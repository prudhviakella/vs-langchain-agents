"""Reconciling the index with a freshly parsed document.

    what is in the index          what we just parsed
            |                             |
            +-------------+---------------+
                          |
                          v
                    set difference
                          |
        +-----------+-----+-----+-----------+
        |           |           |           |
        v           v           v           v
    unchanged     added      removed    moved only
        |           |           |           |
     do nothing  embed +     delete     update metadata,
                 upsert                 no re-embedding

WHY THIS IS A DIFFERENCE AND NOT A REBUILD

Chunk ids come from chunk text. Edit one section of a 200-page report and a
handful of ids change while nearly two hundred stay identical — so embedding
cost tracks what changed, not document size.

The fourth case is the one people miss: text unchanged, position shifted
because a paragraph was inserted above it. The vector is still correct, so it
needs a metadata rewrite and no embedding call at all.

This is also what makes a retry safe. A duplicate run upserts identical vectors
and deletes nothing.


Because chunk ids are derived from chunk text, re-ingestion is a set difference
rather than a rebuild. Editing one section of a 200-page report means a handful of
added and removed chunks against nearly two hundred untouched ones, so embedding
cost is proportional to what changed rather than to document size.

It is also what makes retries safe: a duplicate run upserts identical vectors and
deletes nothing, so a failed task can be retried with no cleanup step.
"""

import json

from .clients import EMBED_DIMS
from .config import NAMESPACE, PINECONE_REQUEST_BYTES
from .embedding import embed_stream
from .index import scan_document

def sync(index, doc_id: str, records: list[dict], window: int = 512) -> dict:
    """Reconcile the index with a freshly parsed document.

    Because chunk ids are derived from chunk text, re-ingestion is a set difference:

        present in both   unchanged — the vector is still correct, do nothing
        new only          embed and upsert
        indexed only      content removed from the document, delete

    Editing one section of a 200-page report typically means a handful of added and
    removed chunks against nearly two hundred untouched ones, so embedding cost is
    proportional to what changed rather than to document size.

    A fourth case matters as much: a chunk whose text is unchanged but whose
    position moved, because a paragraph was inserted above it. Its vector is still
    correct, so it needs only a metadata rewrite, which Pinecone does without
    re-sending the vector and therefore without an embedding call.

    This is also what makes retries safe. A duplicate run upserts identical vectors
    and deletes nothing, so the state machine can retry a failed task with no
    cleanup step and no risk of corrupting the index.
    """
    # Only the fields the comparison reads. Fetching full metadata and vectors for
    # every chunk is what makes a naive sync transfer megabytes per document.
    indexed = scan_document(index, doc_id)
    incoming = {r["meta"]["chunk_id"]: r for r in records}

    added = [r for cid, r in incoming.items() if cid not in indexed]
    removed = [cid for cid in indexed if cid not in incoming]
    shifted = [cid for cid in incoming if cid in indexed and (
        int(indexed[cid].get("position", -1)) != incoming[cid]["meta"]["position"]
        or int(indexed[cid].get("page", -1)) != incoming[cid]["meta"]["page"])]

    plan = {"indexed": len(indexed), "incoming": len(incoming), "added": len(added),
            "removed": len(removed), "unchanged": len(incoming) - len(added),
            "metadata_only": len(shifted)}
    print(f"  {plan}", flush=True)

    if added:
        # Batch size from measured payload against Pinecone's 2 MB request limit
        # rather than a fixed count: at 1536 dimensions a fixed 100 fits
        # comfortably, and at 3072 with large metadata the same 100 starts failing.
        per_vector = len(json.dumps(added[0]["meta"]).encode()) + EMBED_DIMS * 4 + 128
        size = max(1, min(1000, int(PINECONE_REQUEST_BYTES * 0.8 // per_vector)))

        # embed_stream yields windows rather than returning every vector at once, so
        # peak memory stays flat regardless of how many chunks the document made.
        for offset, vectors in embed_stream([r["text"] for r in added], batch=window):
            block = added[offset:offset + len(vectors)]
            for i in range(0, len(block), size):
                index.upsert(
                    vectors=[{"id": r["meta"]["chunk_id"], "values": v.tolist(),
                              "metadata": r["meta"]}
                             for r, v in zip(block[i:i + size], vectors[i:i + size])],
                    namespace=NAMESPACE,
                )
            print(f"    upserted {min(offset + len(vectors), len(added))}"
                  f"/{len(added)}", flush=True)

    # Metadata-only rewrites: no vector sent, no embedding call.
    for cid in shifted:
        index.update(id=cid, set_metadata=incoming[cid]["meta"], namespace=NAMESPACE)

    # Deletes last. Upserting the new version before removing the old means there is
    # never a window where the document is missing from the index.
    for i in range(0, len(removed), 200):
        index.delete(ids=removed[i:i + 200], namespace=NAMESPACE)

    return plan
