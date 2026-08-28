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

# EMBED_DIMS:
#     The embedding dimension, probed once at connect time. Used here only to
#     estimate how large one vector will be on the wire.
from .clients import EMBED_DIMS

# NAMESPACE:
#     The Pinecone namespace to write into. Empty string is the default one.
#
# PINECONE_REQUEST_BYTES:
#     The maximum size of a single upsert request. Batch size is derived from
#     it rather than fixed, because a fixed count breaks when the dimension or
#     the metadata size changes.
from .config import NAMESPACE, PINECONE_REQUEST_BYTES

# embed_stream():
#     Yields vectors in windows rather than returning all of them, so peak
#     memory stays flat regardless of how many chunks a document produced.
from .embedding import embed_stream

# scan_document():
#     Reads back only the fields this comparison needs — not full metadata and
#     not vectors. Fetching everything is what makes a naive sync transfer
#     megabytes per document.
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
    # ------------------------------------------------------------------
    # THE TWO SIDES OF THE COMPARISON
    # ------------------------------------------------------------------
    #
    #     indexed    what is in Pinecone right now, from a previous run
    #     incoming   what we just built from the document as it is today
    #
    # Both are keyed by chunk_id, and chunk_id is derived from chunk TEXT.
    # That is what makes this a set comparison rather than a rebuild: text
    # that did not change produces the same id, so it appears in both.
    indexed = scan_document(index, doc_id)
    incoming = {r["meta"]["chunk_id"]: r for r in records}

    # ------------------------------------------------------------------
    # FOUR CASES
    # ------------------------------------------------------------------

    # NEW TEXT. Not in the index, so it has never been embedded.
    #     -> embed and upsert. This is the only case that costs money.
    added = [
        r
        for cid, r in incoming.items()
        if cid not in indexed
    ]

    # TEXT THAT IS GONE. In the index, absent from the document.
    #     -> delete. Leave it and a query can return a passage that no longer
    #        exists in the source, with no way for the reader to tell.
    removed = [
        cid
        for cid in indexed
        if cid not in incoming
    ]

    # SAME TEXT, DIFFERENT PLACE.
    #
    # The subtle one, and the case people miss. A paragraph was inserted
    # above this chunk, so its page or position changed — but the text did
    # not, so the vector is still exactly correct.
    #
    #     -> rewrite the metadata. NO embedding call at all.
    #
    # Without this case, inserting one paragraph on page 2 re-embeds
    # everything after it.
    shifted = [
        cid
        for cid in incoming
        if cid in indexed and (
            int(indexed[cid].get("position", -1)) != incoming[cid]["meta"]["position"]
            or int(indexed[cid].get("page", -1)) != incoming[cid]["meta"]["page"]
        )
    ]

    # The fourth case is UNCHANGED — same id, same position. It does not
    # appear above because nothing is done with it. That is the point: on a
    # typical re-ingestion it is most of the document.

    plan = {"indexed": len(indexed), "incoming": len(incoming), "added": len(added),
            "removed": len(removed), "unchanged": len(incoming) - len(added),
            "metadata_only": len(shifted)}
    print(f"  {plan}", flush=True)

    if added:
        # Batch size from measured payload against Pinecone's 2 MB request limit
        # rather than a fixed count: at 1536 dimensions a fixed 100 fits
        # comfortably, and at 3072 with large metadata the same 100 starts failing.
        # ------------------------------------------------------------------
        # BATCH SIZE IS MEASURED, NOT FIXED
        # ------------------------------------------------------------------
        #
        # A fixed count works until something changes:
        #
        #     100 vectors at 1536 dims, small metadata   -> fits comfortably
        #     100 vectors at 3072 dims, large metadata   -> exceeds the limit
        #
        # And the failure arrives as a request-too-large error partway
        # through a corpus, after some documents have already been written.
        #
        # So: measure one vector, divide the request limit by it, and keep
        # 20% headroom for the request envelope itself.
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
    # ------------------------------------------------------------------
    # METADATA-ONLY UPDATES
    # ------------------------------------------------------------------
    #
    # No vector is sent and no embedding call is made. The vector already in
    # the index is correct — only where the chunk sits in the document
    # changed.
    for cid in shifted:
        index.update(id=cid, set_metadata=incoming[cid]["meta"], namespace=NAMESPACE)

    # Deletes last. Upserting the new version before removing the old means there is
    # never a window where the document is missing from the index.
    # ------------------------------------------------------------------
    # DELETES LAST
    # ------------------------------------------------------------------
    #
    # Order matters. Upserting the new version before removing the old means
    # there is never a moment when the document is missing from the index.
    #
    # Delete first and a query landing in that window returns nothing, with
    # no error and no indication that anything is being rebuilt.
    for i in range(0, len(removed), 200):
        index.delete(ids=removed[i:i + 200], namespace=NAMESPACE)

    return plan
