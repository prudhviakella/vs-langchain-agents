"""Reading chunks back out of the vector index.

The graph is built from the same chunks the vector index holds, and joined to them
by `chunk_id`. Rather than importing the ingestion pipeline to re-derive them, this
reads them from the index directly.

That keeps the two packages independent: this module needs Pinecone credentials and
an index name, not the other package's source. If the ingestion pipeline changes how
it chunks, this keeps working, because the contract between them is the index — not
a function signature.
"""

import os

from . import config


def index():
    """Connect to the vector index, read only.

    A dimension check is not needed here — this module never writes vectors, and it
    reads metadata rather than computing similarity. The failure it would guard
    against cannot occur.
    """
    from pinecone import Pinecone

    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not client.has_index(config.INDEX_NAME):
        raise RuntimeError(
            f"index {config.INDEX_NAME!r} does not exist. Run the ingestion "
            "pipeline first, or set INDEX_NAME to an index that does.")
    return client.Index(config.INDEX_NAME)


def _ids_from(page) -> list[str]:
    """Pull plain id strings out of whatever `index.list()` yielded.

    The shape has changed across Pinecone SDK versions and is not stable:

        a list of str                    ["doc:abc:0", ...]
        a list of objects with .id       [ListItem(id="doc:abc:0"), ...]
        a ListResponse with .vectors     ListResponse(vectors=[ListItem(...)])

    Assuming one shape and passing the result straight to fetch() produces a
    confusing failure rather than an obvious one: the SDK stringifies whatever
    it was given, and the error is about an ID being longer than 512 characters
    — with the entire response object printed as the offending "id".

    Normalising here means every caller sees a list of strings, whichever
    version is installed.
    """
    # A ListResponse — unwrap to the items it holds.
    vectors = getattr(page, "vectors", None)
    if vectors is not None:
        page = vectors

    if isinstance(page, str):
        return [page]

    ids = []
    for entry in page or []:
        # A ListItem or similar, versus a plain string.
        ids.append(getattr(entry, "id", entry))
    return [i for i in ids if isinstance(i, str)]


def load_chunks(index, doc_id: str | None = None,
                namespace: str = config.NAMESPACE) -> list[dict]:
    """Every chunk's metadata, including its text, in reading order.

    Chunk ids from the ingestion pipeline are prefixed with the document id, so a
    prefix listing scopes this to one document. Passing None reads the whole index,
    which is what a multi-document graph wants.

    Vectors come back whether or not they are wanted — Pinecone has no metadata-only
    fetch — so each page is discarded as it is consumed rather than accumulated.
    """
    found: list[dict] = []
    prefix = f"{doc_id}:" if doc_id else None

    for listing in index.list(prefix=prefix, namespace=namespace):
        # `listing` is whatever this SDK version yields; _ids_from turns it into
        # plain strings. Passing it to fetch() directly produces the "ID length
        # exceeds 512 characters" error with the whole response object printed
        # as the offending id.
        id_batch = _ids_from(listing)
        if not id_batch:
            continue
        fetched = index.fetch(ids=id_batch, namespace=namespace).vectors
        for vector in fetched.values():
            found.append(dict(vector.metadata))
        del fetched

    # Reading order, so a graph built from consecutive chunks reflects the document.
    found.sort(key=lambda meta: (meta.get("doc_id", ""), int(meta.get("position", 0))))
    return found


def fetch_text(index, chunk_ids: list[str],
               namespace: str = config.NAMESPACE) -> dict[str, dict]:
    """Chunk metadata for a set of ids, keyed by id.

    This is the second half of the join: a traversal ends by naming chunk ids, and
    this turns them back into passages. Fetching by id is exact — no similarity
    involved — which is why the graph can be precise where a vector search is
    approximate.
    """
    if not chunk_ids:
        return {}
    # Pinecone caps ids per fetch; batching keeps a large traversal working.
    found: dict[str, dict] = {}
    for start in range(0, len(chunk_ids), 100):
        page = index.fetch(ids=chunk_ids[start:start + 100], namespace=namespace)
        for vector in page.vectors.values():
            found[vector.metadata["chunk_id"]] = dict(vector.metadata)
    return found
