"""Vector index access, and the manifest that keeps both halves in agreement.

    ingestion                          retrieval
        |                                  |
        v                                  v
    write_manifest()  --> manifest --> read_manifest()
                                           |
                                     settings match?
                                        |     |
                                       yes    no
                                        |     |
                                        v     v
                                     proceed  RAISE

WHY IT RAISES RATHER THAN WARNS

Query an index with the wrong embedding model and nothing breaks. You get
results, with scores that look reasonable, that are meaningless.

A warning gets read once and ignored. This failure has no other symptom, so it
has to stop you.


The manifest is the seam between ingestion and retrieval. Ingestion records how the
index was built; retrieval reads it and refuses to run if its own configuration
disagrees. Without it the mismatch is invisible: the query returns results, the
scores look reasonable, and the answers are quietly wrong.
"""

import json
import time
from datetime import datetime

from pinecone import ServerlessSpec

from .clients import EMBED_DIMS, pc
from .config import (CLOUD, EMBED_MODEL, INDEX_NAME, MANIFEST_PATH, METRIC,
                     NAMESPACE, REGION, CHUNK_TOKENS, VISION_MODEL)

def open_index(create: bool = False):
    """Connect to the Pinecone index, optionally creating it.

    The dimension check is the important part. Querying an index built with a
    different embedding model does not fail — Pinecone happily computes similarity
    between vectors of the same width regardless of what produced them, and returns
    ranked results that mean nothing. When the widths differ you at least get an
    error; when they match you get silent nonsense. Failing loudly here is cheap.
    """
    if not pc.has_index(INDEX_NAME):
        if not create:
            raise RuntimeError(f"{INDEX_NAME} does not exist; run ingestion first")
        pc.create_index(
            name=INDEX_NAME, dimension=EMBED_DIMS, metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        # Creation is asynchronous. Never busy-wait without a deadline: a failed
        # creation would otherwise spin silently forever.
        deadline = time.time() + 180
        while not pc.describe_index(INDEX_NAME).status.get("ready", False):
            if time.time() > deadline:
                raise TimeoutError(f"{INDEX_NAME} not ready after 180s")
            time.sleep(2)

    description = pc.describe_index(INDEX_NAME)
    if description.dimension != EMBED_DIMS:
        raise ValueError(
            f"{INDEX_NAME} has dimension {description.dimension}; "
            f"{EMBED_MODEL} produces {EMBED_DIMS}"
        )
    return pc.Index(INDEX_NAME)


# Fields needed to reconcile an index with a re-parsed document. Fetching full
# metadata and vectors for every chunk is what makes a naive sync transfer megabytes
# per document; this keeps it to what the comparison actually reads.
SYNC_FIELDS = ("chunk_id", "content_hash", "position", "page")


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


def scan_document(index, doc_id: str, namespace: str = NAMESPACE,
                  fields: tuple[str, ...] = SYNC_FIELDS) -> dict[str, dict]:
    """Indexed metadata for one document, keyed by chunk_id.

    Pinecone has no way to fetch metadata without vectors, so the vectors come back
    whether or not they are wanted. Discarding each page as it arrives keeps peak
    memory at one fetch batch rather than the whole document — the difference
    between a few megabytes and a few hundred on a long protocol.
    """
    found: dict[str, dict] = {}
    for listing in index.list(prefix=doc_id + ":", namespace=namespace):
        # `listing` is whatever this SDK version yields; _ids_from turns it into
        # plain strings. Passing it to fetch() directly is what produces the
        # "ID length exceeds 512 characters" error with the whole response
        # object printed as the offending id.
        id_batch = _ids_from(listing)
        if not id_batch:
            continue
        fetched = index.fetch(ids=id_batch, namespace=namespace).vectors
        for vector in fetched.values():
            meta = vector.metadata
            found[meta["chunk_id"]] = {k: meta[k] for k in fields if k in meta}
        # Dropped per batch, so peak memory is one fetch rather than the
        # whole document.
        del fetched
    return found


def write_manifest(doc_id: str, n_chunks: int, extra: dict | None = None) -> dict:
    """Record how the index was built, for the retrieval side to verify against.

    Accumulates one entry per document rather than overwriting, so a corpus ingested
    over several runs ends up with a complete inventory.
    """
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    manifest.update({
        "index_name": INDEX_NAME,
        "namespace": NAMESPACE,
        "metric": METRIC,
        "embed_model": EMBED_MODEL,
        "embed_dims": EMBED_DIMS,
        "chunk_tokens": CHUNK_TOKENS,
        "vision_model": VISION_MODEL,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    manifest.setdefault("documents", {})[doc_id] = {"chunks": n_chunks, **(extra or {})}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest() -> dict:
    """Load the manifest and verify this process agrees with how the index was built.

    Raises rather than warns. A warning would be read once and ignored, and the
    failure it prevents produces no other symptom — plausible rankings, plausible
    scores, wrong answers.
    """
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"{MANIFEST_PATH} not found; run ingestion first")
    manifest = json.loads(MANIFEST_PATH.read_text())
    for key, current in (("embed_model", EMBED_MODEL), ("embed_dims", EMBED_DIMS),
                         ("index_name", INDEX_NAME)):
        if manifest.get(key) != current:
            raise ValueError(
                f"manifest {key}={manifest.get(key)!r} but this process has "
                f"{current!r}; re-ingest or align the configuration"
            )
    return manifest


def list_rerank_models() -> list[dict]:
    """Rerankers this Pinecone account can currently call.

    Model names move between SDK versions, and a wrong name fails at query time
    rather than at import. Printing the live list means students choose from what
    exists rather than from what was true when this was written.
    """
    try:
        models = pc.inference.list_models(type="rerank")
    except Exception as exc:
        return [{"error": f"could not list models: {exc}"}]
    return [{"model": getattr(m, "model", getattr(m, "name", "?")),
             "provider": getattr(m, "provider_name", getattr(m, "provider", "")),
             "max_docs": getattr(m, "max_batch_size", None),
             "max_tokens": getattr(m, "max_sequence_length", None)}
            for m in models]
