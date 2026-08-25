"""Configuration and primitives shared by ingestion and retrieval.

Everything here is either a setting both halves of the pipeline must agree on, or a
primitive both halves call. Keeping them in one module is what stops the two from
drifting: if ingestion embeds with one model and retrieval queries with another, the
index returns well-formed results with plausible scores that happen to be
meaningless, and nothing errors.

The manifest at the bottom is the guard against exactly that.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"   # what chunks and queries are embedded with
VISION_MODEL = "gpt-4o-mini"             # figure descriptions, during extraction
RERANK_MODEL = "bge-reranker-v2-m3"      # cross-encoder, hosted by Pinecone
LLM_MODEL = "gpt-4o-mini"                # generation and query rewriting

# ─────────────────────────────────────────────────────────────────────────────
# Index
# ─────────────────────────────────────────────────────────────────────────────
INDEX_NAME = "rag-docs"
NAMESPACE = ""            # one shared namespace: cross-document search stays possible
CLOUD, REGION = "aws", "us-east-1"

# Fixed at index creation and not changeable afterwards without rebuilding.
# For the unit-normalised vectors these models return, dot product and cosine rank
# identically — so this costs nothing today. It is additionally required for
# sparse-dense hybrid retrieval, so choosing cosine would quietly foreclose that.
METRIC = "dotproduct"

# Stamped on every chunk and applied as a filter on every query. Written from the
# first ingestion even when there is only one group, because adding an access field
# after the fact means re-embedding the corpus.
ACCESS_GROUPS = ["public"]

# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────
# Retrieval-quality target. Smaller chunks match more precisely; larger ones carry
# more of the answer in one vector.
#
# 1024 is the default because table summaries carry the semantic weight for tabular
# content, so fragments no longer have to stand alone and can stay small enough to
# keep similarity scores sharp. Above roughly 2000 tokens a single vector averages
# too many concepts together and blunts both retrieval and reranking.
#
# This is a hyperparameter, not a constant. Sweep it against the labelled set:
#   CHUNK_TOKEN_TARGET=512 python ingest.py --pdf x.pdf
CHUNK_TOKEN_TARGET = int(os.getenv("CHUNK_TOKEN_TARGET", "1024"))

# Hard sequence limits, from provider documentation. Exceeding one does not raise:
# the API accepts the input, embeds the first N tokens, and returns a valid-looking
# vector for part of a chunk. The chunk budget is capped by this for that reason.
SEQUENCE_LIMITS = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}

# ─────────────────────────────────────────────────────────────────────────────
# Documented API limits
#
# Batch sizes are derived from these rather than hard-coded, because a fixed count
# that fits comfortably at 1536 dimensions silently starts failing at 3072.
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_METADATA_BYTES = 40 * 1024      # per vector
PINECONE_REQUEST_BYTES = 2 * 1024 * 1024  # per upsert request
OPENAI_EMBED_MAX_INPUTS = 2048           # per embeddings request
OPENAI_EMBED_MAX_TOKENS = 300_000        # per embeddings request

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache")
EMBED_CACHE = CACHE_DIR / "embeddings"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
CACHE_DIR.mkdir(exist_ok=True)
EMBED_CACHE.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Clients and derived constants
# ─────────────────────────────────────────────────────────────────────────────
client = OpenAI()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
ENCODING = tiktoken.encoding_for_model(EMBED_MODEL)

CHUNK_TOKENS = min(CHUNK_TOKEN_TARGET, SEQUENCE_LIMITS.get(EMBED_MODEL, CHUNK_TOKEN_TARGET))

# Probed rather than assumed. Dimension varies with the model and, for
# Matryoshka-capable models, with the requested `dimensions` parameter — so a
# hard-coded 1536 becomes wrong the moment anyone changes EMBED_MODEL.
EMBED_DIMS = len(client.embeddings.create(model=EMBED_MODEL, input=["probe"]).data[0].embedding)


def slugify(text: str) -> str:
    """Filesystem- and identifier-safe form of arbitrary text.

    Used for document ids, cache filenames and index names, so it has to be stable:
    the same input must always produce the same slug, or cached parses stop being
    found and documents get re-ingested under a second id.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


# ═════════════════════════════════════════════════════════════════════════════
# Embedding
# ═════════════════════════════════════════════════════════════════════════════

def _cache_path(digest: str) -> Path:
    """Where a single embedding is cached on disk.

    Sharded by the first two characters of the digest. A corpus of a few thousand
    documents produces hundreds of thousands of these files, and most filesystems
    degrade badly when that many land in one directory — directory lookups go
    linear, and `ls` becomes unusable for debugging.

    The model name is part of the path, not just the digest, so switching embedding
    models cannot silently return vectors from the previous one.
    """
    shard = EMBED_CACHE / slugify(EMBED_MODEL) / digest[:2]
    shard.mkdir(parents=True, exist_ok=True)
    return shard / (digest + ".npy")


def _batches(texts: list[str]) -> list[list[int]]:
    """Group text indices into requests within both API limits.

    The embeddings endpoint bounds a request two ways — by number of inputs and by
    total tokens — and a batch that respects one can still violate the other. A
    thousand short strings hit the input cap; fifty long chunks hit the token cap.
    Closing the batch on whichever comes first is why this is not just a fixed size.

    Returns indices rather than the texts themselves so the caller can write results
    back into the right positions.
    """
    groups, current, tokens = [], [], 0
    for i, text in enumerate(texts):
        cost = len(ENCODING.encode(text))
        if current and (len(current) >= OPENAI_EMBED_MAX_INPUTS
                        or tokens + cost > OPENAI_EMBED_MAX_TOKENS):
            groups.append(current)
            current, tokens = [], 0
        current.append(i)
        tokens += cost
    if current:
        groups.append(current)
    return groups


def embed(texts: list[str], use_cache: bool = True, verbose: bool = False) -> np.ndarray:
    """Embed texts, caching each one by (model, text).

    Returns a float32 array rather than nested lists. At corpus scale that matters:
    a Python list of floats costs roughly seven times the memory of the same numbers
    packed in an array, because every float is a separate boxed object with its own
    header and pointer.

    The cache is what makes experimentation free. Re-running after a chunking or
    retrieval change re-embeds nothing, and boilerplate shared across documents is
    embedded once for the whole corpus.
    """
    digests = [hashlib.sha256((EMBED_MODEL + "\x00" + t).encode()).hexdigest()[:24]
               for t in texts]
    out = np.empty((len(texts), EMBED_DIMS), dtype=np.float32)
    missing = list(range(len(texts)))

    if use_cache:
        missing = []
        for i, digest in enumerate(digests):
            path = _cache_path(digest)
            if path.exists():
                out[i] = np.load(path)
            else:
                missing.append(i)

    for group in _batches([texts[i] for i in missing]):
        # _batches indexes into the *missing* list, so map back to real positions.
        indices = [missing[j] for j in group]
        for attempt in range(4):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL, input=[texts[i] for i in indices]
                )
                break
            except Exception:
                # Rate limits and transient network errors both land here. Four
                # attempts with exponential backoff covers a rate-limit window
                # without hanging a batch job for minutes on a real outage.
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        for i, item in zip(indices, response.data):
            out[i] = item.embedding
            if use_cache:
                np.save(_cache_path(digests[i]), out[i])

    if verbose:
        print(f"{len(texts)} texts | {len(texts) - len(missing)} cached "
              f"| {len(missing)} embedded")
    return out


def embed_stream(texts: list[str], batch: int = 512, use_cache: bool = True):
    """Yield (offset, vectors) so the caller can upsert as it goes.

    Ingesting a large corpus should never hold every vector in memory at once. A
    250-page document produces thousands of chunks, and materialising all of them
    before the first upsert makes peak memory a function of document size — which
    is exactly what decides whether a Fargate task fits its memory limit.

    Windowing keeps that flat regardless of how long the document is.
    """
    for start in range(0, len(texts), batch):
        yield start, embed(texts[start:start + batch], use_cache=use_cache)


# ═════════════════════════════════════════════════════════════════════════════
# Index
# ═════════════════════════════════════════════════════════════════════════════

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


def scan_document(index, doc_id: str, namespace: str = NAMESPACE,
                  fields: tuple[str, ...] = SYNC_FIELDS) -> dict[str, dict]:
    """Indexed metadata for one document, keyed by chunk_id.

    Pinecone has no way to fetch metadata without vectors, so the vectors come back
    whether or not they are wanted. Discarding each page as it arrives keeps peak
    memory at one fetch batch rather than the whole document — the difference
    between a few megabytes and a few hundred on a long protocol.
    """
    found: dict[str, dict] = {}
    for id_batch in index.list(prefix=doc_id + ":", namespace=namespace):
        page = index.fetch(ids=id_batch, namespace=namespace).vectors
        for vector in page.values():
            meta = vector.metadata
            found[meta["chunk_id"]] = {k: meta[k] for k in fields if k in meta}
        del page
    return found


# ═════════════════════════════════════════════════════════════════════════════
# Manifest
#
# The seam between the two halves of the pipeline. Ingestion records how the index
# was built; retrieval reads it and refuses to run if its own configuration
# disagrees. Without this the mismatch is invisible: the query returns results, the
# scores look reasonable, and the answers are quietly wrong.
# ═════════════════════════════════════════════════════════════════════════════

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
