"""Embedding, with a cache keyed on content.

    texts[]
       |
       v
    hash each one            sha256(model + text)
       |
       v
    cached?  --yes-->  read from disk
       |
      no
       |
       v
    batch by token count and by count
       |
       v
    OpenAI embeddings API
       |
       v
    write to cache  ->  ndarray

THE CACHE KEY IS (model, text), AND THAT MATTERS

The key fully determines the result, so this cache cannot go stale. Contrast a
cache keyed on a filename, where changing a setting returns work made under the
old one and the change appears to have done nothing.

Returns a numpy array, NOT lists. Seven times less memory at corpus scale, and
`sync.py` calls `.tolist()` at the point of upsert.


The cache is what makes experimentation free. Re-running after a chunking or
retrieval change re-embeds nothing, and boilerplate shared across documents is
embedded once for the whole corpus.

Its key is (model, text), which fully determines the result — so unlike a cache
keyed on a filename, this one cannot return work made under different settings.
"""

import hashlib
import time
from pathlib import Path

import numpy as np

from .clients import EMBED_DIMS, client
from .config import (EMBED_CACHE, EMBED_MODEL, ENCODING, OPENAI_EMBED_MAX_INPUTS,
                     OPENAI_EMBED_MAX_TOKENS, slugify)

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
