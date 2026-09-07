# Embedding and sync

## Batch by token budget — and check every code path uses it

The embeddings endpoint bounds a request two ways: number of inputs *and* total
tokens. A batch respecting one can still violate the other.

```python
def _batches(texts, max_inputs=None):
    cap = min(API_MAX_INPUTS, max_inputs or API_MAX_INPUTS)
    groups, current, tokens = [], [], 0
    for i, text in enumerate(texts):
        cost = len(ENCODING.encode(text))
        if current and (len(current) >= cap or tokens + cost > MAX_TOKENS):
            groups.append(current); current, tokens = [], 0
        current.append(i); tokens += cost
    if current: groups.append(current)
    return groups
```

**Leave a real margin below the API's hard limit.** Set at exactly the limit
there is no room to absorb a discrepancy between your tokenizer's count and the
API's own. Roughly 15-20% headroom is reasonable.

**The failure worth learning from**: a streaming variant of this sliced purely by
count (`range(0, len(texts), 512)`) and never consulted the token budget at all.
Two documents in a 20-document run failed with
`Requested 458743 tokens, max 300000` — no chunk was individually oversized.

The first attempted fix lowered the token ceiling in `_batches`, which the
failing path never called. **Writing a token-aware batcher is not enough; verify
the path that actually runs calls it.** Trace the real traceback, not the
function you assume is involved.

A streaming batcher should defer to the same function, treating its window size
as a count ceiling:

```python
def embed_stream(texts, batch=512, use_cache=True):
    for group in _batches(texts, max_inputs=batch):
        yield group[0], embed([texts[i] for i in group], use_cache=use_cache)
```

Groups must stay contiguous and in order — callers use the yielded offset to line
vectors up with records. Assert that if you change this.

## Cache by content hash

Key on `sha256(text + model)`. Re-embedding identical text is pure waste, and it
happens constantly during development.

## Size in tokens, never characters

Character counts fail *silently*: the API accepts an oversized input, embeds the
first N tokens, and returns a valid-looking vector for half a chunk. Use the
embedding model's own tokenizer.

## Content-addressed ids enable incremental sync

```
chunk_id = {doc_id}:{sha256(text)[:16]}:{occurrence}
```

| Part | Why |
|---|---|
| `doc_id` | scopes the hash — two PDFs sharing a boilerplate disclaimer would otherwise collide, and upsert is last-write-wins, so one document silently deletes the other's chunk with no error |
| `hash` | a positional id changes for every chunk after an edit, forcing a full re-embed when one paragraph changed |
| `occurrence` | the same text can legitimately repeat (a disclaimer on every page); without a counter all copies collapse into one record and the other page numbers are lost |

What `occurrence` does *not* solve: worthless repeated text stays repeated, once
per copy. That is the drop filter's job, and it runs first.

## Sync

```
ids currently in the store for this doc_id
ids the fresh parse produced
  |
  +-- in new, not in old      -> embed and upsert
  +-- in old, not in new      -> delete
  +-- in both                 -> untouched, no cost
```

Verified on a real re-run: `{'added': 0, 'removed': 0, 'unchanged': 34}`.

An unchanged document costing nothing is the whole point. If a re-ingest reports
everything as added, something upstream is non-deterministic — figure descriptions
are the usual culprit.

## Store the embedding model on every record

```python
"embed_model": EMBED_MODEL
```

A model mismatch between ingestion and query produces plausible rankings and no
error. Recording it lets retrieval refuse to query an index built with a
different one.

Check dimensions at index-open time too — probe the model once, compare against
the index's declared dimension, fail loudly if they differ.

## Batch runner

For a corpus, process smallest document first: a systemic problem surfaces in the
first few minutes rather than after the largest file has burned an hour.

Commit after each document and record status in a run log. A failure is recorded,
not raised — one bad document must not stop the other nineteen. Skip on
`(size, mtime)` so a resumed run is genuinely incremental.
