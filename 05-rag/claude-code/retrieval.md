# Retrieval

## The query path

```
question
  |
  v
embed          same model as ingestion — verify, do not assume
  |
  v
dense search   retrieve a WIDE pool (~50), not the final k
  |
  v
rerank         cross-encoder over the pool
  |
  v
with_table_rows()   a matched table summary pulls in its own rows by table_id
  |
  v
with_neighbours()   a SHORT or truncated result pulls in prev/next
  |
  v
assemble       fill the context budget, in reading order
  |
  v
generate
```

## Retrieve wide, then rerank

Dense retrieval over a wide pool then reranking beats retrieving k directly. The
embedding model is fast and approximate; the cross-encoder is slow and accurate.
Use each for what it is good at.

## Expansion by exact lookup, not similarity

Both expansion steps fetch *specific ids*. A known chunk_id is either in the
store or it is not — asking a vector search to find it pays for a rank that is
not in question. Use a filtered query with `$in` on the id and a zero vector.

### `with_table_rows`

A table summary is written to be *findable* — it holds the words someone would
search for. The rows hold the exact values. When the summary matches, pull the
rows by `table_id` so the answer has the numbers.

### `with_neighbours` — the structural alternative to overlap

Overlap exists because naive splitters cut mid-fact. This pipeline's cuts land on
element and sentence boundaries, so the usual justification mostly does not
apply — and overlap manufactures near-duplicate vectors everywhere, which
directly contradicts the drop filter's purpose.

Instead, store `prev_id`/`next_id` at ingestion and expand *at query time*, only
for results that need it:

```python
ids_needed = {r[field] for r in results
              for field in ("prev_id", "next_id")
              if (r["n_tokens"] < min_tokens or r.get("truncated")) and r.get(field)}
```

The cost is paid for the rare question that needs it, not on every chunk in the
corpus whether or not it is ever used.

**The trap**: it is easy to write `prev_id`/`next_id` into metadata at ingestion,
carry them on every result, and never actually read them. Grep for the field
names in the retrieval module — if they appear only in a field list, the
mechanism is dead code.

**Second trap**: a field must survive the metadata-flattening step to be
checkable. A `truncated` flag that is not in the copied-fields list is always
`None` at query time, so a condition testing it can never fire.

Use `setdefault` when merging expanded results so a neighbour that also matched
on its own merit keeps its real rerank score instead of being overwritten with
zero.

## Filters

`headings` stored as a list supports `$in`. Stored as a joined string it supports
nothing.

Filter by `content_type` to answer "only tables" or "only figures". Filter by
`doc_id` to scope to one document. Both are free at query time and impossible to
add later without re-ingesting.

## Assemble in reading order

Sort the final set by position before building the context. A model reading
passages in retrieval-score order is reading them shuffled.

Budget the context explicitly and report what was dropped.

## The thing that is missing

Everything above is reasoned, not measured. Without an eval set — questions with
known-correct chunk ids — there is no way to tell whether reranking helps, whether
neighbour expansion helps, or whether a chunking change improved or degraded
retrieval.

Twenty questions with recorded correct answers is a few hours of work and turns
every future tuning decision from an argument into a measurement. It is the
highest-value missing piece in most RAG pipelines, including this one.

Cover the known failure modes deliberately: a question whose answer is in a
table, one whose answer is a specific figure, one whose answer spans two chunks
(tests neighbour expansion directly), one about a figure with no linked caption.
