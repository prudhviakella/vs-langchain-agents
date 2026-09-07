---
name: rag-pipeline-builder
description: Build a production document RAG ingestion pipeline — PDF parsing with Docling, structure-aware chunking, embedding, and vector store sync with incremental updates. Use this skill whenever the user wants to ingest PDFs or documents into a vector database, build or improve a RAG pipeline, chunk documents for retrieval, extract tables or figures from PDFs for search, or fix a RAG system that returns poor results. Also use it when the user mentions Docling, HybridChunker, chunking strategy, document parsing quality, or asks why their RAG retrieval is missing content — these are all signs the underlying ingestion is the real problem, even when the question sounds like it is about retrieval or embeddings.
---

# Building a document RAG ingestion pipeline

This skill encodes a pipeline built and debugged against a real 20-document,
2,348-page corpus of clinical trial protocols and financial research reports.
Every non-obvious decision here came from a measured failure, and the
measurements are recorded so you can tell which decisions are load-bearing and
which are tuning.

## The one idea that matters most

**A parser is not a chunker, and a chunker is not a retrieval strategy.**

Docling gives you typed elements with page provenance. `HybridChunker` splits
those into token-bounded chunks. Neither one decides what is *worth indexing*,
what belongs *together*, or what is *too small to stand alone*. Those are your
decisions, they live in your code, and skipping them is why most RAG pipelines
retrieve badly.

Concretely, the chunker will not:

- filter anything (a page-number chunk and a real paragraph are equal to it)
- enforce a minimum size (a 6-token element becomes a 6-token chunk)
- respect element type when merging (it will weld prose to a table)

So the pipeline is: parse → **your policy passes** → embed → sync.

## Build order

Build in this order. Each stage is verifiable before the next one exists, which
matters because a chunking bug is nearly invisible once it is buried under
embeddings.

1. `config.py` — settings, read at import time
2. `clients.py` — API clients, probe embedding dimensions once
3. `docling_io.py` — parse the PDF, describe figures
4. `headings.py` — demote false headings *before* chunking
5. `tables.py` — serialize tables, detect broken structure
6. `chunking.py` — the policy passes (the heart of it)
7. `embedding.py` — batch by token budget, cache by content hash
8. `index.py` / `sync.py` — vector store, incremental updates
9. `inspect.py` — the reports that make failures visible
10. `retrieval.py` — query, rerank, expand
11. `ingest_all.py` — batch runner, crash-safe

Read `references/architecture.md` for what each module owns and why the
boundaries are where they are.

## The decisions that came from real failures

These are not style preferences. Each one is a bug that was measured, and
reverting any of them reintroduces a specific, known failure.

### Turn `merge_peers` off

`HybridChunker(merge_peers=True)` merges consecutive chunks that share a heading
path and fit the budget. It never checks element type. Measured on page 1 of a
research report: one 1007-token chunk containing four paragraphs, a figure
description, an eight-person contact table, and a legal disclaimer — because all
of it sat under one heading.

Write your own merge with a type check instead. Prose merges with prose;
tables and figures never merge with anything.

### Demote false headings before chunking, not after

A heading does three things at once: it is a chunk boundary, it is the entire
merge predicate, and it is prepended into the embedded text of every chunk
beneath it. One wrong heading therefore corrupts all three.

Measured: `'Exhibit 6:'`, `'through 2024.'`, and a click-here disclaimer were all
classified as `SECTION_HEADER` on a 7-page report. Three distinct root causes —
a bare numbered label, a page-break sentence split whose remainder looked like a
heading, and a positional misfire right after a real heading.

Rules that generalize: ends in sentence punctuation, starts with a conjunction,
matches `^(exhibit|figure|table)\s*\d+:?$`, exceeds a length ceiling. Rules based
on a document's specific visual layout do not generalize — write them anyway when
a corpus needs them, but expect to add one per new document type.

### Drop page furniture, and count what you dropped

Measured: 20 of 55 text chunks on one report were attribution lines and a logo
glyph. Six were byte-identical — six copies of one point in vector space, able to
occupy an entire top-k between them.

Gate every pattern behind a token ceiling (nothing over ~15 tokens is furniture)
and anchor patterns at string start. Return a *reason*, not a bool, and print it.
A silent deletion is one nobody audits.

### Batch embeddings by token budget, not by count

This one failed twice, and the second failure is instructive. Slicing by count
(`range(0, len(texts), 512)`) sends up to 512 × chunk_size tokens per request.
Measured: two documents in a 20-document run failed with
`Requested 458743 tokens, max 300000` — no individual chunk was oversized; the
*aggregate* was.

The subtle part: it is not enough to write a token-aware batching function. Check
that every code path actually calls it. The first fix lowered a token ceiling in
a function the failing path never invoked.

### Content-addressed chunk ids make incremental sync possible

```
chunk_id = {doc_id}:{sha256(text)[:16]}:{occurrence}
```

`doc_id` scopes the hash (two PDFs sharing a disclaimer would otherwise collide,
and upsert is last-write-wins — one document silently deletes another's chunk).
The hash means an edit only changes ids where text changed. `occurrence`
disambiguates legitimately repeated text.

Verified: re-ingesting an unchanged document reports
`{'added': 0, 'removed': 0, 'unchanged': 34}`.

## The reports are not optional

Extraction and chunking fail *quietly*. A document parses, chunks, embeds, and
syncs with no error, and the retrieval is bad six weeks later for reasons nobody
can reconstruct.

Every real bug found in this pipeline was found by reading a report, not by an
exception. Build `inspect.py` early and make it print:

- element counts by type, per document
- captions linked / total figures
- headings that do not look like headings, with the text
- tables whose structure looks broken, with the reason
- chunk size distribution (median, count under 50 tokens, max)
- what was dropped, with a sample

Then read them. See `references/diagnostics.md`.

## Building it

Read `references/architecture.md` first for module boundaries. Then read the
reference file for whichever module you are building — each one contains the
interface, the design decisions specific to it, and the failure it prevents.

- `references/architecture.md` — module boundaries, data shapes, build order
- `references/chunking.md` — the policy passes, in detail (the hard part)
- `references/parsing.md` — Docling options, figure caching, heading cleanup
- `references/tables.md` — serialization, broken-structure detection, fallbacks
- `references/embedding-sync.md` — batching, caching, incremental sync
- `references/diagnostics.md` — what to report and why
- `references/retrieval.md` — query path, reranking, neighbour expansion

## Verifying as you go

Do not trust a stage until you have looked at its output on a real document.

```python
# after chunking, before embedding — always
sizes = sorted(r["meta"]["n_tokens"] for r in records)
print(f"median {sizes[len(sizes)//2]}, under 50: {sum(1 for s in sizes if s < 50)}")
print(Counter(r["meta"]["content_type"] for r in records))
```

A median under ~100 tokens means the merge passes are not working. A large count
under 50 tokens means the drop filter or the floor is missing. Both are invisible
after embedding.

## Scope and honesty

This pipeline was tuned on clinical protocols and financial reports — dense,
table-heavy, born-digital PDFs. The *architecture* generalizes. The specific
regex rules in heading cleanup do not; expect to extend them per corpus.

Alternatives were tested and measured, not assumed: a different layout model lost
2 of 3 tables; a VLM pipeline lost all 3 and degenerated into repetition on a
dense page; MinerU handled some table shapes better and flattened others. If the
user asks about switching parsers, the honest answer is that it is a real
trade-off, not an upgrade — see `references/parsing.md`.

**What this pipeline does not have, and should**: a retrieval eval set. Every
tuning decision above rests on chunk statistics rather than measured answer
quality. If the user is building this fresh, twenty questions with known-correct
chunk ids is the highest-value thing they can add.
