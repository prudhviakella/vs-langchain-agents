# Diagnostics

Every real bug in this pipeline was found by reading a report, not by catching an
exception. Extraction and chunking fail quietly; this is what makes them audible.

## Two reports, two questions

**Extraction report** — "did the PDF parse correctly?"
Every element the parser produced, in reading order, with page numbers and types.

**Chunk report** — "is what I am about to embed any good?"
Every final chunk, exactly as it will be embedded.

Separate files, because they answer different questions and you read them at
different times.

## What the extraction report must show

```
pages=252  tables=109  tables_suspect=19  pictures=3  pictures_described=3
layout: 1/3 captions linked, 5/330 headers look false
```

- element counts by type
- **captions linked / total figures** — the single most diagnostic number for
  figure retrieval
- headings that do not look like headings, *with their text*
- tables whose structure looks broken, with the specific reason
- figures that produced no description, grouped rather than listed one per line

### Summarize repetition rather than listing it

`52 of 63 figures produced no description, on 52 page(s): p1, p9...` plus a note
that roughly one per page usually means a header wordmark. Fifty-two individual
lines is noise; one line with a hypothesis is a diagnosis.

## What the chunk report must show

```
480 records from 1165 chunks (92 of 109 tables summarised)
types: {'figure': 3, 'text': 280, 'table_summary': 92, 'table': 105}
size: median 353 tokens, 7 under 50, max 1024
dropped 5 chunk(s) as page furniture: {'a single glyph': 2, 'a page marker': 1}
merged 676 adjacent prose chunk(s) under a shared heading
merged 96 chunk(s) ACROSS a heading boundary to reach the 150-token floor
```

The numbers that actually diagnose problems:

- **median size** — under ~100 means the merge passes are not working
- **count under 50 tokens** — means the drop filter or floor is missing
- **what was dropped, with a sample** — a silent deletion is one nobody audits
- **cross-heading merge count** — a spike means the floor is doing more than
  intended

## Say what a number means, not just the number

```
NOTE: 2 figure(s) have no caption linked in the parse. They are retrievable by
description but not by exhibit number unless the vision model happened to read it.
```

versus `captions_linked: 1`. The first tells someone what to do about it.

## Report the impossible cases explicitly

Some conditions are not "unusual", they are "something is broken":

```
ERROR: 45 tables were extracted but none could be linked to a record. No
summaries were generated and table_id is empty on every record. Check
table_ref_of() against this docling version.
```

Zero table groups against a non-zero table count means the chunk-to-table link is
broken — not that the document lacks tables. Distinguishing those two in the
message saves an hour.

## Diagnostic ratios worth computing

**Entity mentions per unique entity** (if you extract entities): near 1.0 means
resolution is failing — every entity mentioned once, nothing connecting. A graph
of isolated nodes looks identical to "nothing to connect" unless you check.

**Captions linked / figures**: low is not always a bug. On clinical protocols,
`1/63` turned out to be correct — the source genuinely does not caption inline
diagrams. Verify by looking at what surrounds an unlinked figure before treating
it as a defect.

## An important negative result

A low number is not automatically a bug. Two cases from this corpus:

- `1/63 captions` — real property of the document type, not a parser failure
- welded table cells reproducing identically across two independent parsers —
  source document ambiguity, not a bug in either

Check whether the input actually contains what you expect before fixing the code
that reads it.

## Verify a stage before building on it

```python
sizes = sorted(r["meta"]["n_tokens"] for r in records)
print(f"median {sizes[len(sizes)//2]}, under 50: {sum(1 for s in sizes if s < 50)}")
print(Counter(r["meta"]["content_type"] for r in records))
```

Run this after chunking, before embedding, on every new corpus. Both failure
modes it catches are invisible once vectors exist.
