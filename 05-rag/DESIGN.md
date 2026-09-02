# Why the chunking code looks like this

Every decision below was measured on a real document. The code says what
happens; this says why, and what happened when it was done the other way.

---

## `merge_peers=False`

Docling's own merge looks like the thing you want and is not. From the source:

```python
if headings == current_headings and self._count_chunk_tokens(candidate) <= self.max_tokens:
```

Two conditions, both mechanical: the heading paths must be **equal**, the
chunks must be **consecutive**, and the result must fit. **Element type is
never consulted.** "Peers" does not mean siblings in the tree; it means an
equal heading path.

A section heading owns its prose, its exhibits and its tables alike, so they
all carry the same path and all merge. Measured on page 1 of a research
report with it ON — **one chunk, 1007 tokens, labelled "table"**, containing:

- the section heading
- four paragraphs of prose
- the Exhibit 1 caption and figure description
- a contact table of eight names, emails and phone numbers
- the legal disclaimer

It also runs *inside* `chunker.chunk()`, before our figure pass can see
anything, so Exhibit 1's description was indexed twice — once inside the
merged chunk, once as its own record.

`_merge_prose` does the same merge with the type check Docling does not offer.

**There is no `min_tokens` parameter.** Some documentation mirrors list one.
It does not exist in the source, and pydantic ignores unknown keyword
arguments silently — so passing it looks like it worked and does nothing.

---

## The drop filter

On a 7-page research report, **20 of 55 text chunks were page furniture** —
attribution lines and the publisher's logo glyph, read as a text element on
every page. Six of the twenty were byte-identical to each other. Identical
text embeds to an identical vector, so those were six copies of one point in
the index, collectively able to occupy an entire top-k.

**This is not a parsing bug.** The page really does say
`Source: Morgan Stanley Research`. Docling read it correctly. There is no
parse setting that suppresses it and there should not be — the extraction
report is supposed to show everything on the page. The judgement about what
deserves a vector belongs at the point where vectors are decided.

**The risk, stated plainly:** a pattern that is too greedy deletes content and
nothing downstream can tell. Hence the hard token ceiling, patterns anchored
at the start of the string rather than searched anywhere in it, and every drop
counted and printed with a sample.

---

## `MERGE_ACROSS_EXHIBITS`

Adjacent-only merging almost never fires on a document whose prose is
interleaved with exhibits:

```
paragraph   figure   caption   paragraph   figure   paragraph
```

Every run is length one. Measured on the research report:

| | records | prose median |
|---|---|---|
| adjacent only | 47 | 61 tokens |
| reaching past exhibits | 37 | 171 tokens |

**What it costs:** a merged record's page range spans exhibits it does not
contain, and `prev_id` / `next_id` no longer step through strict document
order for those records. Retrieval uses those links only for neighbour
expansion, so the damage is small — but it is real.

The search stops at the first entry under a **different heading**, not at the
first non-text entry. A heading boundary still ends everything.

---

## `MIN_CHUNK_TOKENS` — the floor

The chunker never merges across a heading, and that is right when headings
mean sections. On a form it is not.

Measured on a 15-page IRB protocol with 57 headings:

- **43 of 48 heading paths owned exactly one record**
- median 86 tokens, 18 records under 30

```
'Section N:  Sample Collection\nNone'                    8 tokens
'A5.  Funding Source:\nBaylor College (Internal Only)'  17 tokens
```

Each is a complete answer to one question, which is the argument for leaving
them alone. Against that: retrieval returns *k* of them, and k fragments of 15
tokens is 200 tokens of context for the model to answer from. At corpus scale
it is thousands of near-identical vectors whose embedding is mostly the
heading.

**Validated across three documents at 150:**

| | floor 0 | floor 150 |
|---|---|---|
| 7-page research report | 34 records, median 260 | 33 records, median 297 |
| 15-page IRB form | 63 records, median 92 | 32 records, median 302 |
| 97-page protocol | 257 records, median 130 | 168 records, median 274 |

It merges only while the accumulating record is under the floor, so a document
whose chunks already clear it is untouched. That is what makes one value safe
across a mixed corpus.

**What it costs:** it joins sections that are genuinely separate. A record can
answer about funding AND about institution, and a query for one retrieves
both. On the 97-page protocol, three-quarters of records still span zero or
one section; the worst case spanned five consecutive subsections of section 9,
all about analysis sets — arguably the right neighbourhood rather than
contamination.

Every heading crossed is written into the body, so its context is still
embedded. `headings` metadata keeps the first path only, so `section_id` and
filtering stay stable.

---

## One record per figure

Each figure description is a self-contained fact about a different chart.
Packing several into one vector produces a vector representing none of them,
and hands the model five descriptions when it asked about one.

### The caption test had to be widened

An earlier version required **every** item in the chunk to be a picture. That
excluded exactly the well-formed case: when the layout model *does* link a
caption to its figure, the serializer emits both as one chunk, so the strict
test rejected it and the figure fell through to the generic path — where the
record text comes from the serializer, which renders the caption and the
classification but **not** a description attached after the parse.

Measured: **10 of 16 figures came out as `Exhibit 4: ... \n\n Bar chart`** —
30 tokens, no description — while the 6 *unlinked* figures got full ones. The
better the parse, the worse the record.

A table still disqualifies the chunk. Tables carry values that must be indexed
as themselves.

### Captions come from the document

`item.caption_text(doc)`. Without it the caption becomes a separate 25-token
chunk saying `Exhibit 18: Our Conceptual Roadmap for AI Developments` and
nothing else, while the figure it names carries no exhibit number — so neither
is retrievable by the one string a reader would search for.

The vision model sometimes reads the exhibit title off the image, which is why
some figure records already contain it. That is luck, not linkage.

---

## Content-addressed chunk ids

```
chunk_id = {doc_id}:{sha256(text)[:16]}:{occurrence}
```

| Part | Why |
|---|---|
| `doc_id` | scopes the hash. Two PDFs sharing a boilerplate disclaimer would produce the same id, and upsert is last-write-wins — one document silently deletes the other's chunk, with no error anywhere. |
| `hash` | a positional id changes for every chunk after an edit, forcing a full re-embed when one paragraph changed. A hash changes only where the text changed. |
| `occurrence` | the same text can legitimately repeat — a disclaimer on every page. Without a counter all copies collapse into one record and the other page numbers are lost. |

This single decision is what makes the incremental sync possible. **Verified:**
a second ingest of an unchanged document reported `added 0, removed 0,
unchanged 34`.

What `occurrence` does **not** solve: repeated text that is worthless stays
repeated, once per copy. That is the drop filter's job, and it runs first.

---

## A large table is not forced into one chunk

```
logical table
      |
 +----+----+
 v    v    v
chunk chunk chunk        fragments, each its own record
 +----+----+
      |
 same table_id
      |
      v
 ONE summary            an ADDITIONAL record
```

The fragments hold the exact values. The summary holds the words someone would
search for. Neither replaces the other, and `table_id` is how retrieval walks
from a matched summary to the rows carrying the numbers.

The summary is read from the complete grid on the document object, not
stitched back together from fragments.

---

## Sizing in tokens, not characters

Characters are a proxy that fails silently: the API accepts an oversized
input, embeds the first N tokens, and returns a valid-looking vector for half
a chunk. The tokenizer used is the embedding model's own.

---

## What is deliberately not here

**Parent-section retrieval.** `section_id` is written on every record and
nothing reads it. It is written now because adding it later means re-embedding
the corpus.

**Heading hierarchy.** Docling assigns every section header the same level, so
`8.2` and `8.3.2` are siblings of `8`. The numbering carries a real hierarchy
the parse discards. Recoverable from the text; not done.

**A retrieval eval.** Every number above describes the *shape* of an index and
says nothing about whether it answers questions. `PROSE_TARGET_TOKENS`,
`MERGE_ACROSS_EXHIBITS`, the floor at 150 — all of them rest on chunk
statistics and judgement. Twenty questions with known-correct chunk ids would
turn each into a measurement. That is the most valuable thing still missing.
