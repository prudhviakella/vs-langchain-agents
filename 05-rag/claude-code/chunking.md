# Chunking — the policy passes

This is the hard part and the part that decides retrieval quality.

## What HybridChunker actually does

Four stages. Only the last one combines anything.

```
1. HierarchicalChunker      one chunk per element
2. _split_by_doc_items      split oversized chunks BETWEEN elements
3. _split_using_plain_text  split on sentences when one element alone is too big
4. merge_chunks_with_matching_metadata    only if merge_peers=True
```

### Stage 1 exceptions worth knowing

One chunk per element, except:

- **page headers/footers** — removed before this stage entirely (furniture layer)
- **SECTION_HEADER / TITLE** — no chunk; becomes the heading path
- **consecutive LIST_ITEM** — merged into one chunk
- **a CAPTION linked to its figure/table** — stays with it

The caption case only fires when the layout model actually *linked* them. When
it fails to, you get an orphan caption chunk and a figure with no exhibit number
— retrievable by description but not by the string a reader would search for.

### Stage 2 splits on element boundaries, and needs a seam

A chunk splits only if it is *both* over budget *and* has more than one element.
One giant element passes through unchanged, still oversized, and stage 3 handles
it by splitting the text itself.

Note the budget counts the heading path too, not just the body — a chunk with a
40-token heading path has 984 tokens of room, not 1024.

### Stage 4 is type-blind — this is the bug

The predicate, verified against the installed source:

```python
if headings == current_headings and self._count_chunk_tokens(candidate) <= self.max_tokens:
```

Two conditions. Element type is never consulted. `captions` is not consulted
either, despite what some documentation says.

Set `merge_peers=False` and write your own.

## The passes to write

```
chunks
  |
  v
_to_entries()      classify, drop furniture, mark figure slots
  |
  v
_merge_prose()     merge adjacent text under ONE heading
  |
  v
_apply_floor()     small records absorb forward, ACROSS headings (optional)
  |
  v
_to_records()      figure slots expand to one record per figure
  |
  v
_table_summaries() one summary per table — extra records, not replacements
  |
  v
_finalise()        reading order, prev/next links, size limits
```

### `_to_entries` — classify, filter, defer

Three jobs: decide what each chunk *is*, drop what is not worth a vector, and
mark figures for special handling later.

The figure-slot test:

```python
pictures = [i for i, label in zip(items, labels) if "picture" in label]
beside = [label for label in labels if "picture" not in label]
if pictures and all("caption" in label for label in beside):
    # figure_slot
```

**Captions must be allowed in.** An earlier version required *every* item to be
a picture, which excluded exactly the well-formed case — when the layout model
succeeds at linking, the serializer emits caption and picture together.
Measured: 10 of 16 figures came out as 30-token stubs with no description, while
the 6 *unlinked* ones got full ones. The better the parse, the worse the record.

A table in the chunk disqualifies it — tables carry values that must be indexed
as themselves.

**Type classification needs a `list` branch.** A chunk of only `LIST_ITEM`
elements falls through a naive type check and gets labelled `"text"`, which means
the merge pass will happily weld two unrelated bulleted lists together. Ask
whether *all* items are list items before defaulting to text.

### `_merge_prose` — same heading, both text

The chunker's own predicate plus the type check it lacks:

```python
if entry["kind"] == "text":
    for candidate in reversed(out):
        if candidate["head"] != entry["head"]:
            break                     # heading boundary: stop
        if candidate["kind"] == "text":
            target = candidate; break
        if not MERGE_ACROSS_EXHIBITS:
            break
```

**`MERGE_ACROSS_EXHIBITS`** lets the backward search reach past a figure or table
to find prose under the same heading. Without it, a document whose prose is
interleaved with exhibits never merges at all — every run is length one.

Measured: 47 records / 61-token median → 37 records / 171-token median.

The cost is real: a merged record's page range spans exhibits it does not
contain, and prev/next no longer step through strict document order for those
records.

**Bound how far it reaches.** Reaching back past an unbounded number of non-text
elements is how prose gets glued to sidebar content four elements away. Stopping
at a table (harder boundary than a figure) is a reasonable rule.

### `_apply_floor` — the only pass that crosses headings

Off by default (`MIN_CHUNK_TOKENS=0`). When on, a text entry under the floor
absorbs the next one *even across a heading boundary*, writing the crossed
heading into the body so context is not silently lost.

Measured at 150 tokens across three document types:

| Document | floor=0 | floor=150 |
|---|---|---|
| 7-page research report | 34 records, median 260 | 33 records, median 297 |
| 15-page IRB form | 63 records, median 92 | 32 records, median 302 |
| 97-page protocol | 257 records, median 130 | 168 records, median 274 |

It merges only *while* the accumulator is under the floor, so documents that
already clear it are untouched. That is what makes one value safe across a mixed
corpus.

**Bound what it absorbs, not just the accumulator.** Checking only
`len(previous) < FLOOR` lets a 7-token fragment swallow an entire 380-token
paragraph block in one step. Check the size of the thing being absorbed too.

### `_to_records` — figures expand here

Each figure description is a self-contained fact about a different chart. Packing
several into one vector produces a vector representing none of them.

Get the caption from the *document* via `caption_text(doc)`, not from the chunk.
Without it, the caption becomes a separate 25-token chunk and the figure carries
no exhibit number.

An unmerged entry keeps the chunker's own `contextualize()` string; a merged one
is rebuilt by hand, because joining two contextualized strings repeats the
heading path mid-chunk.

### `_table_summaries` — additional, not replacement

```
logical table
  +-- chunk  chunk  chunk        fragments, each already a record
  |
  +-- ONE summary                an ADDITIONAL record, same table_id
```

Fragments hold exact values; the summary holds the words someone would search
for. `table_id` is how retrieval walks from a matched summary to the rows.

Read the summary from the complete grid on the document, not stitched back from
fragments.

### `_finalise` — order, links, limits

Sort on `(position, is-not-a-summary)` so each summary lands immediately before
the rows it describes.

Store `prev_id`/`next_id` explicitly — content-addressed ids cannot be derived
from position.

Compute the metadata budget from the store's real cap minus measured overhead
minus a buffer, rather than guessing a character limit.

Truncating an unsplittable oversized element loses that one item; raising would
abort a 250-page document over one row. Flag it so the loss is visible.

## Tuning values, with their measured basis

```python
FURNITURE_MAX_TOKENS = 15      # nothing larger is furniture, whatever it looks like
PROSE_TARGET_TOKENS = 400      # well under CHUNK_TOKENS on purpose
MERGE_ACROSS_EXHIBITS = True   # measured: 61 -> 171 token median
MIN_CHUNK_TOKENS = 0           # per-corpus; 150 validated on forms and protocols
```

`PROSE_TARGET_TOKENS` is deliberately well below the chunk ceiling. The merge
should improve small chunks, not manufacture maximal ones. Current benchmarks
converge on 300–500 tokens as the useful range, with quality degrading past
roughly 2,500.
