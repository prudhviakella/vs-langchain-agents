# Tables

## Serialize as markdown, not triplets

Docling's default chunk serializer renders a table as
`**Column**, row 1 = x` triplets, which flattens the grid and embeds badly. Pass
a markdown table serializer instead:

```python
class MarkdownTableProvider(ChunkingSerializerProvider):
    # parameter MUST be named `doc` — HybridChunker calls this by keyword
    def get_serializer(self, doc, **kwargs):
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())
```

## Detecting broken structure

TableFormer sometimes produces a grid that does not match the visual table. The
common failure is a header row where one logical header spans several columns and
the cell content gets duplicated or welded.

**The rule that distinguishes a span from a real split**: a repeated adjacent
header is a *span* if any row below distinguishes those columns. It is a *split*
if no row ever does.

This matters enormously. A naive "repeated adjacent header = broken" check
flagged 22 of 45 tables on one protocol; with the span rule, 2. The other 20 were
correct tables with legitimate column spans.

A duplicated header that is *itself* welded (`Row Lab Adopter 31,912` as one
cell) is always broken.

## When the grid is wrong, describe the image

Summarizing a broken grid produces a confident description of a table that does
not exist. Render the table region and send it to a vision model instead.

Requires `generate_table_images=True` at parse time. If `get_image()` returns
nothing, the parse predates that setting — say so explicitly rather than falling
through silently, because the fallback then produces a summary from the broken
markdown and nothing indicates it.

## A defect neither parser can fix

Some tables weld a sub-label and its value into one cell:

```
Container Description   | Type: Single use vial | Material: clear glass | Size: 10 mL
```

Each cell holds a label *and* a value, spread across columns as if each were a
different drug. Verified: two independent parsers produce this identically on the
same table, and the same defect recurs on every page where that template repeats.

When two unrelated tools fail the same way on the same input, the ambiguity is in
the source document. Detect it (`N/M cells hold a label and a number`), route to
the image fallback, and move on — there is no parser setting that fixes it.

## Markdown flattening duplicates spanning cells

A cell that genuinely spans three columns comes out as the same content three
times in markdown (`MedImmune | MedImmune | MedImmune`). That is wasted tokens at
embedding time and a real duplicate-vector risk.

HTML-based table representations handle this correctly with `colspan`. If table
fidelity matters more than pipeline simplicity for a given corpus, this is a real
argument for a different serializer — but measure it on the actual documents
first.

## Layout tables

Not every detected table is data. A 3-cell "table" is usually layout — a
header block, a two-column arrangement. Skip summarizing those and say why:

```
skipped table on p196: 3 cells, treated as layout
```

## Grouping fragments

A large table splits across chunks. Group the fragments by the table's own ref so
one summary can be written for all of them, and give every fragment plus the
summary the same `table_id` so retrieval can walk between them.

Note `MERGE`-identity versus search: the summary is keyed on `(measure, doc)` or
similar so unrelated tables do not collapse into one node, while `table_id` is
what links the family together.
