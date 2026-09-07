# Parsing

## What the parse actually is

Several models in sequence, not one. Knowing which is which is what lets you read
a bad result and know where to look.

```
PDF
 |
 v
layout model      finds boxes, labels them, sets reading order,
 |                links captions to figures
 +-- TableFormer     rows and columns inside a table box
 +-- OCR             text on scanned pages with no text layer
 +-- classifier      chart / photo / logo / diagram
 +-- CodeFormula     equations and code
 |
 v
DoclingDocument   a graph of typed objects, not a string
 |
 v
describe_figures()  vision model, CACHED on rendered bytes  <- costs money
```

## Figure descriptions must be cached

Verified: with `temperature=0, seed=0`, 12 of 17 descriptions changed across two
identical runs. Because chunk ids are content-addressed, a changed description
means a changed id — meaning a "no-op" re-ingest silently churns the index.

Cache on `sha256(png_bytes + prompt + model)`. Note the render is part of the
key: change `FIGURE_RENDER_SCALE` or the layout model and every figure re-renders
to different bytes, correctly invalidating the cache.

Zero cache hits on a *repeat* parse of an unchanged PDF means the rendered bytes
changed — worth warning about, because every figure chunk_id changed with them.

## Heading cleanup

Runs on the document, before chunking. See the SKILL.md for why the ordering
matters.

Rules that generalize across corpora:

```python
SENTENCE_END   = r"[.,;]\s*$"                      # a heading does not end like a sentence
CONTINUATION   = r"^(and|but|or|the|a|an)\b"        # nor start mid-sentence
NUMBERED_LABEL = r"^(exhibit|figure|table|chart|panel)\s*\d+\s*:?\s*$"
MAX_HEADING_CHARS = 120                             # with an exemption for numbered sections
```

The length ceiling needs care. At 80 characters it demoted two *real* numbered
protocol sections with zero correct demotions on the same document. Exempt
numbered headings (`^\d+(\.\d+)*\s`) from it.

Rules based on a document's visual layout — a coloured box label inside an
exhibit, for instance — do not generalize. Write them when a corpus needs them,
accept they will be inert elsewhere, and expect roughly one new rule per new
document type.

**Verify the demotion actually happened.** Setting `item.label = TEXT` does not
work if the chunker tests `isinstance(SectionHeaderItem)` rather than reading the
label. Replace the object at its index instead, preserving `self_ref`, then
re-read the document and warn if any demoted heading survived. A demotion pass
that prints success while changing nothing ran undetected for two full ingestions
here.

## Heading hierarchy is flat by default

The layout model flags regions as `SECTION_HEADER` without a level, so every
heading defaults to `level=1` and `8.2` becomes a sibling of `8`.

Recent Docling versions ship a fix — a separate stage that rewrites levels from
bookmarks, then numbering markers, then visual style. Measured on a 113-page
protocol: 354 of 356 records at depth ≤1 became depth 2–4, matching the
protocol's real numbering.

It needs `generate_parsed_pages=True` for the style-fallback signal, and it is
off by default because a wrong level is worse than a missing one for pipelines
already tuned around flat headings. Wire the precondition in alongside the flag
so it cannot be forgotten.

## Confidence scores

`ConversionResult.confidence` gives Docling's own assessment: `mean_grade`,
`low_grade`, and per-component scores, with a per-page breakdown.

This is a **triage signal, not a fix**. A page can score EXCELLENT and still feed
a downstream chunking bug, because confidence describes the parse, not what your
code does with it. Use it to decide which pages are worth reading by hand.

`nan` on a component appears to mean "not applicable" (no tables to score, no OCR
needed) rather than "failed" — treat it as neutral, never as a failure.

Note the full `ConversionResult` is easy to discard by accident: `convert(...).document`
throws confidence away on the same line.

## Choosing a parser — measured, not assumed

Alternatives were tested on the same corpus:

| | tables | headings |
|---|---|---|
| default layout model | 3/3 | 2 false, patched downstream |
| alternative layout model | 1/3 | fewer false |
| VLM pipeline | 0/3 | 0 false |
| VLM + hybrid text mode | 0/3 | 0 false |

The VLM path genuinely fixed the heading problem — twice confirmed — and lost
every table doing it, degenerating into repeated-token output on the densest
page. That failure is *silent*: nothing flags it, the text just is not there.

A separate parser (MinerU) handled some table shapes markedly better — a grid
that broke every Docling configuration came out byte-exact — and flattened others
that Docling got right. Same shape of trade, not an upgrade.

Two useful findings from that comparison:

- When two independent parsers produce the *identical* error on the same text,
  the defect is in the source PDF, not either parser. An `AI`→`Al` misread and a
  welded label+value table cell both reproduced exactly across both.
- Specialized non-generative models beat general VLMs on table structure
  specifically. Reaching for a bigger model is the wrong instinct here.

If the user asks about switching: the honest answer is that every alternative
trades one failure for another, and the cost of a switch is rewriting every
module that depends on the document object model.
