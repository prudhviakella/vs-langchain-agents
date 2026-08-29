# RAG Pipeline — local

Turn PDFs into a searchable index, then ask questions of it.

Two notebooks. Run them in order.

| Notebook | What it does |
|---|---|
| `01_ingestion.ipynb` | Read a PDF, check the parse, split it, put it in the index |
| `02_retrieval.ipynb` | Search, rerank, filter, answer |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=pc-...

jupyter lab 01_ingestion.ipynb
```

Put your PDFs in a `pdfs/` folder next to the notebooks.

Keep keys out of the repo. A `.env` committed once is a `.env` in the history
forever.

## The code

The notebooks are short on purpose. All the machinery is in the `rag` package,
one module per step:

| Module | What |
|---|---|
| `config.py` | Every setting both notebooks must agree on |
| `docling_io.py` | Reading the PDF, and describing its figures |
| `inspect.py` | Checking the parse worked, and writing readable reports |
| `headings.py` | Demoting regions the layout model wrongly called headings |
| `tables.py` | Tables: reading, checking, summarising |
| `chunking.py` | Splitting into pieces, with metadata |
| `embedding.py` | Embedding, with a cache |
| `index.py` | Talking to the vector database |
| `sync.py` | Working out what changed since last time |
| `retrieval.py` | Search, rerank, answer |
| `clients.py` | API clients |

You do not need to read them to follow the notebooks. Open one if you want the
details of a step.

## Three things the chunker will not do for you

Docling's `HybridChunker` splits on document structure, then refines against a
token budget. That is the right foundation and there is no better one for a
layout-parsed PDF — it is the only chunker that consumes the parse directly, so
page numbers, heading paths and table references survive into the records.

But read its source and it is four stages, of which exactly one ever combines
anything:

```
HierarchicalChunker                      one chunk per detected element
_split_by_doc_items                      window the items to fit the budget
_split_using_plain_text                  semchunk whatever is still oversized
_merge_chunks_with_matching_metadata     only when merge_peers=True
```

It has a ceiling and no floor. It never filters. And its merge is blind to
element type — the predicate is `headings == current_headings` on consecutive
chunks, nothing more, which is why turning it on glues a paragraph to a contact
table.

So three policies are ours, and they live in `chunking.py`:

| Policy | What it does |
|---|---|
| the drop filter | attribution lines and logo glyphs never become vectors |
| the prose merge | adjacent prose under one heading, up to `PROSE_TARGET_TOKENS` |
| the figure pass | one record per figure, never several in one vector |

There is **no `min_tokens` parameter** on `HybridChunker`, whatever some
documentation mirrors say. Passing one is silently ignored.

## What the layout model decides

Everything downstream keys off the labels the layout model assigns. Two of its
outputs matter more than they look:

**Which regions are headings.** A heading is a chunk boundary *and* the string
prepended into every vector beneath it *and* the entire merge predicate. A
region wrongly called a section header does all three kinds of damage at once.
`headings.py` repairs what it can.

**Which captions belong to which figure.** When the link fails, the caption
survives as a loose text element and the figure record has no exhibit number —
so "what does Exhibit 9 show" has nothing to match.

Neither has a pipeline flag. The only lever is `LAYOUT_MODEL`:

```bash
LAYOUT_MODEL=heron_101    # the accuracy variant of the default
```

The extraction report prints both numbers, so comparing two models is reading
two lines:

```
layout (default): 10/17 captions linked, 3/16 headers look false
```

The `egret_*` variants are faster and more accurate on paper but crash on some
builds — their HuggingFace configs use hyphenated label names that Docling's
label map does not normalise. Try `heron_101` first.

## Settings

Parse enrichments, in `config.py`:

```bash
DO_CHART_EXTRACTION=0   # reads numbers off charts. Measured 0 of 17 on
                        # vector-drawn charts, which is most financial PDFs.
                        # Already off by default.
DO_FORMULA=0 DO_CODE=0  # pure waste if your documents have no equations
TABLE_MODE_ACCURATE=0   # FAST is several times quicker and worse on nested
                        # headers. Fine for simple grids.
# DO_OCR=0              # do NOT. It is conditional, so it saves almost nothing,
                        # and it is what reads text inside a graphic.
FIGURE_RENDER_SCALE=1.0 # 2x is four times the pixels, per figure. But at 1x
                        # the vision model cannot read axis labels and starts
                        # inventing numbers, so check the descriptions after.
```

Parse structure, in `docling_io.py`:

```bash
LAYOUT_MODEL=heron_101          # see above
TABLE_CELL_MATCHING=0           # try this on duplicated adjacent header cells
CACHE_FIGURE_DESCRIPTIONS=0     # hand figure descriptions back to docling
```

Chunking, in `chunking.py`:

```bash
MERGE_ACROSS_EXHIBITS=0         # stop prose merging past a figure or table
```

`python check_config.py` prints every one of these, the environment value, and
whether they agree. They disagree more often than you would think, because each
module reads the environment once, at import.

## Figure descriptions are cached

The vision model is the only stage that costs money per call, and the only
non-deterministic one — `temperature=0` and a fixed seed make OpenAI
best-effort reproducible, not reproducible.

That matters more than it sounds. Chunk ids are hashes of chunk text, and a
figure's description is part of that text. Reword sixteen descriptions and you
change sixteen chunk ids, and `sync.py` deletes and re-embeds every figure in a
document nobody edited.

So the description step runs after the parse, in `describe_figures()`, against a
cache keyed on the rendered image bytes plus the prompt plus the model. Change
any of those and the key changes on its own. Change nothing and the description
is byte-identical, forever, for free.

The parse itself is still not cached, and should not be: it depends on the
settings above as much as on the file, and a cache keyed on the filename returns
work done under different settings.

## If parsing is slow

Every enrichment is a **model pass, on CPU, per element**. A 7-page report with
17 figures is not a small document to this pipeline.

Twenty minutes for eight pages means something is running that should not be.
Find out which, rather than guessing:

```bash
python profile_parse.py pdfs/your.pdf
```

It parses the same PDF several times, adding one flag at a time, and prints what
each one costs on your machine with your document.

### Conditional or unconditional — this is the distinction that matters

Some models run **once per element, whether or not that element needs them**.
Others run **only where there is work to do**. Switching off the wrong kind saves
almost nothing and loses content.

| | Runs on |
|---|---|
| chart extraction | every figure |
| classification | every figure |
| formula / code | every candidate region |
| rendering at 2× | every figure, four times the pixels |
| **OCR** | **only regions with no extractable text layer** |

On a digital PDF, OCR is nearly free — most text is already in the layer. But it
is the **only** thing that reads text baked into a graphic. An exhibit drawn as
coloured boxes with a bulleted list inside loses its entire contents without it.

**Turn off the unconditional ones. Leave OCR on** unless you have measured that
it costs you something.

The first run also downloads about 500 MB of model weights. That is one-time,
and it is not the 22 minutes.

## The idea worth remembering

**When this pipeline goes wrong, it usually does not crash.** A setting left off
means an equation becomes a placeholder, or a chart never gets described, or a
table comes out with a broken grid. Everything downstream runs perfectly happily
on top of it, and you get an index that looks complete and is missing content.

Reporting that something happened is not evidence that it did, either. An
earlier version of `headings.py` printed `demoted 4 false headings` on every run
and changed no chunk boundary at all, because it set an attribute the chunker
never reads. It ran that way for two full ingestions. `clean_headings` now
re-reads the document afterwards and warns if a demotion did not take.

So every step has a checkpoint, and ingestion writes reports you can read next
to the original PDF:

```
reports/<doc>/<doc>.extract.md    every element, with its page and its description
reports/<doc>/<doc>.chunks.md     every chunk exactly as it will be stored
```

One folder per document. A 20-document corpus writes 80 report files, and flat
that is a directory nobody opens. Reports written by an earlier version sit
flat in `reports/` — delete them or leave them, they are outputs and every one
is regenerated by a re-run.

Read those before trusting anything. Then:

```bash
python check_config.py                       # what settings are in effect
python check_wiring.py                       # is the loaded code the current code
python check_chunks.py reports/<doc>/<doc>.chunks.json
```

`check_chunks.py` names which of six failures you have, and each has a different
fix. It reads only the reports, so it needs no keys and no parse.
