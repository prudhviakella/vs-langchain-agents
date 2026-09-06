# RAG Pipeline

Turn a folder of PDFs into a searchable index, then ask questions of it.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=pc-...

python ingest_all.py --dry-run     # what would run, parses nothing
python ingest_all.py               # ingest everything in pdfs/
```

Then read what it produced, before trusting any of it:

```
reports/_run.md                    one row per document
reports/<doc>/<doc>.extract.md     every element, with page and description
reports/<doc>/<doc>.chunks.md      every chunk exactly as it will be stored
```

Keep keys out of the repo. A `.env` committed once is a `.env` in the history
forever.

---

## The idea worth remembering

**When this pipeline goes wrong, it does not crash.** A setting left off means
an equation becomes a placeholder, a chart never gets described, a table comes
out with a broken grid. Everything downstream runs happily on top of it, and
you get an index that looks complete and is missing content.

Reporting that something happened is not evidence that it did. An early version
of `headings.py` printed `demoted 4 false headings` on every run and changed no
chunk boundary at all, because it set an attribute the chunker never reads. It
ran that way for two full ingestions. `clean_headings` now re-reads the document
afterwards and warns if a demotion did not take.

So every stage has a checkpoint, and the reports are written next to the PDF
before anything is embedded.

---

## Running it

| Command | What |
|---|---|
| `python ingest_all.py` | everything in `pdfs/` |
| `python ingest_all.py --dry-run` | list what would run and what would be skipped |
| `python ingest_all.py --only NCT03164772` | one document, matched on filename |
| `python ingest_all.py --force` | re-ingest even if already indexed |

Smallest document first, one at a time, committed as it goes. A crash on
document 18 costs you document 18, not the first 17. A document that fails is
recorded and the run continues.

Resume is on file size and mtime, so replacing a PDF re-ingests it. Use
`--force` if you swapped in a file of identical size and timestamp.

Two notebooks do the same thing interactively, one document at a time:

| Notebook | What |
|---|---|
| `01_ingestion.ipynb` | read a PDF, check the parse, split it, index it |
| `02_retrieval.ipynb` | search, rerank, filter, answer |

---

## The code

| Module | What |
|---|---|
| `config.py` | every setting both halves must agree on |
| `docling_io.py` | reading the PDF, and describing its figures |
| `inspect.py` | checking the parse worked, and writing readable reports |
| `headings.py` | demoting regions the layout model wrongly called headings |
| `tables.py` | tables: reading, checking, summarising |
| `chunking.py` | splitting into records, with metadata |
| `DESIGN.md` | why chunking looks like this — every measurement behind it |
| `embedding.py` | embedding, with a cache |
| `index.py` | talking to the vector database |
| `sync.py` | working out what changed since last time |
| `retrieval.py` | search, rerank, answer |
| `clients.py` | API clients |

Checkers, none of which need API keys:

| Script | What |
|---|---|
| `check_config.py` | what settings are actually in effect |
| `check_wiring.py` | is the loaded code the code on disk |
| `check_chunks.py` | names which of six chunking failures you have |
| `profile_parse.py` | times each enrichment separately |
| `diagnose_figure.py` | why one figure has no description |

---

## Three things the chunker will not do for you

Docling's `HybridChunker` splits on document structure, then refines against a
token budget. It is the right foundation and there is no better one for a
layout-parsed PDF — it is the only chunker that consumes the parse directly, so
page numbers, heading paths and table references survive into the records.

Read its source and it is four stages, of which exactly one ever combines
anything:

```
HierarchicalChunker                      one chunk per detected element
_split_by_doc_items                      window the items to fit the budget
_split_using_plain_text                  semchunk whatever is still oversized
_merge_chunks_with_matching_metadata     only when merge_peers=True
```

It has a ceiling and no floor. It never filters. Its merge is blind to element
type — the predicate is `headings == current_headings` on consecutive chunks,
nothing more, which is why turning it on glues a paragraph to a contact table.

So three policies are ours, in `chunking.py`:

| Policy | What |
|---|---|
| the drop filter | attribution lines and logo glyphs never become vectors |
| the prose merge | adjacent prose under one heading, up to `PROSE_TARGET_TOKENS` |
| the figure pass | one record per figure, never several in one vector |

There is **no `min_tokens` parameter** on `HybridChunker`, whatever some
documentation mirrors say. Passing one is silently ignored.

---

## What the layout model decides

Everything downstream keys off its labels, and two of its outputs matter more
than they look.

**Which regions are headings.** A heading is a chunk boundary *and* the string
prepended into every vector beneath it *and* the entire merge predicate. A
region wrongly called a section header does all three kinds of damage at once.
`headings.py` repairs what it can.

**Which regions are tables.** A table becomes a grid of exact values plus a
searchable summary. The same region called a picture becomes a paragraph of
prose from a vision model.

Neither has a pipeline flag. The lever is `LAYOUT_MODEL`, and the extraction
report prints the numbers so comparing two models is reading two lines:

```
layout (default): 10/17 captions linked, 3/16 headers look false
```

**Measured, on a 7-page research report:** `heron_101` scores better on
DocLayNet and was worse here — it reclassified two of three tables as pictures.
Benchmark accuracy is measured on the benchmark's distribution, with the
benchmark's notion of correctness. Your corpus is neither. Test, don't assume.

The `egret_*` variants crash on some builds — their HuggingFace configs use
hyphenated label names that Docling's label map does not normalise.

---

## Settings

Everything is an environment variable. `python check_config.py` prints all of
them, the environment value, and whether they agree — they disagree more often
than you would think, because each module reads the environment once, at import.

Parse enrichments (`config.py`):

```bash
DO_CHART_EXTRACTION=0   # reads numbers off charts. 0 of 17 on vector-drawn
                        # charts, which is most financial PDFs. Off by default.
DO_FORMULA=0 DO_CODE=0  # pure waste if your documents have no equations
TABLE_MODE_ACCURATE=0   # FAST is quicker and worse on nested headers
FIGURE_RENDER_SCALE=1.0 # 2x is four times the pixels per figure. But at 1x the
                        # vision model cannot read axis labels and invents numbers.
# DO_OCR=0              # do NOT. It is conditional, so it saves almost nothing,
                        # and it is what reads text inside a graphic.
```

Parse structure (`docling_io.py`):

```bash
LAYOUT_MODEL=heron_101        # see above. Default is heron.
LAYOUT_SCORE_THRESHOLD=0.5    # detection confidence. Raise to suppress
                              # marginal boxes without changing model.
TABLE_CELL_MATCHING=0         # match TableFormer cells to the PDF's text cells
CREATE_ORPHAN_CLUSTERS=0      # whether isolated elements get their own cluster
CACHE_FIGURE_DESCRIPTIONS=0   # hand figure descriptions back to docling
```

Chunking (`chunking.py`):

```bash
MIN_CHUNK_TOKENS=150          # 0 = off. Small records merge ACROSS a heading.
MERGE_ACROSS_EXHIBITS=0       # stop prose merging past a figure or table
```

### `MIN_CHUNK_TOKENS` is the one to set per corpus

The chunker never merges across a heading, and that is right when headings mean
sections. On a form it is not. Measured:

```
                              floor 0                floor 150
7-page research report     34 rec, median 260     33 rec, median 297
15-page IRB form           63 rec, median  92     32 rec, median 302
97-page protocol          257 rec, median 130    168 rec, median 274
```

It merges only while the accumulating record is still under the floor, so a
document whose chunks already clear it is untouched. **150 is validated across
all three.** Every heading crossed is written into the body, so the context of
each part is still embedded.

---

## Figure descriptions are cached

The vision model is the only stage that costs money per call, and the only
non-deterministic one. `temperature=0` and a fixed seed make OpenAI best-effort
reproducible, not reproducible: **12 of 17 descriptions came back reworded**
across two runs with identical settings.

That matters because chunk ids are hashes of chunk text and a figure's
description is part of it. Reword twelve descriptions, change twelve chunk ids,
and `sync.py` deletes and re-embeds every figure in a document nobody edited.

So the description step runs after the parse, in `describe_figures()`, against a
cache keyed on the rendered image bytes plus the prompt plus the model. Change
any of those and the key changes on its own.

**Do not clear `.cache/figures`.** New keys sit alongside old ones; nothing
stale is ever read. Clearing it only means paying for the same descriptions
again.

The parse itself is not cached, and should not be: it depends on the settings
above as much as on the file, so a cache keyed on the filename returns work done
under different settings.

---

## If parsing is slow

Every enrichment is a **model pass, on CPU, per element**. Find out which,
rather than guessing:

```bash
python profile_parse.py pdfs/your.pdf
```

**Conditional or unconditional is the distinction that matters.** Some models
run once per element whether or not it needs them. Others run only where there
is work to do. Switching off the wrong kind saves almost nothing and loses
content.

| | Runs on |
|---|---|
| chart extraction | every figure |
| classification | every figure |
| formula / code | every candidate region |
| rendering at 2× | every figure, four times the pixels |
| **OCR** | **only regions with no extractable text layer** |

On a digital PDF, OCR is nearly free. But it is the **only** thing that reads
text baked into a graphic — an exhibit drawn as coloured boxes with a bulleted
list inside loses its entire contents without it.

Turn off the unconditional ones. Leave OCR on unless you have measured that it
costs you something.

The first run also downloads about 500 MB of model weights. One-time.

---

## Reading the reports

`reports/_run.md` is the corpus view. Per document:

```
reports/<doc>/<doc>.extract.md     what the parse found, and what went wrong
reports/<doc>/<doc>.chunks.md      what will be stored, exactly
```

The extract report opens with two numbers no setting controls, both pure layout
model quality:

```
- captions linked to a figure: 10 of 17
- section headers that do not look like headings: 3 of 16
```

Read the second as a floor, not a truth — text rules only catch what text rules
can catch. A box label inside an exhibit reads exactly like a real heading.

Then:

```bash
python check_config.py
python check_wiring.py
python check_chunks.py reports/<doc>/<doc>.chunks.json
```

`check_chunks.py` names which of six failures you have, and each has a different
fix. It reads only the reports, so it needs no keys and no parse.

---

## Known limits

**Heading hierarchy is lost.** Docling assigns every section header the same
level, so `8.2` and `8.3.2` are siblings of `8`. The numbering carries a real
hierarchy the parse discards. Recoverable from the text, not yet done.

**Captions are the layout model's job and it often fails.** When it does, the
caption survives as a loose text element and the figure record has no exhibit
number, so "what does Exhibit 9 show" has nothing to match. Read the ratio over
*described* figures — a count that includes page wordmarks is meaningless.

**Filenames lie.** In one 20-document corpus, 15 filenames disagreed with the
document's own title page. `doc_id` comes from the filename. Check page 1
against the filename before indexing anything you intend to cite.

**There is no retrieval eval.** Every chunking decision here — the floor at 150,
`MERGE_ACROSS_EXHIBITS`, which layout model — rests on chunk statistics, which
describe the shape of an index and say nothing about whether it answers
questions. Twenty questions with known-correct chunk ids would turn all of them
into measurements. That is the next thing worth building.
