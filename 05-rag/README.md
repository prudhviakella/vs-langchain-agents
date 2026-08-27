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

## The code

The notebooks are short on purpose. All the machinery is in the `rag` package, one
module per step:

| Module | What |
|---|---|
| `config.py` | Every setting both notebooks must agree on |
| `docling_io.py` | Reading the PDF |
| `inspect.py` | Checking the parse worked, and writing readable reports |
| `tables.py` | Tables: reading, checking, summarising |
| `chunking.py` | Splitting into pieces, with metadata |
| `embedding.py` | Embedding, with a cache |
| `index.py` | Talking to the vector database |
| `sync.py` | Working out what changed since last time |
| `retrieval.py` | Search, rerank, answer |
| `clients.py` | API clients |

You do not need to read them to follow the notebooks. Open one if you want the
details of a step.

## If parsing is slow

Every enrichment is a **model pass, on CPU, per element**. A 7-page report with
17 figures is not a small document to this pipeline — it is 17 crops rendered at
2×, then classified, then chart-extracted, then sent to a vision model. Four
passes over each figure.

Twenty minutes for eight pages means something is running that should not be.
Find out which, rather than guessing:

```bash
python profile_parse.py pdfs/your.pdf
```

It parses the same PDF several times, adding one flag at a time, and prints what
each one costs on your machine with your document. The difference between two
rows is that flag's price.

Then switch off what you are not using:

```bash
DO_CHART_EXTRACTION=0   # reads numbers off charts. Measured 0 of 17 on
                        # vector-drawn charts, which is most financial PDFs.
                        # Already off by default.
DO_FORMULA=0 DO_CODE=0  # pure waste if your documents have no equations
TABLE_MODE_ACCURATE=0   # FAST is several times quicker and worse on nested
                        # headers. Fine for simple grids.
FIGURE_RENDER_SCALE=1.0 # 2x is four times the pixels, per figure. But at 1x
                        # the vision model cannot read axis labels and starts
                        # inventing numbers, so check the descriptions after.
```

The first run also downloads about 500 MB of model weights. That is one-time,
and it is not the 22 minutes.

## The idea worth remembering

**When this pipeline goes wrong, it usually does not crash.** A setting left off
means an equation becomes a placeholder, or a chart never gets described, or a table
comes out with a broken grid. Everything downstream runs perfectly happily on top of
it, and you get an index that looks complete and is missing content.

So every step has a checkpoint, and ingestion writes reports you can read next to the
original PDF:

```
reports/<doc>.extract.md    every element, with its page and its description
reports/<doc>.chunks.md     every chunk exactly as it will be stored
```

Read those before trusting anything.
