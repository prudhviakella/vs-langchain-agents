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
