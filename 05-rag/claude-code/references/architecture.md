# Architecture

## Module boundaries

Each module owns one thing. The boundaries matter because they are what let you
verify a stage in isolation.

```
config.py       settings, read from env AT IMPORT TIME
clients.py      OpenAI + vector store clients, EMBED_DIMS probed once
docling_io.py   PDF -> DoclingDocument, figure descriptions (cached)
headings.py     demote false headings ON the document, before chunking
tables.py       table serialization, broken-structure detection
chunking.py     DoclingDocument -> records (the policy passes)
embedding.py    texts -> vectors, batched by tokens, cached by hash
index.py        vector store handle, dimension check
sync.py         incremental add/remove against what is already indexed
inspect.py      reports: extraction quality, chunk quality
retrieval.py    query -> rerank -> expand -> answer
ingest_all.py   batch runner, crash-safe resume
```

### What each module must NOT do

- `docling_io.py` does not chunk, embed, or decide what is worth keeping
- `chunking.py` does not call the embedding model or write to the vector store
- `headings.py` runs before chunking and modifies the document in place
- `inspect.py` reads; it never modifies

## Data shapes

Two shapes flow through the pipeline. Keeping them distinct is what makes the
merge passes simple.

### entry — working data inside chunking

Mutable, plain dicts. Merged and rewritten by the policy passes.

```python
{
    "kind": "text",            # text | table | figure_slot | formula | code
    "body": "...",              # raw text, WITHOUT the heading path
    "contextualized": "...",    # chunker's own heading+text string
    "merged": False,
    "head": ["Section A"],      # heading path
    "page": 1, "page_end": 1,
    "ref": None,                # table ref, if a table fragment
    "image_uri": "",
}
```

A `figure_slot` is different — a placeholder with no body yet:

```python
{"kind": "figure_slot", "refs": ["#/pictures/2"],
 "head": [...], "page": 2, "page_end": 2}
```

### record — the final thing that gets embedded

```python
{
    "text": "...",              # what goes to the embedding model
    "meta": {
        "chunk_id": "doc:hash:0",
        "content_hash": "...", "occurrence": 0,
        "doc_id": "...", "source": "report.pdf",
        "doc_date": "2025", "ingested_at": 1234567890,
        "position": 4,
        "embed_model": "...", "access": ["public"],
        "n_tokens": 221,
        "page": 1, "page_end": 1,
        "headings": ["Section A"],      # a LIST, not a joined string
        "section_id": "...",
        "content_type": "text",
        "table_id": "",                  # non-empty for table fragments + summary
        "image_uri": "",
        # added by the final pass:
        "n_positions": 34, "prev_id": "...", "next_id": "...",
    }
}
```

One entry becomes one record, except:
- a `figure_slot` becomes zero or more (one per picture)
- a table entry also seeds a group that produces one extra summary record

### Why `headings` is a list

Vector stores filter a list with `$in` and cannot filter a comma-joined string
at all. This is not stylistic — joining it here makes a whole class of query
impossible later.

## Settings read at import time

`config.py` reads every setting with `os.getenv(...)` at module level, not lazily
inside functions. This is a deliberate trade: settings are frozen once, so there
is one source of truth per process and no mid-run drift.

The consequence, which bites people: **anything that loads environment variables
must run before the first import of your package.** In a notebook, `load_dotenv()`
belongs in a cell above the imports, not inside a function called later. This
failure is silent — the code runs with defaults and nothing errors.

## Build order rationale

Build `config` and `clients` first because everything imports them.

Build `docling_io` next and *stop* — parse a real PDF, dump the elements, look at
them. You cannot debug chunking on top of a parse you have not inspected.

Build `headings` before `chunking` because heading cleanup mutates the document
and chunking reads the result. Getting this order wrong produces chunk boundaries
derived from headings you later decide were wrong.

Build `inspect` early, not last. It is what makes every subsequent stage
debuggable.

Build `retrieval` last, and be honest that until there is an eval set, you are
tuning against statistics rather than answer quality.
