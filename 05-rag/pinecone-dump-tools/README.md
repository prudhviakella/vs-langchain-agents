# Loading the pre-built Pinecone data

```
  instructor's Pinecone index                your own Pinecone index
  (already has all the documents)             (empty, brand new)
         |                                             ^
         |  export_pinecone.py                         |  import_pinecone.py
         v                                             |
      dump.json  --------------------------------------+
         (one file, shared with you)
```

You do not need to run the AWS pipeline yourself. The instructor already
ran it and has a working Pinecone index full of real data. This one file
is a copy of that data. Loading it into your own account takes one
command and a few minutes — no AWS account, no Docling, no OpenAI calls.

## 1. What you need

- Your own Pinecone account (free tier is enough) — sign up at
 pinecone.io if you don't have one yet
- Your own Pinecone API key — Pinecone console → API Keys
- Python 3.10+
- The `dump.json` file the instructor shared with you

## 2. Install the one dependency

```bash
pip install pinecone
```

That's it — these two scripts need nothing else.

## 3. Load the data

```bash
PINECONE_API_KEY=your-key-here python import_pinecone.py \
  --file dump.json --index my-first-index
```

Replace `your-key-here` with your real API key, and `my-first-index`
with whatever name you want your index to have — it does not need to
match the instructor's.

If an index by that name doesn't exist yet in your account, this creates
it automatically, sized correctly for the data in the file. You don't
need to know or guess the dimension or the distance metric — the file
already knows, and the script reads it from there.

You'll see something like:

```
reading dump.json
  4213 vector(s), dimension=1536, metric='dotproduct', exported 2026-09-15T10:00:00+00:00
  index 'my-first-index' does not exist — creating it (dimension=1536, metric='dotproduct', cloud=aws, region=us-east-1)
  index 'my-first-index' ready
  loaded 4213 of 4213 vector(s)

done — 4213 vector(s) now in index 'my-first-index', namespace ''
```

That's the whole exercise. Your index now has the same data the
instructor's does.

## 4. If something goes wrong

**`PINECONE_API_KEY is not set`** — you forgot the `PINECONE_API_KEY=...`
part at the start of the command, or mistyped it.

**`... already exists with dimension X, but this file's vectors are
dimension Y`** — you're trying to load into an index you already used
for something else. Pick a new `--index` name instead.

**Really slow, or seems stuck** — normal for the first minute or two
while Pinecone provisions a brand-new index. After that, loading a few
thousand vectors usually takes well under a minute.

---

## For the instructor: producing `dump.json`

```bash
PINECONE_API_KEY=your-key python export_pinecone.py \
  --index rag-docs --out dump.json
```

Add `--gzip` if the file is too large to share easily — `.gz` is added
to the filename automatically, and `import_pinecone.py` reads a gzipped
file exactly the same way, no extra flag needed on the student's side.

If the index uses more than the default namespace, export each one
separately with `--namespace <name>` and give students one file per
namespace, since one export only covers one namespace at a time.
