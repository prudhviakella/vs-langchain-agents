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

If your index uses more than the default namespace, each export only covers one namespace at a time — export each one separately with `--namespace <name>`.

---

# Loading the pre-built Neo4j graph

```
  instructor's Neo4j database                 your own Neo4j database
  (already has the full graph)                 (empty, brand new)
         |                                             ^
         |  export_neo4j.py                            |  import_neo4j.py
         v                                             |
    graph_dump.json  -------------------------------------+
         (one file, shared with you)
```

Same idea as the Pinecone data, for the graph database instead. One file,
one command, no AWS pipeline needed.

## 1. What you need

- Your own Neo4j instance — Aura's free tier works fine
- Your own connection details: `NEO4J_URI`, `NEO4J_USER` (usually `neo4j`),
 `NEO4J_PASSWORD` — Aura shows these once when you create the instance
- Python 3.10+
- The `graph_dump.json` file the instructor shared with you

## 2. Install the one dependency

```bash
pip install neo4j
```

## 3. Load the data

```bash
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=your-password \
python import_neo4j.py --file graph_dump.json
```

This only works against an **empty** database — the one your fresh Aura
instance starts as. If you've already put something in it, either use a
different, empty instance, or pass `--force` if you're sure you want to
add to what's already there.

You'll see something like:

```
reading graph_dump.json
  5764 node(s), 8912 relationship(s), exported 2026-09-15T10:00:00+00:00
connecting to Neo4j
  created 5764 of 5764 node(s)
  created 8912 of 8912 relationship(s)

done — 5764 node(s) and 8912 relationship(s) now in the database
```

## 4. If something goes wrong

**`NEO4J_PASSWORD is not set`** — forgot that part of the command, or
mistyped the variable name.

**`... is not a real Neo4j connection URI`** — Aura's URI starts with
`neo4j+s://`, not `bolt://`. Copy it exactly from the Aura console.

**`This database already has N node(s)`** — you're pointing at an
instance you already used for something else. Create a fresh one, or use
`--force` if you're sure.

---

## For the instructor: producing `graph_dump.json`

```bash
NEO4J_URI=... NEO4J_USER=neo4j NEO4J_PASSWORD=... \
python export_neo4j.py --out graph_dump.json
```

Add `--gzip` the same way as the Pinecone export if the file needs to be
smaller to share. This one exports every node and relationship regardless
of label or type, so it works unchanged even if the graph schema changes
later.
