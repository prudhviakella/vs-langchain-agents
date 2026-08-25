# Graph RAG — local

A knowledge graph over the same documents, in Neo4j. Answers the questions vector
search cannot: joins across documents, and how things connect.

Run the RAG notebooks first — this reads the chunks out of that index.

| Notebook | What it does |
|---|---|
| `01_build_graph.ipynb` | Build the graph in three layers |
| `02_query_graph.ipynb` | Query it, and compare against vector search |

## Setup — Neo4j Aura

Create a free instance at https://neo4j.com/cloud/aura. It gives you a credentials
file when the instance is created:

```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<generated>
```

**The password is shown once and never again.** If it is lost, reset it from the
Aura console.

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-... PINECONE_API_KEY=pc-...
export NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=...

jupyter lab 01_build_graph.ipynb
```

Two things about Aura worth knowing before you start.

**The URI scheme matters.** Aura needs `neo4j+s://`, which is encrypted. A plain
`bolt://` URI fails with a message about routing, which has nothing to do with the
real cause. `driver()` checks this and says so.

**A free instance pauses after 3 days of no activity** and is deleted after 30. Your
graph is not permanent. Rebuild it from the notebooks — that takes minutes and the
extraction is cached, so it costs nothing the second time.

The free tier caps you at 200,000 nodes and 400,000 relationships. Twenty clinical
protocols will not come close.

### Or run it locally

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5

export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=password
```

Browse at http://localhost:7474. Aura has the same browser built in.

**Never put a password in a source file.** A password written into code is in every
copy of that code, and changing it means rewriting history.

## Three layers

```
1  structure   Document → Section → Chunk        free, from metadata
2  registry    Trial and everything it links     free, from ClinicalTrials.gov
3  extracted   what only the document knows      one LLM call per chunk
```

The order is the point. Structure first, because it is free. Registry second, because
it is correct by definition. Extraction last, and **only for what the other two
cannot know**.

Most graph pipelines extract everything with a model, including facts that are
already published in a registry. That means paying to guess at answers you can look
up — and leaving no way to tell a known fact from a guess once both are nodes.

Here every node records where it came from, so a query can ask for facts or accept
guesses.

## The useful side effect

Because the registry is correct, you can **measure** how accurate the extraction is.
Pull the sponsor out of the PDF, compare it to the registry, count. That is a real
number for a step nobody usually measures.

## The code

| Module | What |
|---|---|
| `config.py` | Connections and models |
| `structure.py` | Layer 1 — documents, sections, chunks |
| `registry.py` | Layer 2 — ClinicalTrials.gov |
| `schema.py` | Layer 3 — what to extract, and what to reject |
| `extract.py` | The extraction calls, cached |
| `store.py` | Writing to Neo4j, and the queries |
| `answer.py` | Answering from the graph |
| `accuracy.py` | Scoring extraction against the registry |
