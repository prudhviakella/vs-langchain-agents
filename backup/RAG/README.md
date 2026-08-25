# Applied RAG — course materials

Three modules, in order. Each stands on its own; each assumes the one before it.

```
local/
  rag/        turn PDFs into a searchable index, then query it
  cypher/     Neo4j and Cypher from zero
  graphrag/   a knowledge graph over the same documents
aws/          running the ingestion pipeline at scale on Fargate
```

## The order

**1. `local/rag/`** — two notebooks. Read a PDF properly, split it, index it, then
search it. This is the whole pipeline on one document.

**2. `local/cypher/`** — a standalone Neo4j course. Twelve parts and a capstone, on
a dataset you build yourself. Needed before the next module.

**3. `local/graphrag/`** — a graph over the documents from module 1, joined to the
vector index by chunk id. Answers what similarity search cannot.

**4. `aws/`** — the same ingestion code from module 1, running a whole corpus in
parallel on Fargate. Optional; nothing later depends on it.

## What each module needs

| Module | Needs |
|---|---|
| `rag` | OpenAI key, Pinecone key |
| `cypher` | a free Neo4j Aura instance |
| `graphrag` | all of the above, and the index from module 1 |
| `aws` | an AWS account, and a Fargate vCPU quota above 6 |

## Conventions

`cypher/` teaches the Neo4j naming conventions and `graphrag/` follows them:
`UpperCamelCase` labels, `UPPER_SNAKE_CASE` relationship types, `lowerCamelCase`
properties.

The vector store keeps its own field names — `chunk_id`, `doc_id` — because the two
are joined on the *value* of an id, not on the property name. Each side follows its
own conventions.

## Before committing this anywhere

Add a `.gitignore`:

```
.env
.cache/
reports/
manifest.json
gold_set.json
__pycache__/
*.pyc
```

None of the code here holds a credential. Keep it that way.
