# Graph Databases with Neo4j and Cypher

A standalone course. Work through it before the GraphRAG module — that module assumes
you can read a Cypher query and know what a MERGE does.

Twelve parts and a capstone, on an e-commerce dataset you build yourself.

| Part | What |
|---|---|
| 0 | Setup and connection |
| 1 | Why graph databases — and when not to use one |
| 2 | The property graph model |
| 3 | Loading data into Aura |
| 4 | Cypher fundamentals |
| 5 | Aggregation and the `WITH` pipeline |
| 6 | Traversal and paths |
| 7 | Advanced Cypher |
| 8 | Graph analytics in pure Cypher |
| 9 | Performance and query tuning |
| 10 | Writing and refactoring the graph |
| 11 | Modelling deep dive |
| 12 | Capstone project |

## Setup

You need a free Neo4j Aura instance. Part 0 walks through creating one.

```bash
pip install neo4j pandas
jupyter lab 01_cypher_course.ipynb
```

The notebook asks for your credentials with `getpass`, so the password is never
saved into the notebook file. That matters if you share or commit it.

## Conventions this course teaches

The GraphRAG module follows them, so they are worth internalising here:

| Element | Convention | Example |
|---|---|---|
| Label | `UpperCamelCase`, singular | `:Trial`, `:MeSHTerm` |
| Relationship type | `UPPER_SNAKE_CASE`, a verb | `:SPONSORED_BY`, `:HAS_CHUNK` |
| Property | `lowerCamelCase` | `nctId`, `chunkId` |

## Then what

`../graphrag/` builds a knowledge graph over real documents. Everything you learn
here — MERGE, constraints, traversal, `WITH` pipelines — is what that module is
written in.
