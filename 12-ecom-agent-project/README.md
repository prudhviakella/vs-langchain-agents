# Agents

The NLQ agents go here. They read what `../ingestion/` produced:

| Input | Where it comes from |
|---|---|
| PostgreSQL schema `ecom` — 14 tables, keys, table and column comments | `../ingestion/postgres/deployment.json` (host, port, database, secret ARN) |
| Pinecone index `ecom-kb` — namespaces `nlq-schema`, `nlq-examples`, `common` | `PINECONE_INDEX` in `../ingestion/.env` |
| Embedding model for questions: **`text-embedding-3-small`** (1536-d) — the SAME model the KB was embedded with | the index's `embedding_model` tag |
| Querying rules (merchandise revenue, delivered-only delivery times, …) | `../ingestion/data/clean/DATA_DICTIONARY.md` and the `capability_card` records |

Agents connect to PostgreSQL as a read-only user, never as `ecom_admin`.
