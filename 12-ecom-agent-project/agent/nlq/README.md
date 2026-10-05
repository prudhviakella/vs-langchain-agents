# NLQ agent — questions to read-only SQL on RDS PostgreSQL

```
supervisor ──A2A message/stream──► NLQ (AgentCore runtime ecom_nlq)
                                     │ grounding: Pinecone ecom-kb ─► schema · joins · terms · rules · examples
                                     │ gpt-6-sol writes SQL ─► guard ─► execute_sql
                                     │                                   └─SigV4─► Gateway ─► Lambda (VPC) ─► RDS as ecom_reader
                                     ├─ progress, streamed: grounding · executing (with the SQL) · result
                                     └─ result: {columns, rows, sql, entities, unmet_parts, note}
```

Ported from the ACT NLQ agent: rows go to state and never through the model; the model's
output has no field for SQL or data; budgets, guard and scope live in middleware.

## Layout

| Path | What |
|---|---|
| `agent_code/nlq/` | the agent: `core` · `grounding` · `tools` · `middleware` · `guard` · `schemas` · `state` · `prompt` |
| `agent_code/main.py` | A2A server; streams each progress event as a status update |
| `lambda_tools/` | `handler.py` + `sql_exec.py` — the only code that touches the database |
| `infra/` | reader user · network · Lambda · Gateway · runtime |
| `deploy.py` | deploys all of it (steps 0–8) |
| `tests/` | the agent loop and the A2A wire, against real PostgreSQL |

## Deploy

Needs: ingestion done (`ingestion/postgres/deployment.json`, Pinecone `ecom-kb`), and
`OPENAI_API_KEY` + `PINECONE_API_KEY` in `ingestion/.env`. In PyCharm's Terminal:

```bash
cd agent/nlq
pip install -r ../../ingestion/requirements.txt -e agent_code
python deploy.py
```

## Run locally (no AgentCore)

Queries go straight to PostgreSQL as `ecom_reader` (after `deploy.py` created it), or any local copy:

```bash
cd agent/nlq/agent_code
export NLQ_DB_MODE=local PGHOST=<host> PGUSER=ecom_reader PGPASSWORD=<from ecom-nlq/db-reader> PGDATABASE=ecom
export OPENAI_API_KEY=… PINECONE_API_KEY=…
python main.py                         # A2A server on :9000
```

## Test

```bash
cd agent/nlq
PGHOST=localhost PGUSER=ecom_reader PGPASSWORD=… PGDATABASE=ecom PGSSLMODE=disable python -m pytest tests -q
```

## Settings (environment variables, set by deploy.py)

| Variable | Default | Meaning |
|---|---|---|
| `NLQ_MODEL` | `gpt-6-sol` | the model |
| `NLQ_DB_MODE` | `gateway` | `local` runs SQL in-process (development) |
| `NLQ_MAX_TOOL_CALLS` / `NLQ_MAX_QUERIES` / `NLQ_MAX_REPAIRS` | 12 / 6 / 3 | per-question budgets |
| `NLQ_ROW_CAP` / `NLQ_QUERY_TIMEOUT_S` | 1000 / 20 | rows returned / statement timeout |
| `NLQ_EXAMPLES_VERIFIED_ONLY` | `false` | **set `true` once the team has reviewed the examples** (all are `verified = N` today) |
