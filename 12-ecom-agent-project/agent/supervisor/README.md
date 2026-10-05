# Supervisor — plans, asks NLQ, charts with chart_gen, composes the streamed answer

```
backend ──A2A message/stream {question, history, conversation_id}──► supervisor (runtime ecom_supervisor)
   1 route (gpt-6-sol)          reasoning: "Plan: …"
   2 NLQ per sub-question       status (grounding · executing · result) + reasoning: the SQL, row counts
   3 chart_gen on the table     status: charting · reasoning: "Chart: bar — …"
   4 compose (streamed)         token × n
   5 final                      {kind, text, reasoning, entities, artifacts{table, charts}, agents_used, usage_by_agent}
```

Every event is an A2A status update; `final` is the turn's artifact — the shape ACT's backend
reads. The table in `final` is NLQ's rows, attached by code; the answer is written only from
that evidence.

## Deploy — after NLQ and chart_gen

```bash
cd agent/supervisor
python deploy.py
```

## Run the whole chain locally

```bash
cd agent/nlq/agent_code && PORT=9001 python main.py          # with NLQ_DB_MODE=local and PG* set
cd agent/chart_gen/agent_code && PORT=9002 python main.py
cd agent/supervisor/agent_code && NLQ_URL=http://localhost:9001/ CHART_URL=http://localhost:9002/ python main.py
```

## Test — the real chain over HTTP (NLQ and chart_gen as live A2A servers, real PostgreSQL)

```bash
cd agent/supervisor/tests
PGHOST=localhost PGUSER=ecom_reader PGPASSWORD=… PGDATABASE=ecom PGSSLMODE=disable python -m pytest -q
```
