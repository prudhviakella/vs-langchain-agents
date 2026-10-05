# Agents

```
backend ──A2A stream──► supervisor ──A2A stream──► nlq ──SigV4──► Gateway ──► Lambda (VPC) ──► RDS (ecom_reader)
                          │                         └── Pinecone ecom-kb (grounding)
                          └──────A2A stream──────► chart_gen
   status · reasoning · token · final   — streamed back to the backend, then the browser's Reasoning tab
```

| Folder | Agent | Runtime |
|---|---|---|
| `nlq/` | questions → read-only SQL on PostgreSQL | `ecom_nlq` |
| `chart_gen/` | a table → Plotly figures | `ecom_chart_gen` |
| `supervisor/` | plans, calls both, composes the streamed answer | `ecom_supervisor` |

## Deploy — in this order (ingestion must be done first)

```bash
cd agent/nlq
python deploy.py

cd ../chart_gen
python deploy.py

cd ../supervisor
python deploy.py
```

NLQ creates the database's read-only user and the shared API-key secrets; the supervisor reads
NLQ's and chart_gen's `deployment.json` for their runtime ARNs.
