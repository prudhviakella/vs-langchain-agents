# E-commerce NLQ agent

```
12-ecom-agent-project/
├── ingestion/   data in:  raw CSVs ─► clean ─► RDS PostgreSQL (schema ecom)
│                                          └─► Pinecone knowledge base (ecom-kb)
└── agent/       answers out: question ─► knowledge base + SQL on PostgreSQL ─► answer
```

| Folder | What it is | Start with |
|---|---|---|
| `ingestion/` | prepares the data and loads PostgreSQL and Pinecone | [`ingestion/README.md`](ingestion/README.md) |
| `agent/` | the NLQ agents — next | [`agent/README.md`](agent/README.md) |

One virtual environment, `.venv` in this folder, serves both parts. In PyCharm's Terminal:

```bash
cd ingestion
python run.py prepare
```
