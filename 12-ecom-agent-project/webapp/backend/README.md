# Web backend — FastAPI, streams turns from the supervisor, stores them in DynamoDB

```
browser ──/api/v1──► backend ──A2A message/stream (SigV4)──► AgentCore supervisor
                       │  SSE: meta · status · reasoning · token · artifact · entities · sources · done | error
                       └─► DynamoDB ecom-webapp: users · conversations · messages (details_json) · feedback
```

Each answer is saved with everything the screen shows: summary (text), reasoning, citation
(sources), table and charts (one Plotly figure per chart artifact). The turn runs as a background
task — a closed tab does not lose it — and is saved BEFORE `done` is sent.

## Run locally (PyCharm Terminal, project root)

```bash
cd webapp/backend
pip install -r requirements.txt
cp .env.example .env                 # set SESSION_SECRET to any long random string
python run.py                        # http://localhost:8000 · API docs at /api/docs
```

The first start creates the DynamoDB table `ecom-webapp` if it is missing.

## Try a turn without the frontend

```bash
curl -s -c /tmp/ecom.cookies -H 'Content-Type: application/json' localhost:8000/api/v1/auth/login \
  -d '{"username":"prudhvi","password":"a-long-password","first_name":"Prudhvi","last_name":"A"}'
curl -N -b /tmp/ecom.cookies -H 'Content-Type: application/json' localhost:8000/api/v1/chat/message \
  -d '{"text":"Which carrier delivers on time most often?"}'
```

## Test

```bash
cd webapp/backend
python -m pytest tests -q
```
