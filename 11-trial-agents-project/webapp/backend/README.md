# Trial agents — backend

```
browser ──cookie──► ALB /api/* ──► FastAPI (ECS, 1–2 tasks)
                                     │  task role, SigV4
                                     ├──► DynamoDB  trial-app      users, conversations,
                                     │                              messages, feedback
                                     └──► AgentCore supervisor      question + last 10
                                              │                     interactions + username
                                              ├─► trial_graph / trial_search
                                              └─► memory tools (DynamoDB + Pinecone)
```

## Routes

| Route | What |
|---|---|
| `POST /api/auth/login` | Sign in; an unknown username gets `404 {"code":"new_user"}` until first and last name are sent, then it is created |
| `POST /api/auth/logout`, `GET /api/auth/me` | Sign out; the signed-in user |
| `GET /api/conversations?q=` | Mine, newest first; `q` searches titles and questions |
| `POST`, `PATCH`, `DELETE /api/conversations/{id}` | Create, rename, delete (messages and feedback too) |
| `GET /api/conversations/{id}/messages` | Questions and answers; answers carry details and my feedback |
| `POST /api/chat` | One turn, streamed: `conversation` → `progress`… → `answer` or `error` |
| `POST /api/feedback`, `GET /api/feedback` | Rate an answer (resubmit replaces); my feedback |
| `GET /api/agentops/summary`, `/interactions`, `/feedback` | Tokens, cost, latency, agents, tools, trace links — mine, or everyone's for `ADMIN_USERS` |
| `GET /api/health`, `/api/docs` | Health check; Swagger |

An answer's `details`: `agents` (question, rationale, the Cypher that ran,
search stats), `tools` (memory reads and writes), `citations` (best re-ranked
first), `artifact` (table or graph), `memory` (what was recalled),
`decision`, `usage`, `cost_usd`, `latency_ms`, `trace_url`.

## Run locally (no AWS)

```bash
pip install -r requirements.txt
APP_STORE=memory SUPERVISOR_ARN=arn:... CORS_ORIGINS=http://localhost:5173 \
  uvicorn app.main:app --port 8000
```

`APP_STORE=memory` keeps everything in the process. A real `SUPERVISOR_ARN`
and AWS credentials are still needed to get answers.

## Tests

```bash
pytest tests          # uses the agents' repo tests/ for botocore payload validation
```
