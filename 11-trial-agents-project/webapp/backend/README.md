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

## Run locally — everything is stored in DynamoDB

```
where it runs        APP_TABLE          DynamoDB table          created by
your machine         not set            trial-webapp-local      this backend, on its first start
AWS (deployed)       set by the stack   trial-webapp            the CloudFormation stack
```

Users, conversations, messages and feedback always go to DynamoDB — there is
no in-memory mode. A local run uses its own table, `trial-webapp-local`, and
creates it on first start (about 10 seconds, once). It is a separate table so
that the CloudFormation stack can still create `trial-webapp` later.

Your AWS credentials (`aws configure`) are used for DynamoDB and for calling
the supervisor. Use the supervisor ARN printed by `supervisor/deploy.py`
(also in `supervisor/deployment.json`).

**Mac**
```bash
cd ~/PycharmProjects/vs-langchain-agents/11-trial-agents-project/webapp/backend
pip install -r requirements.txt
SUPERVISOR_ARN=<supervisor ARN> CORS_ORIGINS=http://localhost:5173 uvicorn app.main:app --port 8000
```

**Windows**
```powershell
cd $HOME\PycharmProjects\vs-langchain-agents\11-trial-agents-project\webapp\backend
pip install -r requirements.txt
$env:SUPERVISOR_ARN="<supervisor ARN>"; $env:CORS_ORIGINS="http://localhost:5173"
uvicorn app.main:app --port 8000
```

The start log says which table it used:
```
INFO app.main DynamoDB table 'trial-webapp-local' created (region us-east-1)     first start
INFO app.main DynamoDB table 'trial-webapp-local' ready (region us-east-1)       every start after
```

Restarting the backend signs you out (a local run has no shared signing key),
but your user and conversations are still in the table — sign in again. To
stay signed in across restarts, also set `SESSION_SECRET` to any long random
string.

To work locally against the **deployed** app's data instead, also set
`APP_TABLE=trial-webapp`. A table named this way is never created by the
backend; if it does not exist, the start stops and says so.

## Tests

```bash
pytest tests          # uses the agents' repo tests/ for botocore payload validation
```
