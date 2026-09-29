"""Trial agents web backend.

    browser ──cookie──► FastAPI (this package) ──SigV4──► AgentCore supervisor
                           │
                           └──SigV4──► DynamoDB (users, conversations,
                                       messages, feedback, interactions)

    settings.py   every environment variable, read once
    passwords.py  PBKDF2 password hashing (standard library only)
    session.py    signed session cookie + the CurrentUser dependency
    models.py     request/response contracts
    store/        Store interface; memory (tests, local) and dynamodb backends
    supervisor.py invoke_agent_runtime with history + user_id; parse the reply
    answers.py    SupervisorResponse -> what the UI shows (agents, queries,
                  tools, citations, usage, trace link)
    routers/      auth, conversations, chat (SSE), feedback, agentops, health
"""
