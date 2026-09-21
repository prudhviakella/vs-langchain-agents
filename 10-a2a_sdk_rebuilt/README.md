# A2A Mini Project — rebuilt with the official A2A Python SDK

This is a replacement for the original `a2a_mini` + `aiohttp` implementation.
The three teaching agents keep the same behavior, but the protocol plumbing is
provided by the official `a2a-sdk`.

## Agents

| Agent | Port | Demonstrates |
|---|---:|---|
| Echo | 9101 | Agent Card, Task, artifact, completion |
| Counter | 9102 | Long-running task, streaming/artifact updates, cancellation |
| Planner | 9103 | `input-required`, same-task continuation, task history |

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

Use three terminals:

```bash
python agents/echo_agent.py
python agents/counter_agent.py
python agents/planner_agent.py
```

Each server exposes its Agent Card at:

```text
http://127.0.0.1:9101/.well-known/agent-card.json
http://127.0.0.1:9102/.well-known/agent-card.json
http://127.0.0.1:9103/.well-known/agent-card.json
```

## What changed from the original

Original:

```text
aiohttp
  -> custom JSON-RPC
  -> custom AgentCard
  -> custom Message/Task/Artifact
  -> custom EventQueue
  -> custom SSE
```

SDK version:

```text
A2A SDK
  -> DefaultRequestHandler
  -> InMemoryTaskStore
  -> AgentExecutor
  -> EventQueue
  -> TaskUpdater
  -> AgentCard routes
  -> JSON-RPC routes
  -> Starlette/Uvicorn transport
```

Your code is now primarily the `AgentExecutor` business logic.

## A2A 1.0 execution rule

The SDK supports two response patterns:

1. Message-only: enqueue exactly one `Message`.
2. Task lifecycle: enqueue a `Task` first, then status/artifact updates until a
   terminal or interrupted state.

These examples intentionally use the task-lifecycle pattern so the concepts
from the original mini implementation remain visible.

## Planner continuation

The important flow is:

```text
Request 1
  "plan a trip to Kyoto"
       |
       v
Planner Task-001
       |
       +--> input-required
             "How many days?"

Request 2
  "3 days"
  same task/context
       |
       v
Planner Task-001
       |
       +--> working
       +--> artifact: itinerary
       +--> completed
```

The application-level user conversation ID is still your responsibility. The
A2A SDK's `context_id` is the A2A execution context identifier; do not assume it
is automatically your product's `CONV-001` unless you deliberately make that
mapping in your application.
