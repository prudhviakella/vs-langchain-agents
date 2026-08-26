# The A2A Protocol

Modules 01 to 03 built agents that lived in one process, sharing a state object
and a framework. A2A is for when they do not.

```
your agent  ──▶  a different agent, on a different server,
                 written by a different team, in a different framework
```

The two sides share nothing — no memory, no prompts, no tools. Only a card and
some messages cross the boundary.

## Run it

```bash
pip install -r requirements.txt
jupyter lab 01_a2a_protocol.ipynb
```

The notebook starts all three agents in-process. To run them the normal way,
three terminals:

```bash
python agents/echo_agent.py       # :9101
python agents/counter_agent.py    # :9102
python agents/planner_agent.py    # :9103
```

## First: do you need this?

Usually not.

If your workflow is deterministic — you know which tools run in what order — a
function call beats an HTTP round trip, a task lifecycle and a card fetch, every
time.

A2A earns its cost in one situation: **the agents are deployed separately, by
different teams, and you do not control both sides.** Then you need a contract
instead of an import.

## MCP or A2A

They are not competitors.

| | MCP | A2A |
|---|---|---|
| Connects | your agent to its tools | your agent to another agent |
| The other side is | a function you call | a system that thinks |
| You control it | yes | often not |

Production systems run both.

## What is here

| File | What |
|---|---|
| `a2a_mini/types.py` | The four objects: AgentCard, Message, Task, Artifact |
| `a2a_mini/queue.py` | The event queue — read this one twice |
| `a2a_mini/server.py` | JSON-RPC + SSE, about 150 lines |
| `a2a_mini/client.py` | discover, send, stream, resubscribe |
| `agents/echo_agent.py` | Card, `message/send`, artifact |
| `agents/counter_agent.py` | Queue, streaming, resubscribe |
| `agents/planner_agent.py` | `input-required`, resuming a task |

No SDK. The whole protocol is about 400 lines because A2A was deliberately built
on things that already existed — HTTP, JSON-RPC 2.0, Server-Sent Events.

## The one idea to take away

**A task outlives the HTTP request that started it.**

Everything else follows from that. The task needs an id, so you can ask about it
later. Events need a queue, because the agent produces them whether or not
anyone is listening. And once there is a queue, you get things that are
otherwise impossible:

- the connection drops and the work continues
- a client reattaches and catches up with no gap
- two clients watch the same task

Wire an agent straight to an HTTP response instead, and a dropped connection
kills work that was running perfectly well.

The counter agent demonstrates this directly: start a count, disconnect at step
3, wait, then `tasks/resubscribe`. Steps 1–3 replay instantly, 4 onward arrive
live. The count never paused and never restarted.

## `input-required`

The state people forget, and the one that separates a task from a function call.

A function that needs more input has to fail and be called again, losing
everything it had. A task pauses, keeps its history, and continues when the
answer arrives — an hour later, from a different process.

The planner agent asks how long the trip is. You answer on the same task id, and
never re-send the destination: it is still on the server.

## What is missing, deliberately

**Auth infrastructure.** The card declares a security scheme; this ignores it.
Real deployments use OAuth 2.0 and check every call.

The *principle* is in the notebook, because it is protocol-level and easy to get
wrong: when an agent calls another agent, forward the **caller's** token, do not
mint a service token. Otherwise every downstream check sees the supervisor
rather than the person, and one user's access quietly becomes everyone's.

**Push notifications.** For work measured in hours, a webhook beats an open
stream.

**A durable queue.** Ours is a Python list. Restart the process and running tasks
are gone. Redis or a database, in production.

**Signed cards.** v1.0 added cryptographic signing, so a card can be verified as
coming from who it claims.
