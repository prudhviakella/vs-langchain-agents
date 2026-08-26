# A Real Multi-Agent System

Module 10 taught the protocol with agents that reversed strings. This is the same
protocol with agents that do something.

```
                    supervisor  (9200)
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
     vdb (9201)     cypher (9202)    chart (9203)
     Pinecone         Neo4j          Plotly spec
     module 05        module 08
```

Nothing new is built here. Modules 05 and 08 made the capabilities; this wraps
each in an A2A agent so a supervisor can use them without importing them.

**The agents share no code at runtime.** They could be four repositories, four
languages, four accounts. What holds them together is a contract in the message.

## Run it

```bash
pip install -r requirements.txt

export PROVIDER=openai            # or anthropic
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=pc-...  INDEX_NAME=rag-docs
export NEO4J_URI=... NEO4J_USER=neo4j NEO4J_PASSWORD=...

jupyter lab 01_multi_agent_system.ipynb
```

The notebook hosts all four in-process. Normally each is its own terminal:

```bash
python agents/vdb_agent.py        # :9201
python agents/cypher_agent.py     # :9202
python agents/chart_agent.py      # :9203
python agents/supervisor.py       # :9200
```

Needs the index from module 05 and the graph from module 08.

## Either provider

`PROVIDER=openai` or `PROVIDER=anthropic`. Both paths are written; one variable
picks. The architecture does not care which model routes the questions, and
saying so in code is more convincing than saying it in a slide.

Structured output is enforced by the API in both — a JSON response format for
OpenAI, a forced tool call for Anthropic. "Respond with only JSON" is a request
the model grants most of the time, and most of the time is not a contract.

## The contract is the system

`agents_lib/contracts.py` is the shortest file here and the most important.

The supervisor cannot see inside a specialist. Not its memory, not its prompt,
not what it did. All it gets is an artifact — so the artifact has to say what it
is:

```python
{"shape": "table",    "columns": [...], "rows": [...], "row_count": 43}
{"shape": "passages", "passages": [...], "count": 5}
{"shape": "empty",    "reason": "no trials match that sponsor"}
{"shape": "error",    "reason": "neo4j refused the connection"}
```

`empty` and `error` are separate deliberately. *"No trials match"* is an answer.
*"The database is down"* is a failure. Collapse them and the supervisor tells
someone there is no data when the database was down — the worse of the two, and
the one you get by default.

## The shape

```
question
  -> AvailableAgents      who exists, discovered this turn
  -> RequireAgentCall     no "answered" decision with zero calls
  -> AgentCallBudget      at most N specialist calls
  -> NoRepeatCall         no re-asking an agent that already answered
  -> model                ONE tool: call_agent
  -> artifacts land in STATE; the model sees only summaries
  -> Decision             what it found, NOT the data
  -> render()             charts, in CODE, after the loop
  -> compose()            the prose answer
```

## One tool, not one per agent

```python
call_agent(agent_name, question, rationale, observation)
```

The agent NAME is data. Which agents exist comes from discovery — the supervisor
fetches their cards each turn, and what it finds becomes the AVAILABLE AGENTS
block in the prompt.

So adding an agent is configuration. No new `@tool`, no capability list to
update, and therefore no second list to drift out of step with the first.

And an agent that is down cannot be routed to: it is absent from the block the
model reads, so the model cannot pick something that is not there.

## It loops

`create_agent` gives a loop, not a router. The model calls a specialist, sees the
summary, and decides again.

```
plan  ->  execute  ->  replan  ->  answer
```

*"Which sponsors run more than one trial, and what do their protocols say?"*
needs the graph first and then document search, and the second question depends
on the first answer. A router cannot do that.

### `ToolStrategy`, not a bare schema

```python
response_format=ToolStrategy(schema=Decision)
```

Passing the schema alone lets `create_agent` pick a strategy, and for models with
native structured output it picks the provider's constrained decoding. That
forces schema-conforming JSON — **and a response constrained to emit JSON cannot
emit a tool call.**

The result is a model that answers on turn one having called nothing: every field
filled with no data behind it, `answerable: true`, a confident non-answer.

`ToolStrategy` makes the schema *another tool* beside `call_agent`, so the model
chooses between calling an agent and answering. You can see it in the binding:
`['call_agent', 'ask_user', 'Decision']`.

## Render and compose sit OUTSIDE the loop

The model finishes deciding. Then code charts anything table-shaped. Then a
second model call writes the prose.

Neither can be forgotten, because neither is a tool.

### The composer is a different call, and sees different data

The planner never sees rows, because it LOOPS — anything it holds could be
retyped into a follow-up query.

The composer runs ONCE, terminally. Given only counts it writes *"the query
returned 2 rows"*, which is a status report, not an answer. So it gets a
**bounded, verbatim** slice and is told to quote exactly and never compute.

The safety properties still hold: bounded, terminal, verifiable against the table
rendered beside it, and it cannot query — so a misreading cannot become a wrong
query.

## Bulk data never reaches the model

The specialist returns 43 rows. The supervisor keeps them and sends the model:

```
table: 43 rows, columns ['sponsor', 'trials']. First 3: [...]
```

The full set goes to the chart agent by reference.

A model handed 43 rows will sometimes retype values into its answer, and a
retyped NCT number that is one digit wrong reads exactly like a correct one. No
error, no warning, and someone acts on it.

## State, with reducers chosen per field

Results go into `SupervisorState` via `Command(update=...)`, not into a module
variable. Two tool calls can land in the same step, so a plain dict key raises
`INVALID_CONCURRENT_GRAPH_UPDATE` — but *which* reducer is a question about
meaning:

| field | reducer | why |
|---|---|---|
| `calls` | `operator.add` | a list that accumulates |
| `captured` | union | each entry is a different call's artifact; last-write-wins would silently drop one |
| `clarify` | last-wins | only one question can be put to the user |

Same bug class, three different correct answers.

And `call_id` comes from `tool_call_id`, never from `len(state["calls"]) + 1` —
two parallel calls would compute the same id, and `{**old, **new}` with the same
key overwrites regardless of the reducer.

## Four policies the model cannot route around

A prompt saying "call at most three agents" is a suggestion. The model follows it
most turns and ignores it on the turn where the question is hard — which is
exactly the turn where an unbounded loop costs money.

| Middleware | Refuses |
|---|---|
| `AvailableAgents` | — injects who exists, so the prompt holds no agent list |
| `RequireAgentCall` | a final "answered" decision with zero calls made |
| `AgentCallBudget` | the fourth specialist call |
| `NoRepeatCall` | re-asking an agent that already answered, without saying what it missed |

`RequireAgentCall` is the one worth understanding. The failure it exists for:

```
tool_calls: []          never attempted a call
calls: []               no agent ran
answerable: true        ...and yet reported as answered
```

The model works out which agent the question needs, then emits its decision on
the same turn instead of calling anything. What reaches the user is prose
indistinguishable from a real search that came back empty. No search ran.

Schema validation cannot catch it — `answerable: true` is a correctly typed bool,
and a validator sees only its own fields, never `state["calls"]` where the
evidence is. The check has to happen where both are visible.

`NoRepeatCall` only counts calls that **produced something**. Counting a failed
call arms the guard wrongly: the model gets told an agent already answered when
the call failed and captured nothing.

### Three things that cost me an hour

**`request.tool_call["name"]`, not `request.tool_name`.** The latter does not
exist, and a getattr default gives every call the same empty key — so a distinct
call looks like a repeat and the loop stops with no error.

**Return a `ToolMessage`, not a string.** A bare string is inserted as the wrong
message type and the model reads its own refusal as if a person had said it.

**`awrap_*`, not `wrap_*`.** LangChain dispatches on how the agent was invoked.
Define only the sync hooks, call `ainvoke`, and it raises `NotImplementedError`
from inside the graph — which reads as a LangGraph problem rather than a missing
method.

## Two guards, both in code

**The Cypher agent refuses writes.** A model writes the query — that is the point
and the risk. `_is_read_only()` checks before it runs.

The split matters: the prompt can be edited, tuned, replaced. The guard cannot.
Anything that must hold regardless of what the model produces belongs in code,
where it is reviewed and shows up in a diff.

It matches whole words, so `t.deleted_at` and a sponsor called "Reset Pharma" are
allowed. A guard that fires on legitimate queries gets switched off, and then it
protects nothing.

**The chart agent sanitises its own spec.** Plotly renders HTML in titles and
labels, so a string in the spec reaches a browser's DOM. The values being charted
come from the data — sponsor names, site names. The realistic path is not the
model misbehaving, it is the model faithfully copying a name field into a label.

`<br>`, `<b>`, `<sub>` survive because analysts rely on them. `<script>` loses its
contents as well as its tags.

## What is missing

**Auth.** Module 12 adds SigV4, which authenticates an IAM principal rather than
a person — so the supervisor calls specialists as itself and there is no per-user
authorisation downstream. Fine when every analyst sees the same data; a real
limit if they should not.

**Durability.** Restart the supervisor and running tasks are gone.

**Retries.** A timeout fails the turn. Retry transient network faults, not agent
errors — retrying a refusal burns the budget to be refused again.

**Budget enforcement.** `MAX_AGENT_CALLS` exists but this supervisor makes one
call per turn. A plan-and-replan loop needs it, in the tool rather than the
prompt: a prompt asking for efficiency is a suggestion; a counter that refuses
the fifth call is a wall.
