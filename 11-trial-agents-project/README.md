# Trial agents on Bedrock AgentCore

```
                           analyst question
                                  │
                                  v
 ┌──────────────────────── supervisor (A2A, :9000) ─────────────────────────┐
 │  route            PROBABILISTIC   OpenAI model picks a specialist          │
 │  render_decision  DETERMINISTIC   table -> chart, graph -> graph           │
 │  compose          PROBABILISTIC   writes the answer from bounded evidence  │
 └───────┬──────────────────────────────────────────────────┬───────────────┘
         │ invoke_agent_runtime (SigV4), A2A message/send    │
         v                                                   v
   trial_graph (A2A)                                   trial_search (A2A)
   NL -> Cypher                                        passages + expansion
         │ MCP over SigV4                                    │ MCP over SigV4
         v                                                   v
   Gateway -> Lambda -> Neo4j                 Gateway -> Lambda -> Pinecone (+ Neo4j NEXT)
```

## Where each agent's configuration comes from

```
env        PARAM_PREFIX=/trial-agents/<agent>        the only environment variable
SSM        /trial-agents/<agent>/*                    limits, gateway URL, ids, versions
SSM        /trial-agents/registry/<agent>             {arn, description} — written by each
                                                      specialist; read by the supervisor
Secrets    trial-agents/openai                        {"api_key", "model"} — shared
Bedrock    Prompt Management                          prompts/*.md, version pinned in SSM
Bedrock    Guardrail  trial-agents-guardrail          applied with ApplyGuardrail
```

Settings are loaded once, at container start. A missing parameter, a
placeholder secret or an unrendered prompt variable stops the container
from starting, and the reason is in `/aws/bedrock-agentcore/runtimes/`.

## Observability

```
container   opentelemetry-instrument python main.py      aws-opentelemetry-distro exports;
                                                          the Runtime presets OTEL env vars
spans       <agent>.request   root span per request, with session and result attributes
            LangGraph / model / tools                     openinference instrumentor
            supervisor.route / .render_decision / .compose / .call_agent
            guardrail.input / guardrail.output            one per ApplyGuardrail call
one trace   supervisor ─ invoke_agent_runtime(traceParent) ─► specialist attaches it
view        CloudWatch -> GenAI Observability -> Bedrock AgentCore; spans in aws/spans
```

The first deploy enables CloudWatch Transaction Search for the account
(`infra/observability.py`). Without it no spans appear, and nothing errors.
Spans can take about ten minutes to show after it is first enabled. The
deploying identity needs `logs:PutResourcePolicy` and
`xray:UpdateTraceSegmentDestination` for that one step.

## Deploy

```
  0  prerequisites (once)        IAM user or SSO role, aws configure, Docker Desktop
  │
  1  trial_graph   deploy.py  ── creates its stack, registers itself
  2  secrets                  ── trial-agents/openai, trial-graph/neo4j
  3  trial_graph   setup_neo4j.py ── the fulltext index
  │
  4  trial_search  deploy.py  ── needs trial-graph/neo4j and the index from 3;
                                 registers itself
     secret                   ── trial-search/pinecone
  │
  5  supervisor    deploy.py  ── reads the registry; deployed LAST
  │
  6  verify                   ── registry, runtimes, one question end to end
  7  observe                  ── CloudWatch GenAI Observability
```

The order matters: `trial_search` reuses `trial_graph`'s Neo4j secret, and
the supervisor's IAM role is scoped to whatever is in the registry when it
deploys.

### What every `deploy.py` does before creating anything

```
preflight           AWS credentials + region            from aws configure
                    Docker daemon, buildx, a linux/arm64 builder
                    every IAM action this deploy calls, simulated for YOU
                    -> stops and names what is missing; nothing created yet
observability       CloudWatch Transaction Search       first deploy only
ECR login           automatic, every deploy             the token lasts 12 h
```

Every step after that is idempotent. If a deploy fails part-way, fix the
cause and run the same command again: what exists is found and reused.

### Step 0 — Prerequisites (once)

1. **An IAM user or SSO role, not root.** Root access keys cannot be
   restricted by any policy, and the preflight cannot check them. Create an
   IAM user or SSO role with administrator access, then:
   ```bash
   aws configure            # or: aws configure sso
   aws sts get-caller-identity    # must NOT end in :root
   ```
2. **Docker Desktop running.** On Apple Silicon, linux/arm64 builds
   natively. On an Intel machine, if the preflight says the builder cannot
   build linux/arm64, run once:
   ```bash
   docker run --privileged --rm tonistiigi/binfmt --install arm64
   ```
3. **Deploy-time Python packages** (in your virtualenv):
   ```bash
   pip install boto3 neo4j httpx
   ```
   `httpx` is for `deploy.py --image`, which copies a prebuilt image from
   Docker Hub into ECR without Docker.

### Prebuilt images — deploying without Docker

The instructor publishes every image once (`docker login` first):

```bash
python publish_images.py --user <dockerhub-user> --tag 1.0
```

It runs each component's `deploy.py --publish`, then checks each image can be
pulled anonymously — a private Docker Hub repository is reported, not passed.
Students then add `--image` to every deploy and need no Docker at all:

```bash
cd trial_graph  && python deploy.py --image <dockerhub-user>/trial-graph-agent:1.0
cd ../trial_search && python deploy.py --image <dockerhub-user>/trial-search-agent:1.0
cd ../supervisor   && python deploy.py --image <dockerhub-user>/trial-supervisor-agent:1.0
cd ../webapp/deploy && python deploy.py --image <dockerhub-user>/trial-webapp-backend:1.0 \
                       --frontend-image <dockerhub-user>/trial-webapp-frontend:1.0
```

The frontend image holds only the built files (`FROM scratch` + `/dist`); the
web app deploy uploads them to S3, so students need neither Docker nor Node.

### Step 1 — Deploy trial_graph

```bash
cd trial_graph && python deploy.py
```

It ends with:

```
done. trial_graph runtime: arn:aws:bedrock-agentcore:us-east-1:<account>:runtime/trial_graph-XXXXXXXXXX
```

and two reminders that `trial-graph/neo4j` and `trial-agents/openai` still
hold placeholders. That is expected: the next step sets them.

Created: Neo4j and OpenAI secrets (placeholders), the tools Lambda, the
Gateway and its target, the shared guardrail, the `trial-graph-system`
prompt, parameters under `/trial-agents/trial_graph`, the ECR repository
and image, the Runtime, and `/trial-agents/registry/trial_graph`.

### Step 2 — Set the secrets trial_graph needs

```bash
aws secretsmanager put-secret-value --secret-id trial-agents/openai \
    --secret-string '{"api_key":"sk-...","model":"gpt-6-sol"}'

aws secretsmanager put-secret-value --secret-id trial-graph/neo4j \
    --secret-string '{"uri":"neo4j+s://<id>.databases.neo4j.io","user":"neo4j","password":"..."}'
```

`model` is the API model ID, not the display name — `gpt-6-sol`, not
"GPT-6 Sol". `trial-agents/openai` is shared by all three agents and by
trial_search's embedding Lambda.

Secrets are read when a container starts. Setting them needs no redeploy.

### Step 3 — Create the Neo4j fulltext index

```bash
python setup_neo4j.py
```

Expected output:

```
  index 'trial_entity_names': ONLINE
```

Both name lookups query this index: trial_graph's `find_entity_by_name`
and trial_search's `resolve_trial`. Without it, every name lookup fails.
Running it again is safe (`IF NOT EXISTS`).

### Step 4 — Deploy trial_search, then set its secret

```bash
cd ../trial_search && python deploy.py --pinecone-index rag-docs

aws secretsmanager put-secret-value --secret-id trial-search/pinecone \
    --secret-string '{"api_key":"pcsk_..."}'

# Cohere Rerank — key from dashboard.cohere.com -> API keys
aws secretsmanager put-secret-value --secret-id trial-search/cohere \
    --secret-string '{"api_key":"...","model":"rerank-v3.5"}'
```

Check the name lookup against the real graph (no Lambda, no agent — the
handler's own Cypher, with your AWS credentials):

```bash
python check_resolve.py
python check_resolve.py "IMbrave150" "the glaucoma trial"
```

No prompt lists the trials. `resolve_trial` reads a trial's NCT number,
title and protocol `doc_id` from the graph when a question names it, so a
protocol added to the graph and the index is findable with no prompt change.

`--pinecone-index` must be the index the RAG pipeline wrote to.

Search is recall, then precision: Pinecone returns a pool of 40 by vector
similarity, and Cohere Rerank keeps the best `top_k`. Until the Cohere key is
set — or when Cohere refuses a call — searches still work, in vector order,
and each result says `"reranked": false` with the reason. A trial key allows
10 rerank calls a minute and 1,000 API calls a month, so under load expect
some `reranked: false` results; a production key removes that. The key is
read on the next search after you set it, with no redeploy. If
`trial-graph/neo4j` does not exist yet, the deploy stops and says to deploy
trial_graph first.

### Step 5 — Deploy the supervisor

```bash
cd ../supervisor && python deploy.py
```

Its first step lists the specialists it found:

```
=== STEP 1: registry ===
  trial_graph    arn:aws:bedrock-agentcore:...:runtime/trial_graph-...
  trial_search   arn:aws:bedrock-agentcore:...:runtime/trial_search-...
```

If a specialist is missing from that list, deploy it and run this again.

### Step 6 — Verify

From the project root:

```bash
cd ..

# both specialists registered
aws ssm get-parameters-by-path --path /trial-agents/registry \
    --query 'Parameters[].Name'

# all three runtimes READY
aws bedrock-agentcore-control list-agent-runtimes \
    --query 'agentRuntimes[].[agentRuntimeName,status]' --output table
```

Each agent folder also has a `deployment.json` with its runtime ARN,
parameter prefix, guardrail and prompt versions.

**One question end to end.** The session ID must be 33 to 256 characters.

```bash
cat > request.json << 'EOF'
{"jsonrpc":"2.0","id":"1","method":"message/send","params":{"message":{
  "role":"user","messageId":"msg-smoke-1",
  "parts":[{"kind":"text","text":"Which trials does Novo Nordisk sponsor?"}]}}}
EOF

aws bedrock-agentcore invoke-agent-runtime \
    --agent-runtime-arn "$(jq -r .runtime_arn supervisor/deployment.json)" \
    --qualifier DEFAULT \
    --runtime-session-id "smoke-test-$(uuidgen | tr -d -)" \
    --content-type application/json --accept application/json \
    --payload fileb://request.json \
    response.json

jq . response.json
```

A working answer should name Novo Nordisk's two trials here, PIONEER 4 and STEP 1.

### Step 7 — Observe

Open **CloudWatch → GenAI Observability → Bedrock AgentCore**. One question
is one trace:

```
invoke_agent supervisor
  supervisor → route → supervisor_route → ChatOpenAI
    tools → call_agent → call_agent trial_graph
        invoke_agent trial_graph        (the specialist's container)
          trial_graph → ChatOpenAI → tools → find_entity_by_name → ...
  render_decision → compose → ChatOpenAI
```

Clear the **Agent spans** quick filter to see the full tree; it keeps only
spans classified as agents, models and tools. Spans are stored in the
`aws/spans` log group. The first traces can take about ten minutes to
appear after Transaction Search is enabled.

### When to redeploy

| You changed | Do this |
|---|---|
| A secret (key, model, password) | Nothing. The next container start reads it |
| A limit or prompt version in Parameter Store | Nothing. Next cold start |
| A file in `prompts/` | Run that agent's `deploy.py` — a new version only if the text changed |
| Code in `agent_code/` or `lambda_tools/` | Run that agent's `deploy.py` |
| Added a specialist | Deploy it, then run `supervisor/deploy.py` so its IAM role includes it |

### If something goes wrong

| Symptom | Cause, and what to do |
|---|---|
| Preflight: `lacks these permissions` | Your identity lacks the listed actions. Nothing was created; get them granted and re-run |
| Preflight: `cannot build linux/arm64` | Run the `binfmt` command in Step 0 |
| Invoke returns `403 RuntimeClientError` | The container did not start. Read `/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT` in CloudWatch Logs |
| Log says `still holds placeholder` | A secret was not set — Step 2 or Step 4 |
| Log says `missing parameters under /trial-agents/...` | That agent's deploy did not finish; re-run it |
| Supervisor deploy: `nothing registered` | Deploy trial_graph and trial_search first |
| Name lookups fail with `no such fulltext schema index` | Run Step 3 |
| pip prints `ERROR: ... dependency conflicts` while building the Lambda | Your LOCAL virtualenv has packages pinned to an older `openai`. The Lambda is built into its own folder and is unaffected |
| Search results say `"reranked": false` | `rerank_note` says why: the key is still `replace-me`, HTTP 429 (trial key: 10 rerank calls/minute), or HTTP 401 (wrong key). The search still answered, in vector order |
| An answer shows `[?]` in a threshold | The protocol printed a symbol the PDF extraction could not translate. Known Symbol-font codes (≥ ≤ > < ± × µ …) are translated when a passage is read (`trial_search/lambda_tools/symbol_fonts.py`); `[?]` marks the rest. The composer is told never to guess the sign |
| No traces in CloudWatch | Wait ten minutes after the first deploy; check that `aws xray get-trace-segment-destination` says `CloudWatchLogs` |

## Known data issue: symbol-font codes in chunk text

Protocols set ≥, ≤, >, <, ±, × and µ in the Symbol font. The PDF extraction
kept those as Private Use Area codes (U+F0xx), which render as blank space:
966 of 5,764 chunks, in 10 of 20 protocols. `trial_search` translates them
when it reads a passage, so agents see "BP ≥ 150 mmHg".

The permanent fix is at ingestion: apply `symbol_fonts.normalize()` to the
chunk text right after `chunker.contextualize()` in the RAG pipeline. That
changes those chunks' text hashes and therefore their `chunk_id`s, so
re-ingest into Pinecone AND rebuild the Neo4j Chunk/NEXT graph together.

## Supervisor contract (what the backend sends and receives)

```
A2A message/send  →  supervisor
  parts[0].text          the question
  metadata.history       [{"role": "user"|"assistant", "text": "..."}]  oldest first;
                         the supervisor keeps the newest 20 (10 interactions),
                         each cut to 2,000 characters
  metadata.user_id       the signed-in username — scopes all memory
  metadata.traceparent   joins the backend's trace (optional)

SupervisorResponse (JSON in the reply's text part)
  composed_answer        the answer the analyst reads
  calls[]                specialist calls: agent, question, rationale, result_shape
  tool_calls[]           memory tools: remember_fact, recall_facts,
                         record_episode, recall_episodes — args, result
  results{}              per specialist: cypher, rows, nodes, passages (doc, page,
                         rerank_score) — the UI's queries and citations;
                         "memory:<kind>" entries for recalled memories
  decision               answerable, entities, note, resolved_question,
                         from_conversation
  usage, trace_id, history_turns, render_target
```

Memory lives in DynamoDB `trial-agents-memory` (records) and Pinecone
`trial-agents-memory` (vectors, namespaces `semantic` / `episodic`, filtered by
user). The supervisor's deploy creates both; the Pinecone key comes from
`trial-search/pinecone`. The agent decides when to read or write — see the
MEMORY section of `supervisor/prompts/system.md`.

## Changing a prompt

Edit `prompts/*.md` and run that agent's `deploy.py`. A new Prompt Management
version is created only if the text changed, and the pinned version in SSM
moves to it. To roll back, set `/trial-agents/<agent>/prompt_version` to an
earlier number; the next cold start uses it.

## Tests

```bash
pip install pytest -e trial_graph/agent_code   # or each agent's dependencies
TRIAL_DATA_DIR=<dir with dump.json and graph_dump.json> pytest tests
```

```
test_config.py       loaders: pagination, placeholders, missing params, variables
test_guardrail.py    INPUT/OUTPUT checks in a real async loop; evidence NOT checked
test_trial_graph.py  resolver anchor, write blocking, LIMIT, repair budget
test_trial_search.py the three Lambda tools on real data; loop budgets
test_supervisor.py   registry routing, budgets, hollow-decision guard, composer evidence
test_aws_calls.py    every boto3 call against boto3's service models
test_tracing.py      real trace ids and parent links via the OpenTelemetry SDK
test_payloads.py     every nested deploy payload through botocore's own validator
test_preflight.py    each preflight failure; ECR password on stdin only
test_openai_wire.py  the real ChatOpenAI over a mocked Responses API wire
e2e/test_e2e.py      supervisor -> both specialists as real A2A server processes,
                     and ONE trace across all three processes
```

## What the tests do not cover

No test calls AWS, OpenAI, Neo4j or Pinecone. The guardrail's filter
strengths are reasoned for clinical text, not measured against the live
guardrail. The graph currently holds no Drug nodes; drug questions are
answered as "not loaded", never as "none found".
