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

Prerequisites: `aws configure` done, Docker Desktop running. Each `deploy.py`
starts with a preflight that stops before creating anything if something is
missing, and names it:

```
preflight   AWS credentials + region    from aws configure
            Docker daemon, buildx, a linux/arm64 builder
            every IAM action the deploy calls, simulated for YOU — including
            logs:PutResourcePolicy and xray:UpdateTraceSegmentDestination
            while Transaction Search is still off
then        ECR docker login (automatic, every deploy — the token lasts 12 h)
            Transaction Search (automatic, first deploy only)
```

```bash
cd trial_graph  && python deploy.py                            # registers itself
cd trial_search && python deploy.py --pinecone-index rag-docs  # registers itself
cd supervisor   && python deploy.py                            # reads the registry

aws secretsmanager put-secret-value --secret-id trial-agents/openai \
    --secret-string '{"api_key":"sk-...","model":"<openai model name>"}'
aws secretsmanager put-secret-value --secret-id trial-graph/neo4j \
    --secret-string '{"uri":"neo4j+s://...","user":"neo4j","password":"..."}'
aws secretsmanager put-secret-value --secret-id trial-search/pinecone \
    --secret-string '{"api_key":"..."}'
cd trial_graph && python setup_neo4j.py     # the fulltext index, once Neo4j is set
```

Secrets are read at container start, so setting them needs no redeploy.

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
e2e/test_e2e.py      supervisor -> both specialists as real A2A server processes,
                     and ONE trace across all three processes
```

## What the tests do not cover

No test calls AWS, OpenAI, Neo4j or Pinecone. The guardrail's filter
strengths are reasoned for clinical text, not measured against the live
guardrail. The graph currently holds no Drug nodes; drug questions are
answered as "not loaded", never as "none found".
