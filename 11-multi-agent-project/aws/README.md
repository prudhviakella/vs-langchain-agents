# Deploying the fleet to AWS

The local module runs four agents on four ports. This runs them as four
AgentCore Runtimes, and adds tools they can call over MCP.

```
        ┌──────────────────────────────────────────┐
        │  you  ──SigV4──▶  supervisor runtime      │
        └───────────────────────┬──────────────────┘
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 vdb        cypher       chart          AgentCore Runtime
                    │
                    └──MCP──▶ Gateway ──▶ Lambda        AgentCore Gateway
```

## Steps

```bash
pip install -r requirements.txt
npm install -g @aws/agentcore        # or use the Python starter toolkit

python 1_deploy_tools.py             # Lambda
python 2_create_gateway.py           # Gateway turns it into MCP tools
python 3_deploy_agents.py            # four runtimes
python 4_invoke.py "which sponsors run more than one trial?"

python status.py
python teardown.py
```

Each script writes what it made to `deployed.json`, so nothing has to be pasted
between steps.

## What changes from local, and what does not

**Not the agents.** `execute()` is identical, the cards are identical, the shape
contract is identical. One line of configuration changes: an endpoint that was
`http://127.0.0.1:9202` becomes a runtime ARN.

That is the test of whether the local module taught the right thing. If porting
had required rewriting the agents, the local version was a toy.

## Auth, in two places, and they are different

**Agent to agent — SigV4.** Runtime is invoked through the AgentCore data plane
and signed with your IAM credentials. No Cognito, no token exchange.

Worth being plain about the consequence: **SigV4 authenticates an IAM principal,
not a person.** The supervisor calls specialists as itself, so a specialist
cannot apply per-user authorisation — it does not know who asked. Fine when every
user sees the same data. A real limit if they should not, and the fix then is
JWT propagation rather than SigV4.

**Gateway — JWT inbound, IAM outbound.** The Gateway wants a bearer token from
callers, and uses its own IAM role to invoke the Lambda. So `2_create_gateway.py`
creates a Cognito authorizer, and this is the one place Cognito appears.

If you want to avoid it entirely: call the Lambda directly from the agent and
skip the Gateway. You lose MCP discovery and gain one less moving part. Worth
knowing that is a real option rather than a compromise.

## Why the Gateway at all

It is a managed MCP server. Point it at a Lambda and it exposes each handler as
an MCP tool with a schema, discoverable at an endpoint.

No MCP server to write, host, patch or scale. And because the endpoint is
standard MCP, the same tools work in Claude Desktop, Cursor, or anything else
that speaks it — not only in your agents.

## Tools are not agents

`tools/trial_tools.py` holds two functions. Neither has a model.

```
a tool    does exactly what it is told, every time
an agent  decides what to do
```

`enrollment_stats` exists because summing five numbers is exactly what an agent
does badly on its own — it will add them in its head and occasionally get it
wrong in a way that reads perfectly plausibly. Arithmetic belongs in code.

The vector search and the Cypher query are **agents**, not tools, because each
holds a model and makes a judgement. Getting that line wrong is the common
mistake: wrap an agent as a tool and you lose its streaming and its task
lifecycle; wrap a tool as an agent and you pay for a model that decides nothing.

## Deployment order matters

Specialists first, supervisor last. The supervisor reads their ARNs from its
environment, so deploying it first gives it an empty fleet — and it starts fine,
and answers every question with "no agents reachable", which looks like a routing
bug rather than a deployment order problem.

## Read before running

**These scripts have not been run against a live account.** They are written from
the AgentCore documentation and follow the same shape as the module 09 AWS
scripts, which have been. Expect to fix at least one thing — most likely an API
name, since AgentCore is moving quickly.

**The AgentCore CLI (`@aws/agentcore`) has replaced the Python starter toolkit**
for new projects. `3_deploy_agents.py` shells out to `agentcore configure` and
`agentcore launch` because that is what the AWS documentation describes, and a
wrapper that hid it would make the docs harder to follow rather than easier.

**Costs.** Four runtimes idle at a session timeout, a Lambda, and a Gateway. Run
`teardown.py` when you are done. Cognito is reported rather than deleted, because
a user pool may be shared with something else.
