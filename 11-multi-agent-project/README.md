# A Real Multi-Agent System

Four agents over the data from modules 05 and 08, talking A2A.

```
local/    four processes on four ports — start here
aws/      four AgentCore Runtimes, plus tools over MCP
```

## Order

**`local/` first.** Four agents on localhost, no AWS account, no deployment. It
teaches the architecture: the shape contract, splitting judgement from mechanism,
keeping bulk data away from the model.

**`aws/` second.** The same agents, deployed. The agent code does not change —
only where the endpoints come from.

That is the point of doing it in this order. If porting to AWS had required
rewriting the agents, the local version was teaching a toy.

## What each holds

| | `local/` | `aws/` |
|---|---|---|
| Transport | HTTP on 9200–9203 | AgentCore Runtime, SigV4 |
| Tools | none | Lambda, exposed as MCP by a Gateway |
| Auth | none | SigV4 agent-to-agent, JWT into the Gateway |
| Needs | an OpenAI or Anthropic key | an AWS account |

## The ideas, in both

**The contract is the system.** Four programs sharing no code agree on one field
— `shape` — and that is enough to build on.

**Split the decisions.** Routing is a judgement, so a model does it. Charting a
table is not, so code does. A model that forgets a `render` tool ships a
chartless answer and nothing errors.

**Keep bulk data away from the model.** It routes and writes prose. Data travels
between agents, not through it.

**Tools are not agents.** A tool does exactly what it is told; an agent decides
what to do. Wrap one as the other and you lose something either way.
