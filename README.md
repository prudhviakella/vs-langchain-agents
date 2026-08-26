# Applied GenAI Engineering

Course materials, in teaching order. Each module stands alone; each assumes the one
before it.

| # | Module | What you build |
|---|---|---|
| 01 | `01-agents-foundations` | Models, prompting, tools, memory, multimodal |
| 02 | `02-agents-orchestration` | State, MCP, multi-agent, RAG and SQL agents |
| 03 | `03-agents-production` | Human-in-the-loop, dynamic prompts and tools, streaming, a chat UI |
| 04 | `04-vector-databases` | Pinecone from zero, and hybrid search |
| 05 | `05-rag` | Turn PDFs into a searchable index, then query it |
| 06 | `06-rag-evaluation` | Measure whether the retrieval is any good, with MLflow |
| 07 | `07-graph-databases` | Neo4j and Cypher, twelve parts and a capstone |
| 08 | `08-graphrag` | A knowledge graph over the documents from module 05 |
| 09 | `09-rag-at-scale-aws` | Module 05 running a whole corpus on Fargate |
| 10 | `10-a2a-protocol` | Agents talking to agents across a network |
| 11 | `11-multi-agent-project` | Four agents over the data from 05 and 08 |

## Setup

```bash
cp .env.example .env      # then fill it in
pip install -r requirements.txt
```

`.env` is gitignored. Keep it that way.

## What each module needs

| Module | Needs |
|---|---|
| 01–03 | Anthropic or OpenAI key, Tavily for search |
| 04 | Pinecone key |
| 05 | OpenAI key, Pinecone key |
| 06 | the same, plus a local MLflow server |
| 07 | a free Neo4j Aura instance |
| 08 | all of the above, and the index from module 05 |
| 09 | an AWS account, and a Fargate vCPU quota above 6 |
| 10 | nothing beyond Python — it runs entirely on localhost |
| 11 | modules 05 and 08 running, plus an OpenAI or Anthropic key |

## How 05 through 09 fit together

Module 05 builds a vector index from PDFs and queries it. Module 09 is the same code
running that at scale on AWS — optional, and nothing later depends on it.

**Module 06 is where the course turns from building to measuring.** Everything in 05
is a decision — chunk size, how many candidates to fetch, whether to rerank — and
none of it can be judged without a labelled set and a number. Module 06 supplies
both, on HotpotQA rather than the clinical corpus, so the questions have known
answers.

It teaches three families of metric, and the distinction matters:

| Family | Asks | Costs |
|---|---|---|
| Retrieval | did the search find the right chunks | nothing |
| Lexical | does the answer match the reference word for word | nothing |
| LLM-as-judge | is the answer grounded, complete, sensible | API calls |

Module 07 teaches Neo4j and Cypher on its own dataset. Module 08 then builds a graph
over the *same documents* module 05 indexed, joined to that index by chunk id.

The two stores answer different questions:

| | Vector index | Graph |
|---|---|---|
| Good at | what does this passage say | what connects to what |
| Fails at | joins across documents | returning a passage |

Neither replaces the other, which is why module 08 reads the index rather than
rebuilding it.

## Conventions

Module 07 teaches the Neo4j naming conventions and module 08 follows them:
`UpperCamelCase` labels, `UPPER_SNAKE_CASE` relationship types, `lowerCamelCase`
properties.

The vector store keeps its own field names — `chunk_id`, `doc_id` — because the two
are joined on the *value* of an id, not on the property name.

## A note on credentials

Nothing in this repository holds a key. If you find one, it should not be there.

Neo4j Aura shows its password once, when the instance is created. The Cypher course
asks for it with `getpass` so it never gets saved into the notebook file — a notebook
with a password in cell 3 is the most common way these leak.
