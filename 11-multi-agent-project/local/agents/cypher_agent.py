"""Cypher Agent — a LangChain agent that queries the Neo4j graph.

    python agents/cypher_agent.py     serves on :9202

An agent, not a function call. It writes Cypher, runs it, and if the query fails
it sees the error and can fix it — which is the whole reason to give it a loop
rather than a single structured call.

    write  ──▶  run  ──▶  error?  ──▶  rewrite  ──▶  run

A single call cannot do that. It produces one query, and a typo or a wrong
property name becomes a failed turn instead of a corrected one.

THE GUARD IS NOT IN THE PROMPT

A model writes the query. That is the point and the risk, so `run_cypher` checks
before executing.

The split is worth holding on to: the prompt can be edited, tuned, replaced. The
guard is code — reviewed, diffed, and unchanged by anything the model does. Put
"do not write to the database" only in the prompt and you have a request, not a
constraint.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "10-a2a-protocol"))

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from a2a_mini import A2AServer, AgentCard, Artifact, EventQueue, Message, Part, Skill, Task
from agents_lib import config, contracts, llm

CARD = AgentCard(
    name="cypher",
    description="Answers questions about the trial graph by writing and running Cypher.",
    url=config.endpoint("cypher"),
    capabilities={"streaming": True, "pushNotifications": False},
    skills=[Skill(
        id="query-graph", name="Query the trial graph",
        description=("Answers questions about relationships and counts across "
                     "trials: which sponsors run several, which trials share a "
                     "condition or a drug, where trials are conducted. Not for "
                     "what a protocol document says."),
        examples=["which sponsors run more than one trial?"])],
)

SCHEMA = """
(:Trial {nctId, phase, overallStatus, enrollmentCount})
(:Sponsor {name})  (:Disease {name})  (:Drug {name})
(:Country {name})  (:Site {facility, city})  (:MeSHTerm {term})
(:Document {docId})  (:Section {heading})  (:Chunk {chunkId, page})

(:Trial)-[:SPONSORED_BY]->(:Sponsor)   (:Trial)-[:TARGETS]->(:Disease)
(:Trial)-[:TESTS]->(:Drug)             (:Trial)-[:CONDUCTED_IN]->(:Country)
(:Trial)-[:LOCATED_AT]->(:Site)        (:Trial)-[:INDEXED_AS]->(:MeSHTerm)
(:Document)-[:ABOUT]->(:Trial)
"""

# Anything that writes. Whole words, so a property called `deleted_at` and a
# sponsor called "Reset Pharma" are allowed — a guard that fires on legitimate
# queries gets switched off, and then it protects nothing.
FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s*\{|"
    r"apoc\.|dbms\.|db\.index|FOREACH)\b", re.IGNORECASE)

_last: dict = {}          # the most recent successful result, for the artifact
_events = None
_task = None


def _note(text: str) -> None:
    if _events is not None and _task is not None:
        _events.put({"kind": "progress", "taskId": _task.id, "note": text})


def is_read_only(cypher: str) -> tuple[bool, str]:
    """Reject anything that could write, before it reaches the database.

    Belt and braces — the database user should also be read-only. Two
    independent guards, because either alone is a single point of failure, and
    the one in code is the one you can review.
    """
    match = FORBIDDEN.search(cypher)
    if match:
        return False, f"contains {match.group(0)!r}, which can write"
    if not re.search(r"\bMATCH\b|\bRETURN\b", cypher, re.IGNORECASE):
        return False, "no MATCH or RETURN"
    return True, ""


def _driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(config.NEO4J_URI,
                                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


@tool
def run_cypher(cypher: str) -> str:
    """Run a read-only Cypher query and return the rows.

    Rejects anything that could modify the graph. If the query fails, the error
    is returned so it can be corrected and retried.
    """
    global _last
    _note(cypher[:150])

    ok, why = is_read_only(cypher)
    if not ok:
        # Returned to the model rather than raised, so it can write a different
        # query. Raising would end the turn on a mistake that is fixable.
        return f"REFUSED: {why}. Write a read-only query."

    try:
        driver = _driver()
        with driver.session(database=config.NEO4J_DATABASE) as session:
            records = list(session.run(cypher))
        driver.close()
    except Exception as exc:
        # The database error goes back verbatim. "Unknown property phase2" is
        # exactly what the model needs to fix the query — summarising it away
        # would leave it guessing.
        return f"ERROR: {str(exc)[:300]}"

    if not records:
        _last = contracts.empty("the query ran and returned no rows")
        return "0 rows. The query was valid; the graph has nothing matching."

    columns = list(records[0].keys())
    rows = [[record[c] for c in columns] for record in records]
    _last = contracts.table(columns, rows, note=cypher)
    _note(f"{len(rows)} rows")
    return (f"{len(rows)} rows, columns {columns}. "
            f"First 3: {rows[:3]}")


class Finding(BaseModel):
    """What the agent concluded."""
    summary: str = Field(description="What the data shows, in one or two sentences")
    answerable: bool = Field(description="False if the graph cannot answer this")


def build_agent():
    return create_agent(
        model=llm.chat_model(),
        tools=[run_cypher],
        system_prompt=f"""You answer questions about a Neo4j graph of clinical trials.

Schema:
{SCHEMA}

Write a query, run it, and look at what comes back. If it errors, read the error
and fix the query. If it returns nothing, that is an answer — say so rather than
trying variations forever.

Rules:
- Read only. MATCH, WHERE, RETURN, ORDER BY, LIMIT, WITH, OPTIONAL MATCH.
- Always LIMIT. 100 unless the question implies fewer.
- Return named columns, not whole nodes.
- Set answerable to false if the schema cannot answer the question. Do not
  invent labels or properties.""",
        response_format=ToolStrategy(schema=Finding),
        name="cypher",
    )


AGENT = None


async def execute(task: Task, message: Message, queue: EventQueue) -> None:
    global _events, _task, AGENT, _last
    _events, _task, _last = queue, task, {}

    if AGENT is None:
        AGENT = build_agent()

    import asyncio
    state = await asyncio.get_event_loop().run_in_executor(
        None, lambda: AGENT.invoke({"messages": [{"role": "user",
                                                  "content": message.text}]}))

    finding = state.get("structured_response")
    if not _last:
        result = contracts.empty(
            finding.summary if finding else "no query produced a result")
    else:
        result = dict(_last)
        result["note"] = finding.summary if finding else result.get("note", "")

    server.add_artifact(task, Artifact(name="graph-result",
                                       parts=[Part(text=json.dumps(result))]))


server = A2AServer(CARD, execute)

if __name__ == "__main__":
    print(f"cypher agent -> {config.NEO4J_URI}  model {llm.describe()}")
    server.run(port=config.PORTS["cypher"])
