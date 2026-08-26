"""VDB Agent — a LangChain agent over the vector index from module 05.

    python agents/vdb_agent.py     serves on :9201

A loop, not a single search. The agent searches, reads what came back, and can
search again with different words — which is what a person does when the first
query misses.

That matters because vector search is sensitive to phrasing. "What patients are
excluded" and "exclusion criteria" reach different parts of the space, and the
first attempt is not always the better one.

Returns `passages`. Never `table` — text found by similarity is quotable, not
chartable, and declaring that is what stops the supervisor charting it.
"""

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "10-a2a-protocol"))
sys.path.insert(0, str(HERE.parent.parent / "05-rag"))

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from a2a_mini import A2AServer, AgentCard, Artifact, EventQueue, Message, Part, Skill, Task
from agents_lib import config, contracts, llm

CARD = AgentCard(
    name="vdb",
    description="Searches the clinical trial document index and returns passages.",
    url=config.endpoint("vdb"),
    capabilities={"streaming": True, "pushNotifications": False},
    skills=[Skill(
        id="search-documents", name="Search trial documents",
        description=("Finds passages in clinical trial protocol PDFs by meaning. "
                     "Use for what a protocol SAYS — eligibility wording, "
                     "procedures, safety text, statistical methods. Not for "
                     "counting, comparing across trials, or relationships."),
        examples=["what are the exclusion criteria?"])],
)

_found: list[dict] = []
_events = None
_task = None


def _note(text: str) -> None:
    if _events is not None and _task is not None:
        _events.put({"kind": "progress", "taskId": _task.id, "note": text})


def _index():
    """Open the index lazily, so importing this module needs no credentials."""
    from rag.index import open_index
    return open_index()


@tool
def search(query: str, content_type: str = "") -> str:
    """Search the trial documents for passages matching a query.

    `content_type` optionally narrows to one kind of content: "table_summary"
    for tables, "figure" for charts, "text" for prose. Leave it empty to search
    everything.

    Returns the passages found, with their page numbers. If nothing useful comes
    back, try different wording — vector search is sensitive to phrasing.
    """
    from rag import retrieval

    _note(f"searching: {query[:80]}")
    filters = {"content_type": {"$eq": content_type}} if content_type else None
    hits = retrieval.search(_index(), query, top_k=5, filters=filters)

    if not hits:
        return "0 passages. Try different wording, or a different content_type."

    for hit in hits:
        # Keep every passage found across all attempts. The last search is not
        # necessarily the best one, and discarding earlier results would lose
        # something the agent already had.
        if not any(f["chunk_id"] == hit["chunk_id"] for f in _found):
            _found.append({"text": hit["text"], "page": hit["page"],
                           "doc": hit["doc_id"], "score": hit["rerank"],
                           "chunk_id": hit["chunk_id"]})

    _note(f"{len(hits)} passages, {len(_found)} kept so far")
    return "\n\n".join(f"[p{h['page']} score {h['rerank']:.2f}] {h['text'][:400]}"
                       for h in hits)


class Finding(BaseModel):
    summary: str = Field(description="What the passages say, in one or two sentences")
    answerable: bool = Field(description="False if the documents do not cover this")


def build_agent():
    return create_agent(
        model=llm.chat_model(),
        tools=[search],
        system_prompt="""You find passages in clinical trial protocol documents.

Search, read what comes back, and search again with different wording if it
missed. Vector search is sensitive to phrasing — "what patients are excluded"
and "exclusion criteria" find different things.

Stop when you have what the question needs, or after two or three attempts that
find nothing useful. Set answerable to false in that case rather than answering
from your own knowledge — you are reporting what these documents say, not what
is true in general.""",
        response_format=ToolStrategy(schema=Finding),
        name="vdb",
    )


AGENT = None


async def execute(task: Task, message: Message, queue: EventQueue) -> None:
    global _events, _task, AGENT
    _events, _task = queue, task
    _found.clear()

    if AGENT is None:
        AGENT = build_agent()

    state = await asyncio.get_event_loop().run_in_executor(
        None, lambda: AGENT.invoke({"messages": [{"role": "user",
                                                  "content": message.text}]}))

    finding = state.get("structured_response")
    if not _found:
        result = contracts.empty(
            finding.summary if finding else "no passages matched")
    else:
        result = contracts.passages(
            _found, note=finding.summary if finding else "")

    server.add_artifact(task, Artifact(name="search-result",
                                       parts=[Part(text=json.dumps(result))]))


server = A2AServer(CARD, execute)

if __name__ == "__main__":
    print(f"vdb agent -> index {config.INDEX_NAME}  model {llm.describe()}")
    server.run(port=config.PORTS["vdb"])
