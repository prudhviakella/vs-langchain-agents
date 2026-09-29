"""Semantic and episodic memory — written and read only when the agent decides.

    supervisor model
        │  decides, per turn, whether memory is worth touching
        ├─ remember_fact(fact, topic)          SEMANTIC  a durable fact or preference
        ├─ recall_facts(query)                            about THIS user
        ├─ record_episode(summary, outcome)    EPISODIC  what happened in a past
        └─ recall_episodes(query)                         interaction worth finding again
                │
                v
        MemoryStore
          DynamoDB  trial-agents-memory    PK user_id, SK memory_id   source of truth:
                                                                      list, delete, audit
          Pinecone  trial-agents-memory    namespace "semantic" | "episodic",
                                           filter user_id             semantic search
          OpenAI    text-embedding-3-small (same model as the corpus index)

WHY THE AGENT DECIDES, NOT THE PIPELINE

Writing every turn blindly fills memory with noise ("thanks", retries,
half-finished questions) and makes recall return it. Reading on every turn
spends an embedding and a query on questions that need neither. The model
sees the conversation and knows when "I prefer tables" is a preference
worth keeping, or when "the trial we looked at last week" needs a lookup.
Every call it makes is recorded as a tool call the analyst can see.

WHY TWO STORES

Pinecone answers "which memories are about this?" DynamoDB answers "what
does the agent remember about me?" and "delete that" — questions a vector
index answers badly. Each record is written to both, under one memory_id.

WHOSE MEMORY

USER is set per request from the A2A message metadata, where the backend
puts the signed-in username. Every read and write is scoped to it: a query
filter in Pinecone, the partition key in DynamoDB. A request without a user
has no memory — the tools say so rather than guess an identity.

A FACT IS UPDATED, NOT DUPLICATED

remember_fact first looks for the user's nearest existing fact. At
similarity >= 0.90 it is the same fact restated ("prefers tables" /
"likes answers as tables"): that record is overwritten, not added again.

WHAT THIS DOES NOT DO

    - It does not store conversation history. The backend passes the last
      10 interactions with each request; memory is for what outlives them.
    - It never raises into the agent loop. Any failure becomes a tool
      result saying memory is unavailable, and the turn continues.
    - It does not trust recalled text as instructions: results are fenced
      as untrusted data, like every other retrieved content.
"""
from __future__ import annotations

import contextvars
import logging
import time
import uuid
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

log = logging.getLogger("agent.supervisor.memory")

USER: contextvars.ContextVar[str | None] = contextvars.ContextVar("memory_user", default=None)

EMBED_MODEL = "text-embedding-3-small"
SEMANTIC, EPISODIC = "semantic", "episodic"
SAME_FACT = 0.90             # similarity at which a new fact replaces an old one
RECALL_K = 5
MAX_TEXT = 2000              # characters kept per memory


class MemoryStore:
    """One record, two places: DynamoDB (truth) and Pinecone (search)."""

    def __init__(self, table, index, embed):
        self.table, self.index, self.embed = table, index, embed

    def search(self, kind: str, user_id: str, query: str, k: int = RECALL_K) -> list[dict]:
        result = self.index.query(vector=self.embed(query), top_k=k, namespace=kind,
                                  filter={"user_id": {"$eq": user_id}}, include_metadata=True)
        return [{"memory_id": m.id, "score": round(float(m.score), 3),
                 "text": (m.metadata or {}).get("text", ""),
                 "topic": (m.metadata or {}).get("topic", ""),
                 "created_at": (m.metadata or {}).get("created_at", "")}
                for m in result.matches]

    def write(self, kind: str, user_id: str, text: str, *, memory_id: str | None = None,
              **extra) -> str:
        memory_id = memory_id or f"{kind}-{uuid.uuid4().hex[:12]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        text = text.strip()[:MAX_TEXT]
        meta = {"user_id": user_id, "kind": kind, "text": text, "created_at": now,
                **{k: str(v) for k, v in extra.items() if v}}
        self.table.put_item(Item={"user_id": user_id, "memory_id": memory_id, **meta})
        self.index.upsert(vectors=[{"id": memory_id, "values": self.embed(text),
                                    "metadata": meta}], namespace=kind)
        return memory_id


_store: MemoryStore | None = None


def store() -> MemoryStore:
    """Built once per container from settings."""
    global _store
    if _store is None:
        import boto3
        from openai import OpenAI
        from pinecone import Pinecone

        from .config import settings
        s = settings()
        if not (s.memory_table and s.memory_index and s.pinecone_api_key):
            raise RuntimeError("memory is not configured for this deployment")
        client = OpenAI(api_key=s.openai_api_key)
        _store = MemoryStore(
            table=boto3.resource("dynamodb", region_name=s.region).Table(s.memory_table),
            index=Pinecone(api_key=s.pinecone_api_key).Index(s.memory_index),
            embed=lambda text: client.embeddings.create(
                model=EMBED_MODEL, input=text).data[0].embedding)
    return _store


def use_store(value: MemoryStore | None) -> None:
    """Install a store directly — tests, and nothing else."""
    global _store
    _store = value


# ── tool plumbing ───────────────────────────────────────────────────────

def _result(tool_call_id: str, name: str, args: dict, message: str, *,
            succeeded: bool, found: list[dict] | None = None, kind: str = "") -> Command:
    """Every memory tool ends here: one visible tool-call record, one count
    against the per-turn memory budget, and — for recalls — the items, so
    the composer and the analyst see exactly what was recalled."""
    update = {
        "memory_calls": 1,
        "tool_calls": [{"tool": name, "args": args, "succeeded": succeeded,
                        "result": message[:300], "items": len(found or [])}],
        "messages": [ToolMessage(tool_call_id=tool_call_id, content=message)],
    }
    if found:
        update["captured_results"] = {f"memory:{kind}": {
            "result_shape": "memory", "kind": kind, "query": args.get("query", ""),
            "items": found}}
    return Command(update=update)


def _fenced(items: list[dict]) -> str:
    lines = [f"- ({i['score']:.2f}, {i['created_at'][:10]}) "
             + i["text"].replace("<untrusted_data>", "").replace("</untrusted_data>", "")
             for i in items]
    return "<untrusted_data>\n" + "\n".join(lines) + "\n</untrusted_data>"


def _unavailable(tool_call_id: str, name: str, args: dict, why: str) -> Command:
    return _result(tool_call_id, name, args,
                   f"memory unavailable: {why}. Continue without it.", succeeded=False)


def _recall(kind: str, name: str, query: str, tool_call_id: str) -> Command:
    args = {"query": query}
    user = USER.get()
    if not user:
        return _unavailable(tool_call_id, name, args, "no user identity on this request")
    try:
        found = store().search(kind, user, query)
    except Exception as exc:
        log.warning("%s failed: %s", name, exc)
        return _unavailable(tool_call_id, name, args, type(exc).__name__)
    if not found:
        return _result(tool_call_id, name, args, f"no {kind} memories match this query.",
                       succeeded=True)
    return _result(tool_call_id, name, args,
                   f"{len(found)} {kind} memories, most similar first:\n" + _fenced(found),
                   succeeded=True, found=found, kind=kind)


# ── the four tools ──────────────────────────────────────────────────────

@tool
def remember_fact(fact: str, tool_call_id: Annotated[str, InjectedToolCallId],
                  topic: str = "") -> Command:
    """Store a durable fact or preference about THIS user, for future
    conversations. Use only when the user states something worth keeping
    beyond today — "I focus on oncology trials", "show results as tables" —
    never for the answer to the current question, and never for facts about
    trials (those live in the registry and protocols).

    fact: one self-contained sentence, e.g. "Prefers results as tables."
    topic: a short label, e.g. "preference", "research focus".
    """
    args = {"fact": fact, "topic": topic}
    user = USER.get()
    if not user:
        return _unavailable(tool_call_id, "remember_fact", args, "no user identity")
    try:
        s = store()
        nearest = s.search(SEMANTIC, user, fact, k=1)
        same = nearest[0]["memory_id"] if nearest and nearest[0]["score"] >= SAME_FACT else None
        memory_id = s.write(SEMANTIC, user, fact, memory_id=same, topic=topic)
    except Exception as exc:
        log.warning("remember_fact failed: %s", exc)
        return _unavailable(tool_call_id, "remember_fact", args, type(exc).__name__)
    verb = "updated the existing fact" if same else "stored a new fact"
    return _result(tool_call_id, "remember_fact", args, f"{verb} ({memory_id}).",
                   succeeded=True)


@tool
def recall_facts(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Look up stored facts and preferences about THIS user — before
    answering in a way their preferences would change, or when they refer
    to something about themselves ("my usual format", "my focus area").

    query: what to look for, e.g. "preferred answer format".
    """
    return _recall(SEMANTIC, "recall_facts", query, tool_call_id)


@tool
def record_episode(summary: str, tool_call_id: Annotated[str, InjectedToolCallId],
                   outcome: str = "") -> Command:
    """Record what happened in this interaction so it can be found in a
    LATER conversation. Use at the end of a turn that established something
    the user is likely to come back to — a finding, a resolved identifier,
    a comparison — not for small talk, failed attempts or clarifications.

    summary: what was asked and found, with identifiers, e.g. "Compared
        IMbrave150 (NCT03434379) exclusion criteria; hepatic encephalopathy
        and varices were key exclusions."
    outcome: "answered", "partially answered" or "not answerable".
    """
    args = {"summary": summary, "outcome": outcome}
    user = USER.get()
    if not user:
        return _unavailable(tool_call_id, "record_episode", args, "no user identity")
    try:
        from .core import CONVERSATION          # set per request by orchestrate()
        memory_id = store().write(EPISODIC, user, summary, outcome=outcome,
                                  conversation_id=CONVERSATION.get() or "")
    except Exception as exc:
        log.warning("record_episode failed: %s", exc)
        return _unavailable(tool_call_id, "record_episode", args, type(exc).__name__)
    return _result(tool_call_id, "record_episode", args, f"episode recorded ({memory_id}).",
                   succeeded=True)


@tool
def recall_episodes(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Find past interactions with THIS user — when they refer to earlier
    work that is not in the conversation you can see ("the trial we looked
    at last week", "what did we find about STEP 1 before?").

    query: what to look for, e.g. "IMbrave150 exclusion criteria".
    """
    return _recall(EPISODIC, "recall_episodes", query, tool_call_id)


TOOLS = [remember_fact, recall_facts, record_episode, recall_episodes]
TOOL_NAMES = {t.name for t in TOOLS}
