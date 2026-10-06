"""Semantic and episodic memory — the agent decides; the platform makes it visible.

    one turn
        │
        ├─ overview(user)            DETERMINISTIC, before the model runs.
        │     DynamoDB Query on the user's partition:
        │       semantic-*  -> the stored facts (newest MAX_FACTS_IN_CONTEXT)
        │       episodic-*  -> a count only
        │     rendered into the routing model's system message and the
        │     composer's prompt (core.py). The model now KNOWS what exists.
        │
        ├─ supervisor model, during the loop — tools it MAY call
        │     remember_fact(fact, topic)   SEMANTIC  a durable fact or preference
        │     recall_facts(query)                    the facts beyond the ones shown
        │     recall_episodes(query)       EPISODIC  past work worth finding again
        │
        └─ SupervisorDecision.episode      the model's own one-line record of what
              │                            this turn established, or "" for nothing
              v
           write_episode()  — a graph node after compose, not a tool call
                │
                v
        MemoryStore
          DynamoDB  trial-agents-memory    PK user_id, SK memory_id   source of truth:
                                                                      list, delete, audit
          Pinecone  trial-agents-memory    namespace "semantic" | "episodic",
                                           filter user_id             semantic search
          OpenAI    text-embedding-3-small (same model as the corpus index)

WHY MEMORY WAS NEVER USED BEFORE

    1. The model could not see that memory existed. Nothing told it whether
       this analyst had stored facts, so it recalled only when a question
       said "my" or "last week" — almost never.
    2. Recording an episode was a separate tool call the model had to make
       AFTER the specialists answered and BEFORE its decision. It went
       straight to the decision instead, every time.
    3. The composer never saw a stored preference, so "prefers tables"
       could not change an answer even when recalled.

WHY THE AGENT STILL DECIDES

The overview is a read of what EXISTS, not a decision about what matters:
showing a model its tools' state is context, not a pipeline choosing for
it. Whether to store a fact, search past episodes, or record this turn is
still the model's call. The episode is a FIELD of the decision the model
must emit anyway, so it is decided at the moment the outcome is known and
cannot be skipped by forgetting a tool call.

WHY TWO STORES

Pinecone answers "which memories are about this?" DynamoDB answers "what
does the agent remember about me?" and "delete that" — questions a vector
index answers badly. Each record is written to both, under one memory_id.
memory_id starts with its kind ("semantic-", "episodic-"), so one key-range
Query on the user's partition lists one kind without a scan.

WHOSE MEMORY

USER is set per request from the A2A message metadata, where the backend
puts the signed-in username. Every read and write is scoped to it: a query
filter in Pinecone, the partition key in DynamoDB. A request without a user
has no memory — the overview says so and the tools refuse rather than
guess an identity.

A FACT IS UPDATED, NOT DUPLICATED

remember_fact first looks for the user's nearest existing fact. At
similarity >= 0.90 it is the same fact restated ("prefers tables" /
"likes answers as tables"): that record is overwritten, not added again.

WHAT THIS DOES NOT DO

    - It does not store conversation history. The backend passes the last
      10 interactions with each request; memory is for what outlives them.
    - It never raises into the agent loop or the graph. Any failure becomes
      a tool result, or an overview saying memory is unavailable, and the
      turn continues.
    - It does not trust stored text as instructions: facts and recalls are
      fenced as untrusted data, like every other retrieved content.
    - It does not decide what is worth remembering. The model does.
"""
from __future__ import annotations

import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field
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
# Facts are short, user-owned and apply to every answer, so the newest are
# shown in full. Past this many, recall_facts searches the rest.
MAX_FACTS_IN_CONTEXT = 10


def _fence(text: str) -> str:
    return str(text).replace("<untrusted_data>", "").replace("</untrusted_data>", "")


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

    def _query(self, user_id: str, kind: str, **extra) -> list[dict] | int:
        """Every page of one kind in the user's partition. With Select=COUNT,
        the total count instead of items."""
        args = {"KeyConditionExpression": "user_id = :u AND begins_with(memory_id, :k)",
                "ExpressionAttributeValues": {":u": user_id, ":k": f"{kind}-"}, **extra}
        items, count, start = [], 0, None
        while True:
            page = self.table.query(**args, **({"ExclusiveStartKey": start} if start else {}))
            items += page.get("Items", [])
            count += int(page.get("Count", 0))
            start = page.get("LastEvaluatedKey")
            if not start:
                return count if extra.get("Select") == "COUNT" else items

    def overview(self, user_id: str) -> "Overview":
        """What memory holds for this user: newest facts in full, episodes counted.

        STEP 1  every semantic record (facts are few and short)
        STEP 2  newest first, keep MAX_FACTS_IN_CONTEXT
        STEP 3  episodes: a count only — their content is searched on demand
        """
        facts = sorted(self._query(user_id, SEMANTIC),
                       key=lambda i: i.get("created_at", ""), reverse=True)
        episodes = self._query(user_id, EPISODIC, Select="COUNT")
        return Overview(available=True,
                        facts=[{"text": f.get("text", ""), "topic": f.get("topic", ""),
                                "created_at": f.get("created_at", "")}
                               for f in facts[:MAX_FACTS_IN_CONTEXT]],
                        fact_count=len(facts), episode_count=int(episodes))


@dataclass
class Overview:
    """What one user's memory holds right now. `available=False` carries why."""
    available: bool
    reason: str = ""
    facts: list[dict] = field(default_factory=list)
    fact_count: int = 0
    episode_count: int = 0

    def for_router(self) -> str:
        """The block appended to the routing model's system message."""
        if not self.available:
            return (f"## THIS ANALYST'S MEMORY\nMemory is unavailable for this request "
                    f"({self.reason}). Do not call memory tools; set `episode` to \"\".")
        lines = ["## THIS ANALYST'S MEMORY",
                 f"Stored facts: {self.fact_count}. Recorded episodes: {self.episode_count}."]
        if self.facts:
            shown = (f" (the newest {len(self.facts)}; recall_facts searches the rest)"
                     if self.fact_count > len(self.facts) else "")
            lines.append(f"Facts{shown}:")
            lines.append("<untrusted_data>")
            lines += [f"- [{_fence(f['topic']) or 'fact'}] {_fence(f['text'])}" for f in self.facts]
            lines.append("</untrusted_data>")
        return "\n".join(lines)

    def for_composer(self) -> str:
        """The analyst's stored facts, for the composer. Empty text when none."""
        if not (self.available and self.facts):
            return "No stored preferences."
        return "<untrusted_data>\n" + "\n".join(
            f"- {_fence(f['text'])}" for f in self.facts) + "\n</untrusted_data>"


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


# ── read before the turn, write after it ─────────────────────────────────

def overview() -> Overview:
    """The current user's overview; never raises."""
    user = USER.get()
    if not user:
        return Overview(available=False, reason="no user identity on this request")
    try:
        return store().overview(user)
    except Exception as exc:
        log.warning("memory overview failed: %s", exc)
        return Overview(available=False, reason=f"{type(exc).__name__}: {str(exc)[:120]}")


def write_episode(summary: str, outcome: str, conversation_id: str) -> dict:
    """Record the episode the model put in its decision. Returns the ToolCall
    record the analyst sees, succeeded or not; never raises."""
    args = {"summary": summary, "outcome": outcome}
    user = USER.get()
    if not user:
        return {"tool": "record_episode", "args": args, "succeeded": False,
                "result": "memory unavailable: no user identity", "items": 0}
    try:
        memory_id = store().write(EPISODIC, user, summary, outcome=outcome,
                                  conversation_id=conversation_id)
    except Exception as exc:
        log.warning("record_episode failed: %s", exc)
        return {"tool": "record_episode", "args": args, "succeeded": False,
                "result": f"memory unavailable: {type(exc).__name__}", "items": 0}
    return {"tool": "record_episode", "args": args, "succeeded": True,
            "result": f"episode recorded ({memory_id}).", "items": 0}


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
    lines = [f"- ({i['score']:.2f}, {i['created_at'][:10]}) " + _fence(i["text"])
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


# ── the three tools ─────────────────────────────────────────────────────

@tool
def remember_fact(fact: str, tool_call_id: Annotated[str, InjectedToolCallId],
                  topic: str = "") -> Command:
    """Store a durable fact or preference about THIS analyst, for future
    conversations. Use when the analyst states something about themselves
    worth keeping beyond today — their research focus, how they want answers
    shown — never for the answer to the current question, and never for
    facts about trials (those live in the registry and protocols).

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
    """Search this analyst's stored facts beyond the newest ones already shown
    in THIS ANALYST'S MEMORY. Only useful when that section says more facts
    exist than it shows.

    query: what to look for, e.g. "preferred answer format".
    """
    return _recall(SEMANTIC, "recall_facts", query, tool_call_id)


@tool
def recall_episodes(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Find this analyst's past work that is not in the conversation you can
    see — when they refer to it ("the trial we looked at last week"), or when
    the question is about a trial or topic their recorded episodes may cover.

    query: the trial, topic or finding to look for, with identifiers if known.
    """
    return _recall(EPISODIC, "recall_episodes", query, tool_call_id)


TOOLS = [remember_fact, recall_facts, recall_episodes]
TOOL_NAMES = {t.name for t in TOOLS}
