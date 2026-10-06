"""Supervisor core: an explicit graph that routes between the two
specialists, then composes the answer.

    orchestrate(question)
        │
        v
    build_graph()
        │
        ├─ route             PROBABILISTIC. The inner ReAct agent: an LLM
        │                    choosing which specialist to call, via one
        │                    tool (call_agent), bounded by middleware.
        │
        ├─ render_decision   DETERMINISTIC. Plain code over the shapes
        │                    already in state. No model call, and no
        │                    render TOOL for a model to forget to call.
        │
        └─ compose           PROBABILISTIC. A separate LLM call writes
                             the analyst-facing prose.
        │
        v
    SupervisorResponse

WHY route IS A NESTED GRAPH RATHER THAN REBUILT HERE

create_agent() builds and hides its own StateGraph — model node, tool
node, a conditional edge looping until the model stops calling tools.
That loop already carries ToolStrategy and the budget/require-a-call
middleware. Nesting the compiled agent as this graph's one probabilistic
node keeps those intact while making the OUTER shape — probabilistic,
deterministic, probabilistic — an inspectable graph rather than three
function calls that merely happen to run in order.

WHERE EVERYTHING COMES FROM

    model             OpenAI chat model; key and model name from Secrets Manager
    system prompt     Bedrock Prompt Management, version pinned in Parameter
                      Store; the AVAILABLE AGENTS block is rendered from the
                      registry, so it can never name an agent that is not there
    compose prompt    Bedrock Prompt Management, rendered per question
    specialists       Parameter Store registry: each specialist writes its own
                      ARN and description there when it deploys
    guardrail         ApplyGuardrail — on the question (INPUT), on every model
                      turn, and on the composed answer (OUTPUT)

FIVE FACTS VERIFIED AT RUNTIME, NOT ASSUMED

    1. Middleware state must be declared with `state_schema` (a subclass
       of AgentState). `state_schema_extra` — used by the previous
       version — is not a LangChain attribute: it is ignored and every
       Command update is silently dropped, so call_count stayed 0 and
       RequireAgentCallMiddleware could never see a call that happened.
    2. Under ainvoke, a middleware defining only wrap_tool_call raises
       NotImplementedError on the first tool call.
    3. The same is true of wrap_model_call: sync-only raises
       NotImplementedError under ainvoke. Both forms are defined here.
    4. A tool returning Command must carry the real tool_call_id. The
       previous version passed "" from inside the tool and patched it in
       middleware; the tool now takes InjectedToolCallId, which LangChain
       fills in, so the id is never wrong.
    5. ToolStrategy, not a bare response_format schema — otherwise the
       model can be structurally prevented from calling any tool and
       will answer having called no specialist at all.

WHY DATA STILL NEVER FLOWS THROUGH THE MODEL

call_agent's result is a specialist's full response — graph nodes,
passages, rows. The routing model sees a compact summary; the full response
goes to state for compose() and the final response to read directly.

The summary carries VALUES only for small results (up to 5 rows or nodes),
because a resolution step exists to hand an identifier — a docId — to the
next call. Hiding it cost a second round trip in production.

WHAT THE COMPOSER READS, AND WHAT IS SHOWN

    evidence()          passages chosen by re-ranker score within a
                        character budget, duplicates dropped, then shown in
                        reading order — not the first N in reading order
    _render_decision()  a table or graph a LATER call consumed (its values
                        reappear in that call's question) was a step, not
                        the answer, and is not rendered
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import operator
import re
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from . import agent_client
from .config import settings
from .guardrail import GuardrailBlocked, GuardrailMiddleware, check
from .tracing import set_span_attrs, span
from .memory import TOOL_NAMES as MEMORY_TOOL_NAMES
from .memory import TOOLS as MEMORY_TOOLS
from .memory import USER
from .schemas import (AgentCall, SupervisorDecision, SupervisorResponse, ToolCall,
                      collect_usage)

log = logging.getLogger("agent.supervisor.core")

DECISION_TOOL = "SupervisorDecision"

# The A2A contextId of the conversation this turn belongs to. Set once in
# orchestrate(); read by call_agent to build each specialist's session id.
# A ContextVar rather than a tool argument: the model must not be able to
# choose or forge it. Verified that the value reaches the sync tool's
# thread — LangChain runs sync tools via run_in_executor, which copies
# the calling context.
CONVERSATION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "conversation_id", default=None)


def _clean(value) -> str:
    """Neutralize one value before it enters the model's context. Notes
    are written by another agent's LLM, which may echo trial or sponsor
    text straight out of a PDF — content this project did not author."""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.replace("<untrusted_data>", "").replace("</untrusted_data>", "")
    text = " ".join(text.split())
    return text[:300] + "…" if len(text) > 300 else text


# How much of a result the ROUTING model sees. Row counts and column names
# alone hid the one thing a resolution call exists to return: the docId the
# NEXT call needs. In production the supervisor then re-asked trial_graph to
# "state the docId in your note" — a second round trip (~18 s) for a value
# it already had. Small results now show their values; large ones stay
# summarized, because the routing model needs identifiers, not the data.
_SUMMARY_ROWS, _SUMMARY_COLS, _SUMMARY_NODES, _SUMMARY_VALUE_CHARS = 5, 6, 5, 80


def _short(value) -> str:
    text = _clean(value)
    return text[:_SUMMARY_VALUE_CHARS] + "…" if len(text) > _SUMMARY_VALUE_CHARS else text


def summarize_call(agent_name: str, result: dict) -> str:
    """What the ROUTING model reads after a call: the shape, and — for small
    results — the identifying values, fenced as untrusted data."""
    shape = result.get("result_shape", "unknown")
    parts = [f"{agent_name} returned result_shape={shape!r}"]
    values = []
    if shape == "graph":
        nodes = result.get("nodes", [])
        parts.append(f"{len(nodes)} nodes, {len(result.get('relationships', []))} relationships")
        values = [_short(_node_label(n)) for n in nodes[:_SUMMARY_NODES]]
    elif shape == "table":
        rows, columns = result.get("rows", []), result.get("columns", [])
        parts.append(f"{len(rows)} rows, columns: "
                     + ", ".join(str(c) for c in columns[:8]))
        if len(rows) <= _SUMMARY_ROWS:
            values = [" | ".join(_short(v) for v in row[:_SUMMARY_COLS]) for row in rows]
    elif shape == "passages":
        parts.append(f"{len(result.get('passages', []))} passage(s)")
    else:
        parts.append("no usable result. Do not retry the same question with the "
                     "same agent — ask the other agent, ask a narrower question, "
                     "or report the gap honestly.")
    if values:
        parts.append("values: <untrusted_data>" + " ; ".join(values) + "</untrusted_data>")
    if result.get("result_note"):
        parts.append(f"note: {_clean(result['result_note'])}")
    return " — ".join(parts)


# ── the one tool ────────────────────────────────────────────────────────

@tool
def call_agent(agent_name: str, question: str, rationale: str,
               tool_call_id: Annotated[str, InjectedToolCallId],
               observation: str = "") -> Command:
    """Route a question to a specialist agent.

    agent_name: "trial_graph" for relationships between trials, sponsors,
        drugs, diseases and sites; "trial_search" for what the protocol
        documents actually say.
    question: phrased for the SPECIALIST — narrow or rephrase it if that
        gets a better answer than the analyst's wording.
    rationale: one sentence on why this agent, this question, now.
    observation: one sentence on what a PRIOR call showed. Empty on the
        first call.

    rationale and observation are tool ARGUMENTS rather than surrounding
    text because a model bound with tool_choice=any often emits no text
    at all next to a tool call. An analyst watching a turn of real queries
    with no explanation is a trust problem, not a cosmetic one.
    """
    specialists = settings().specialists
    if agent_name not in specialists:
        return Command(update={"messages": [ToolMessage(
            tool_call_id=tool_call_id,
            content=f"REJECTED: {agent_name!r} is not a specialist. "
                    f"Valid names: {', '.join(sorted(specialists))}.")]})

    # The span is current while the specialist is invoked, so the traceparent
    # agent_client sends makes the specialist's spans children of THIS one.
    # Named per target so the trace tree shows WHICH agent was called; the
    # callee's own root span, "invoke_agent <agent>", nests beneath it.
    with span(f"call_agent {agent_name}", agent=agent_name) as sp:
        try:
            result = agent_client.call_specialist(
                specialists[agent_name]["arn"], agent_name, question, CONVERSATION.get())
            succeeded = True
        except agent_client.AgentCallError as exc:
            log.warning("call to %s failed: %s", agent_name, exc)
            result = {"result_shape": "error", "result_note": str(exc)}
            succeeded = False
        set_span_attrs(sp, succeeded=succeeded, result_shape=result.get("result_shape"))

    record = AgentCall(agent_name=agent_name, question=question,
                       rationale=rationale, succeeded=succeeded,
                       result_shape=result.get("result_shape", "unknown"))
    return Command(update={
        "call_count": 1,
        "agent_calls": [record.model_dump()],
        "captured_results": {agent_name: result},
        "messages": [ToolMessage(tool_call_id=tool_call_id,
                                 content=summarize_call(agent_name, result))]})


# ── state ───────────────────────────────────────────────────────────────

def _merge(left: dict, right: dict) -> dict:
    """Merge two specialists' results rather than replacing.

    A plain `dict` field has no reducer, so LangGraph REPLACES it on every
    update: calling trial_graph after trial_search dropped the search
    result entirely, and compose() then wrote its answer from half the
    evidence with nothing to indicate anything was missing. Caught by
    asserting on captured_results after a two-specialist turn.
    """
    return {**(left or {}), **(right or {})}


class SupervisorState(AgentState):
    call_count: Annotated[int, operator.add]
    agent_calls: Annotated[list[dict], operator.add]
    captured_results: Annotated[dict, _merge]
    memory_calls: Annotated[int, operator.add]
    tool_calls: Annotated[list[dict], operator.add]
    history_turns: int


# ── middleware ──────────────────────────────────────────────────────────

class AgentCallBudgetMiddleware(AgentMiddleware):
    """Caps specialist calls per turn. Each one is a network round trip
    into another agent's own LangGraph loop."""

    state_schema = SupervisorState

    def __init__(self, cfg=None):
        super().__init__()
        self.limit = (cfg or settings()).max_agent_calls_per_turn

    def _gate(self, request):
        if not request.tool_call["name"].endswith("call_agent"):
            return request
        used = (request.state or {}).get("call_count", 0)
        if used >= self.limit:
            return ToolMessage(
                tool_call_id=request.tool_call["id"],
                content=f"REFUSED: {used} specialist calls already made this "
                        f"turn (limit {self.limit}). Produce "
                        "your decision from what you have, or set "
                        "answerable=false if nothing usable came back.")
        return request

    def wrap_tool_call(self, request, handler):
        gated = self._gate(request)
        return gated if isinstance(gated, ToolMessage) else handler(gated)

    async def awrap_tool_call(self, request, handler):
        gated = self._gate(request)
        return gated if isinstance(gated, ToolMessage) else await handler(gated)


class MemoryBudgetMiddleware(AgentMiddleware):
    """Caps memory tool calls per turn. Each is an embedding call plus a
    vector query; a model unsure what to recall must not loop on it."""

    state_schema = SupervisorState

    def __init__(self, cfg=None):
        super().__init__()
        self.limit = (cfg or settings()).max_memory_calls_per_turn

    def _gate(self, request):
        if request.tool_call["name"] not in MEMORY_TOOL_NAMES:
            return request
        used = (request.state or {}).get("memory_calls", 0)
        if used >= self.limit:
            return ToolMessage(
                tool_call_id=request.tool_call["id"],
                content=f"REFUSED: {used} memory calls already made this turn "
                        f"(limit {self.limit}). Continue with what you have.")
        return request

    def wrap_tool_call(self, request, handler):
        gated = self._gate(request)
        return gated if isinstance(gated, ToolMessage) else handler(gated)

    async def awrap_tool_call(self, request, handler):
        gated = self._gate(request)
        return gated if isinstance(gated, ToolMessage) else await handler(gated)


class RequireAgentCallMiddleware(AgentMiddleware):
    """Makes a SupervisorDecision with answerable=True and zero specialist
    calls structurally impossible.

    The failure this prevents is specific: the model identifies that a
    traversal is needed, then emits its decision without calling anything.
    The composer then writes a plausible negative finding — indistinguish-
    able from a real search that came back empty. No search ran. A silent
    wrong answer is worse than a loud failure.

    The check is on the tool NAME, not on "are there any tool calls":
    under ToolStrategy the decision itself ARRIVES as a tool call, so a
    naive check reads every final answer as "mid-loop, nothing to guard".
    """

    state_schema = SupervisorState

    def _hollow(self, request, response):
        """The decision call, if this response is a hollow one."""
        message = response.result[0]
        calls = getattr(message, "tool_calls", None) or []
        decision = next((c for c in calls if c["name"].endswith(DECISION_TOOL)), None)
        if decision is None:
            return None                      # mid-loop, or calling a specialist
        args = decision.get("args", {})
        if not args.get("answerable", True) or args.get("clarifying_question"):
            return None                      # an honest refusal is not hollow
        state = request.state or {}
        if state.get("call_count", 0) > 0:
            return None                      # a specialist really was called
        if state.get("memory_calls", 0) > 0:
            return None                      # memory was read or written — grounded
        if args.get("from_conversation") and state.get("history_turns", 0) > 0:
            return None                      # answered from turns it was actually shown
        return decision

    def _retry(self, request):
        log.warning("hollow SupervisorDecision — retrying once")
        return request.override(messages=request.messages + [HumanMessage(
            content="You have not called any specialist or memory tool yet. "
                    "Call call_agent (or a memory tool, for questions about "
                    "the user or past work) before producing a decision. Set "
                    "from_conversation=true only if the earlier turns shown "
                    "to you already contain the whole answer.")])

    def wrap_model_call(self, request, handler):
        response = handler(request)
        # One corrective retry only: a guard that can loop is a worse
        # failure than the one it prevents.
        return handler(self._retry(request)) if self._hollow(request, response) else response

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        if self._hollow(request, response):
            return await handler(self._retry(request))
        return response


# ── graph nodes ─────────────────────────────────────────────────────────

def build_react_agent(model=None, cfg=None):
    """The probabilistic core. `model` and `cfg` are injectable for testing.
    The guardrail is first, so its INPUT check runs before anything else."""
    s = cfg or settings()
    return create_agent(
        name="supervisor_route",
        model=model or s.chat_model(), tools=[call_agent, *MEMORY_TOOLS],
        system_prompt=s.system_prompt,
        response_format=ToolStrategy(SupervisorDecision),
        middleware=[GuardrailMiddleware(s.guardrail_id, s.guardrail_version,
                                        decision_tool=DECISION_TOOL),
                    AgentCallBudgetMiddleware(s), MemoryBudgetMiddleware(s),
                    RequireAgentCallMiddleware()])


class GraphState(TypedDict, total=False):
    question: str
    history: list[dict]         # [{"role": "user"|"assistant", "text": ...}], oldest first
    decision: SupervisorDecision
    agent_calls: list[dict]
    tool_calls: list[dict]
    captured_results: dict[str, dict]
    usage: Any
    render_target: str
    composed_answer: str


def history_messages(history: list[dict] | None, limit: int) -> list:
    """Earlier turns as chat messages, oldest first, bounded.

    The newest `limit` messages (10 interactions = 20 messages), each cut to
    2,000 characters: enough to resolve "that trial" or "its sponsor",
    without letting one long answer crowd out the rest. The current
    question comes AFTER these, so the input guardrail — which checks the
    last human message — still checks the question, not an old turn.
    """
    out = []
    for turn in (history or [])[-limit:]:
        text = str(turn.get("text", "")).strip()[:2000]
        if not text:
            continue
        out.append(AIMessage(content=text) if turn.get("role") == "assistant"
                   else HumanMessage(content=text))
    return out


async def _route(state: GraphState) -> dict:
    """PROBABILISTIC. Runs the inner agent and lifts what the rest of the
    graph needs into this graph's own state."""
    prior = history_messages(state.get("history"), settings().max_history_messages)
    with span("supervisor.route", history_messages=len(prior)) as sp:
        result = await build_react_agent().ainvoke(
            {"messages": [*prior, HumanMessage(content=state["question"])],
             "history_turns": len(prior)})
        set_span_attrs(sp, calls=result.get("call_count", 0),
                       memory_calls=result.get("memory_calls", 0),
                       answerable=result["structured_response"].answerable)
    return {"decision": result["structured_response"],
            "agent_calls": result.get("agent_calls", []),
            "tool_calls": result.get("tool_calls", []),
            "captured_results": result.get("captured_results", {}),
            "usage": collect_usage(result["messages"], settings().openai_model)}


def _identifiers(result: dict) -> set[str]:
    """Distinctive values in a table or graph result: the strings another
    call would copy into its question (NCT numbers, docIds, names)."""
    found = set()
    for row in result.get("rows", []) or []:
        found.update(str(v) for v in row if v is not None)
    for node in result.get("nodes", []) or []:
        props = node.get("properties", {})
        found.update(str(props[k]) for k in ("nctId", "docId", "name", "acronym") if props.get(k))
    return {v for v in found if len(v) >= 6}          # "PHASE3" and longer; not "1", "yes"


def _render_decision(state: GraphState) -> dict:
    """DETERMINISTIC. What to show beside the answer, from the results that
    ANSWERED — not from resolution steps.

    A table or graph whose values reappear in a LATER call's question was a
    stepping stone: "resolve IMbrave150" returned nctId + docId, and the
    search question then carried that docId. Rendering it produced a chart
    of one lookup row beside a protocol-text answer. A result nobody later
    consumed is an answer in its own right ("Pfizer trials?" -> a table).

    graph wins over chart when both answered: a network worth drawing is
    rarely also best read as a table.
    """
    calls = state.get("agent_calls", [])
    answered = set()
    for agent, result in state.get("captured_results", {}).items():
        shape = result.get("result_shape")
        if shape not in ("table", "graph"):
            answered.add(shape)
            continue
        produced_at = max((i for i, c in enumerate(calls) if c.get("agent_name") == agent),
                          default=-1)
        later = " ".join(c.get("question", "") for c in calls[produced_at + 1:])
        if not any(v in later for v in _identifiers(result)):
            answered.add(shape)
    target = "graph" if "graph" in answered else "chart" if "table" in answered else "none"
    with span("supervisor.render_decision", render_target=target):
        pass
    return {"render_target": target}


# Bounds on what the composer sees. Enough to state real findings; small
# enough that it cannot retype a whole result, and the full result is shown
# to the analyst beside the answer anyway.
#
# Passages are bounded by CHARACTERS, not by count. A count of 6 × 1,200
# chars, taken in reading order, once gave the composer three title pages
# and three partial criteria while the complete exclusion list — ranked
# highest by the re-ranker — sat unused further down the document.
_ROWS, _NODES = 10, 15
_EVIDENCE_CHARS = 20_000        # ≈ 5k tokens of passage text
_PASSAGE_CHARS = 4_000          # one very long passage cannot fill the budget
_NEIGHBOR_REACH = 3             # positions a neighbour may sit from its hit
_DUPLICATE_OVERLAP = 0.8        # word-set overlap at which two passages repeat


def _node_label(node: dict) -> str:
    """How a node is named in evidence. Trials carry their NCT number and
    acronym beside the title — with the title alone, the composer could not
    cite an identifier and said so ("No trial identifiers are shown")."""
    props = node.get("properties", {})
    if "Trial" in node.get("labels", []):
        parts = [props.get("nctId"), props.get("acronym"), props.get("briefTitle")]
        return " — ".join(str(p) for p in parts if p) or node.get("element_id", "?")
    return str(props.get("name") or props.get("briefTitle") or props.get("facility")
               or props.get("nctId") or props.get("term") or node.get("element_id", "?"))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _priority(passage: dict, hits: list[dict]) -> float:
    """A search hit ranks by its re-ranker score (vector score if re-ranking
    fell back). A neighbour has no score of its own: it was fetched to
    complete a hit, so it ranks just below the nearest hit it extends."""
    own = passage.get("rerank_score")
    if own is None and passage.get("origin") == "search":
        own = passage.get("score")
    if own is not None:
        return float(own)
    near = [h for h in hits if h.get("doc_id") == passage.get("doc_id")
            and h.get("position") is not None and passage.get("position") is not None
            and abs(h["position"] - passage["position"]) <= _NEIGHBOR_REACH]
    if not near:
        return 0.0
    return max(_priority(h, []) - 0.01 * abs(h["position"] - passage["position"])
               for h in near)


def select_passages(passages: list[dict]) -> tuple[list[dict], int]:
    """The passages the composer reads, and how many were left out.

    STEP 1  rank: re-ranker score; a neighbour just below the hit it extends
    STEP 2  drop near-duplicates, keeping the higher-ranked copy — protocols
            repeat whole sections (the synopsis restates the criteria)
    STEP 3  take by rank until the character budget is spent
    STEP 4  return in reading order: document, then position
    """
    hits = [p for p in passages if p.get("origin") == "search"]
    ranked = sorted(passages, key=lambda p: _priority(p, hits), reverse=True)

    kept, kept_words, used = [], [], 0
    for p in ranked:
        words = _words(str(p.get("text", "")))
        if any(len(words & w) >= _DUPLICATE_OVERLAP * min(len(words), len(w))
               for w in kept_words if words and w):
            continue
        size = min(len(str(p.get("text", ""))), _PASSAGE_CHARS)
        if used + size > _EVIDENCE_CHARS:
            continue
        kept.append(p); kept_words.append(words); used += size

    kept.sort(key=lambda p: (str(p.get("doc_id")), p.get("position") or 0))
    return kept, len(passages) - len(kept)


def evidence(captured_results: dict) -> str:
    """What the composer reads: bounded, verbatim, fenced.

    An earlier version gave the composer only summarize_call() — counts like
    "3 passage(s)". It could not state a single finding; the best it could
    write was that a query had returned something. The model that ROUTES
    sees summaries. The model that WRITES must see the evidence it is
    writing about, or its answer has nothing in it.
    """
    blocks = []
    for name, result in captured_results.items():
        shape = result.get("result_shape", "unknown")
        lines = [f"[{name}] result_shape={shape}"]
        if result.get("result_note"):
            lines.append(f"note: {_clean(result['result_note'])}")
        if shape == "table":
            rows = result.get("rows", [])
            lines.append("columns: " + ", ".join(map(str, result.get("columns", []))))
            lines += ["  " + " | ".join(_clean(v) for v in row) for row in rows[:_ROWS]]
            if len(rows) > _ROWS:
                lines.append(f"  ... {len(rows) - _ROWS} more rows shown to the analyst "
                             "in the table, not here")
        elif shape == "graph":
            nodes = result.get("nodes", [])
            lines.append(f"{len(nodes)} nodes, {len(result.get('relationships', []))} relationships")
            lines += [f"  ({'/'.join(n.get('labels', []))}) {_clean(_node_label(n))}"
                      for n in nodes[:_NODES]]
        elif shape == "memory":
            lines.append(f"{result.get('kind')} memories recalled for: "
                         f"{_clean(result.get('query', ''))}")
            lines += [f"  ({i.get('created_at', '')[:10]}) {_clean(i.get('text', ''))}"
                      for i in result.get("items", [])]
        elif shape == "passages":
            chosen, left_out = select_passages(result.get("passages", []))
            for p in chosen:
                text = str(p.get("text", "")).replace("<untrusted_data>", "") \
                                             .replace("</untrusted_data>", "")
                lines.append(f"  doc={p.get('doc_id')} page={p.get('page')} "
                             f"headings={p.get('headings')}")
                lines.append("  " + text[:_PASSAGE_CHARS]
                             + ("…" if len(text) > _PASSAGE_CHARS else ""))
            if left_out:
                lines.append(f"  ({left_out} lower-ranked or repeated passage(s) not shown "
                             "here; the analyst sees all of them)")
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks) or "nothing was retrieved"
    return f"<untrusted_data>\n{body}\n</untrusted_data>"


# Fixed reply for an out-of-scope question. No trial names or counts: the
# platform's trials come from the registry graph and change as it grows.
OUT_OF_SCOPE = ("I can only help with questions about the clinical trials in this "
                "platform — their sponsors, sites, phases, conditions and outcomes from the "
                "registry, and what their protocols say about eligibility, design, endpoints, "
                "dosing and safety. Try asking which trials run in a country, or what a "
                "trial's protocol — named by its NCT number or acronym — says about its "
                "exclusion criteria.")


async def _compose(state: GraphState) -> dict:
    """PROBABILISTIC, deliberately separate from _route: structured output
    and token-by-token streaming are mutually exclusive in one call.

    STEP 1  render the managed compose prompt with the bounded evidence
    STEP 2  write the answer
    STEP 3  OUTPUT guardrail on the answer — this is the text the analyst
            reads, and it is produced outside the agent loop, so the
            loop's middleware never sees it
    """
    # An out-of-scope question gets one fixed reply: no composer call, no tokens,
    # and no chance of the model answering it from general knowledge.
    if state.get("decision") is not None and state["decision"].out_of_scope:
        with span("supervisor.compose", out_of_scope=True):
            return {"composed_answer": OUT_OF_SCOPE}
    from .config import render
    s = settings()
    decision = state.get("decision")
    prompt = render(s.compose_template, {
        "question": (decision.resolved_question if decision and decision.resolved_question
                     else state["question"]),
        "evidence": evidence(state.get("captured_results", {}))})
    with span("supervisor.compose", evidence_chars=len(prompt)):
        response = await s.chat_model().ainvoke([HumanMessage(content=prompt)])
    # The Responses API returns content as a LIST of blocks (text, reasoning,
    # annotations). str() of that list is what an analyst once received:
    # "[{'type': 'text', 'text': ..., 'phase': 'final_answer'}]". .text joins
    # only the text blocks.
    text = response.text() if callable(response.text) else response.text
    # check() returns the answer to show: unchanged, or with personal data
    # masked — protocols print medical monitors' emails and phone numbers.
    text = await asyncio.to_thread(check, text, "OUTPUT", s.guardrail_id, s.guardrail_version)
    return {"composed_answer": text}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("route", _route)
    graph.add_node("render_decision", _render_decision)
    graph.add_node("compose", _compose)
    graph.add_edge(START, "route")
    graph.add_edge("route", "render_decision")
    graph.add_edge("render_decision", "compose")
    graph.add_edge("compose", END)
    return graph.compile(name="supervisor")


async def orchestrate(question: str, context_id: str | None = None,
                      history: list[dict] | None = None,
                      user_id: str | None = None) -> SupervisorResponse:
    """One turn.

    history   earlier messages of this conversation, oldest first, from the
              backend — [{"role": "user"|"assistant", "text": ...}]
    user_id   the signed-in username; scopes every memory read and write

    A guardrail intervention anywhere — the question, a model turn, the
    composed answer — ends the turn with the guardrail's own message.
    """
    from .tracing import current_trace_id
    CONVERSATION.set(context_id)
    USER.set(user_id or None)
    history = history or []
    try:
        state = await build_graph().ainvoke({"question": question, "history": history})
    except GuardrailBlocked as blocked:
        return SupervisorResponse(
            decision=SupervisorDecision(answerable=False,
                                        note=f"Blocked by guardrail ({blocked.source})."),
            composed_answer=blocked.message, render_target="none",
            trace_id=current_trace_id())
    return SupervisorResponse(
        calls=[AgentCall(**c) for c in state.get("agent_calls", [])],
        tool_calls=[ToolCall(**c) for c in state.get("tool_calls", [])],
        decision=state["decision"],
        render_target=state.get("render_target", "none"),
        composed_answer=state.get("composed_answer", ""),
        usage=state.get("usage"),
        results=state.get("captured_results", {}),
        history_turns=min(len(history), settings().max_history_messages),
        trace_id=current_trace_id())
