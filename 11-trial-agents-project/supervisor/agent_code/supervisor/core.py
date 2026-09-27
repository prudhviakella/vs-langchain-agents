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
passages, rows. The model sees a compact summary; the full response goes
to state for compose() and the final response to read directly.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import operator
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from . import agent_client
from .config import settings
from .guardrail import GuardrailBlocked, GuardrailMiddleware, check
from .tracing import set_span_attrs, span
from .schemas import (AgentCall, SupervisorDecision, SupervisorResponse,
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


def summarize_call(agent_name: str, result: dict) -> str:
    """Compact view of a specialist's response — never the full data."""
    shape = result.get("result_shape", "unknown")
    parts = [f"{agent_name} returned result_shape={shape!r}"]
    if shape == "graph":
        parts.append(f"{len(result.get('nodes', []))} nodes, "
                     f"{len(result.get('relationships', []))} relationships")
    elif shape == "table":
        parts.append(f"{len(result.get('rows', []))} rows, "
                     f"columns: {', '.join(str(c) for c in result.get('columns', [])[:8])}")
    elif shape == "passages":
        parts.append(f"{len(result.get('passages', []))} passage(s)")
    else:
        parts.append("no usable result. Do not retry the same question with the "
                     "same agent — ask the other agent, ask a narrower question, "
                     "or report the gap honestly.")
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
    with span("supervisor.call_agent", agent=agent_name) as sp:
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
        if (request.state or {}).get("call_count", 0) > 0:
            return None                      # a specialist really was called
        return decision

    def _retry(self, request):
        log.warning("hollow SupervisorDecision — retrying once")
        return request.override(messages=request.messages + [HumanMessage(
            content="You have not called any specialist yet. Call call_agent "
                    "before producing a decision — there is no way to answer "
                    "without checking trial_graph or trial_search first.")])

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
        model=model or s.chat_model(), tools=[call_agent], system_prompt=s.system_prompt,
        response_format=ToolStrategy(SupervisorDecision),
        middleware=[GuardrailMiddleware(s.guardrail_id, s.guardrail_version,
                                        decision_tool=DECISION_TOOL),
                    AgentCallBudgetMiddleware(s), RequireAgentCallMiddleware()])


class GraphState(TypedDict, total=False):
    question: str
    decision: SupervisorDecision
    agent_calls: list[dict]
    captured_results: dict[str, dict]
    usage: Any
    render_target: str
    composed_answer: str


async def _route(state: GraphState) -> dict:
    """PROBABILISTIC. Runs the inner agent and lifts what the rest of the
    graph needs into this graph's own state."""
    with span("supervisor.route") as sp:
        result = await build_react_agent().ainvoke(
            {"messages": [{"role": "user", "content": state["question"]}]})
        set_span_attrs(sp, calls=result.get("call_count", 0),
                       answerable=result["structured_response"].answerable)
    return {"decision": result["structured_response"],
            "agent_calls": result.get("agent_calls", []),
            "captured_results": result.get("captured_results", {}),
            "usage": collect_usage(result["messages"], settings().openai_model)}


def _render_decision(state: GraphState) -> dict:
    """DETERMINISTIC. A fixed rule over the shapes already in state.

    graph wins over chart when a turn produced both: a network worth
    drawing is rarely also best read as a table.
    """
    shapes = {r.get("result_shape") for r in state.get("captured_results", {}).values()}
    target = "graph" if "graph" in shapes else "chart" if "table" in shapes else "none"
    with span("supervisor.render_decision", render_target=target):
        pass
    return {"render_target": target}


# Bounds on what the composer sees. Enough to state real findings; small
# enough that it cannot retype a whole result, and the full result is shown
# to the analyst beside the answer anyway.
_ROWS, _NODES, _PASSAGES, _PASSAGE_CHARS = 10, 15, 6, 1200


def _node_name(node: dict) -> str:
    props = node.get("properties", {})
    return str(props.get("name") or props.get("briefTitle") or props.get("facility")
               or props.get("nctId") or props.get("term") or node.get("element_id", "?"))


def evidence(captured_results: dict) -> str:
    """What the composer reads: bounded, verbatim, fenced.

    An earlier version gave the composer only summarize_call() — counts like
    "3 passage(s)". It could not state a single finding; the best it could
    write was that a query had returned something. The model that ROUTES
    still sees only summaries. The model that WRITES must see the evidence
    it is writing about, or its answer has nothing in it.
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
            lines += [f"  ({'/'.join(n.get('labels', []))}) {_clean(_node_name(n))}"
                      for n in nodes[:_NODES]]
        elif shape == "passages":
            for p in result.get("passages", [])[:_PASSAGES]:
                text = str(p.get("text", "")).replace("<untrusted_data>", "") \
                                             .replace("</untrusted_data>", "")
                lines.append(f"  doc={p.get('doc_id')} page={p.get('page')} "
                             f"headings={p.get('headings')}")
                lines.append("  " + text[:_PASSAGE_CHARS]
                             + ("…" if len(text) > _PASSAGE_CHARS else ""))
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks) or "nothing was retrieved"
    return f"<untrusted_data>\n{body}\n</untrusted_data>"


async def _compose(state: GraphState) -> dict:
    """PROBABILISTIC, deliberately separate from _route: structured output
    and token-by-token streaming are mutually exclusive in one call.

    STEP 1  render the managed compose prompt with the bounded evidence
    STEP 2  write the answer
    STEP 3  OUTPUT guardrail on the answer — this is the text the analyst
            reads, and it is produced outside the agent loop, so the
            loop's middleware never sees it
    """
    from .config import render
    s = settings()
    prompt = render(s.compose_template, {
        "question": state["question"],
        "evidence": evidence(state.get("captured_results", {}))})
    with span("supervisor.compose", evidence_chars=len(prompt)):
        response = await s.chat_model().ainvoke([HumanMessage(content=prompt)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    await asyncio.to_thread(check, text, "OUTPUT", s.guardrail_id, s.guardrail_version)
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
    return graph.compile()


async def orchestrate(question: str, context_id: str | None = None) -> SupervisorResponse:
    """A guardrail intervention anywhere — the question, a model turn, the
    composed answer — ends the turn with the guardrail's own message."""
    CONVERSATION.set(context_id)
    try:
        state = await build_graph().ainvoke({"question": question})
    except GuardrailBlocked as blocked:
        return SupervisorResponse(
            decision=SupervisorDecision(answerable=False,
                                        note=f"Blocked by guardrail ({blocked.source})."),
            composed_answer=blocked.message, render_target="none")
    return SupervisorResponse(
        calls=[AgentCall(**c) for c in state.get("agent_calls", [])],
        decision=state["decision"],
        render_target=state.get("render_target", "none"),
        composed_answer=state.get("composed_answer", ""),
        usage=state.get("usage"))
