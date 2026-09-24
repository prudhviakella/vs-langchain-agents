"""Supervisor's core agent: routes between trial_graph and trial_search.

This file is built around an EXPLICIT LangGraph StateGraph — three real
nodes, visibly chained, deliberately alternating probabilistic and
deterministic:

    orchestrate(question)
        |
        v
    build_graph()
        |
        |-- route              PROBABILISTIC — an LLM, via the inner
        |                       ReAct agent build_react_agent() builds
        |                       (its own model/tool loop, ToolStrategy,
        |                       AgentCallBudgetMiddleware,
        |                       RequireAgentCallMiddleware — see their
        |                       own docstrings). Decides which
        |                       specialist(s) to call and produces
        |                       SupervisorDecision.
        |
        |-- render_decision     DETERMINISTIC — plain code, no model
        |                       call. Inspects each captured result's
        |                       result_shape and sets render_target.
        |                       No render TOOL exists for the model to
        |                       remember to call — the reference
        |                       Supervisor's own principle, "rendering
        |                       is code, not a tool," made visible here
        |                       as an actual graph node rather than an
        |                       implicit rule buried in a bigger
        |                       function.
        |
        |-- compose             PROBABILISTIC, a SEPARATE model call —
        |                       see its own docstring for why this
        |                       can't be the same call that produces
        |                       SupervisorDecision.
        |
        v
    SupervisorResponse

WHY route IS ITSELF A NESTED GRAPH, NOT REBUILT HERE

build_react_agent() calls create_agent(), which builds and hides its
own inner StateGraph (model node, tools node, a conditional edge that
loops back until the model stops calling tools) behind one .ainvoke()
call. That inner loop already carries real, tested guarantees —
ToolStrategy avoiding the hollow-decision bug, the budget and
require-a-call middleware. Rebuilding it by hand here would mean
re-deriving all of that from scratch for no benefit; nesting the
compiled inner graph as this outer graph's one probabilistic "route"
step keeps it intact while still making the OUTER shape — probabilistic
then deterministic then probabilistic — an explicit, inspectable graph
rather than three function calls in a row that happen to run in order.

WHY DATA STILL NEVER FLOWS THROUGH THE MODEL HERE, EVEN THOUGH THIS
AGENT NEVER TOUCHES A DATABASE ITSELF

The same principle from trial_graph and trial_search applies one level
up: call_agent's own tool result is a specialist's full structured
response (graph nodes, passages, whatever it found) — potentially large.
The model sees a compact summary; the full response is stashed in state
for compose() and the final SupervisorResponse to use directly.
"""
from __future__ import annotations

import logging
import operator
from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from . import agent_client
from .config import CFG
from .prompt import SYSTEM_PROMPT
from .schemas import AgentCall, SupervisorDecision, SupervisorResponse, collect_usage

log = logging.getLogger("agent.supervisor.core")

_SPECIALISTS = {
    "trial_graph": CFG.trial_graph_arn,
    "trial_search": CFG.trial_search_arn,
}


# ── The one tool ─────────────────────────────────────────────────────

@tool
def call_agent(agent_name: str, question: str, rationale: str,
               observation: str = "") -> Command:
    """Route a question to a specialist agent.

    agent_name: "trial_graph" (relationships between trials, sponsors,
        drugs, diseases, sites) or "trial_search" (what the protocol
        documents actually say).
    question: the question to send — phrase it for the SPECIALIST, not
        necessarily verbatim from the analyst; narrow or rephrase if
        that will get a better answer.
    rationale: one sentence on why this specialist, this question, now.
    observation: one sentence on what you learned from a PRIOR call this
        turn, if any — empty on the first call.

    Emitted in this order (rationale/observation as tool ARGUMENTS, not
    separate text) so the reasoning panel reads as a coherent narrative
    even when the model doesn't also emit a text block alongside the
    tool call — a real gap the reference Supervisor found: a model
    bound with tool_choice=any often emits no text at all alongside a
    tool call, and an analyst watching a turn with real queries and zero
    explanation is a genuine trust problem, not a cosmetic one.
    """
    if agent_name not in _SPECIALISTS:
        return ToolMessage(
            content=f"REJECTED: {agent_name!r} is not a known specialist. "
                    f"Valid names: {', '.join(_SPECIALISTS)}.",
            tool_call_id="")  # overwritten by the framework on a real call

    try:
        result = agent_client.call_specialist(
            _SPECIALISTS[agent_name], question, context_id=agent_name)
        succeeded = True
    except agent_client.AgentCallError as exc:
        log.warning("call to %s failed: %s", agent_name, exc)
        result = {"result_shape": "error", "result_note": str(exc)}
        succeeded = False

    record = AgentCall(
        agent_name=agent_name, question=question, rationale=rationale,
        result_shape=result.get("result_shape", "unknown"), succeeded=succeeded,
    )

    return Command(update={
        "agent_calls": [record.model_dump()],
        "captured_results": {agent_name: result},
        "messages": [ToolMessage(content=_summarize_call(agent_name, result),
                                 tool_call_id="")],
    })


def _clean(value) -> str:
    """Neutralize one value before it enters the model's own context.

    Adapted from the reference Supervisor's own _clean() — same
    reasoning applies here even though this corpus has no per-user
    access control to protect: result_note is written by another
    agent's own LLM, which in turn may echo trial/sponsor/site names
    straight out of PDF text. That text was authored by whoever
    submitted the protocol to the registry, not by this project — a
    field like a sponsor's own free-text description is exactly the
    kind of place adversarial or malformed content could show up, even
    if less likely here than in a fraud-investigation context where the
    records are written by parties actively being investigated.
    """
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.replace("<untrusted_data>", "").replace("</untrusted_data>", "")
    text = " ".join(text.split())
    cap = 300
    return text[:cap] + "…" if len(text) > cap else text


def _summarize_call(agent_name: str, result: dict) -> str:
    """Compact view of a specialist's response — enough for the model
    to decide what to do next, never the full data.
    """
    shape = result.get("result_shape", "unknown")
    note = result.get("result_note", "")
    parts = [f"{agent_name} returned result_shape={shape!r}"]

    if shape == "graph":
        parts.append(f"{len(result.get('nodes', []))} nodes, "
                     f"{len(result.get('relationships', []))} relationships")
    elif shape == "table":
        parts.append(f"{len(result.get('rows', []))} rows")
    elif shape == "passages":
        parts.append(f"{len(result.get('passages', []))} passage(s)")
    elif shape in ("empty", "unanswerable", "not_executed", "error"):
        # Adapted from the reference Supervisor's own unanswerable-case
        # guidance: naming the actual next moves (a different agent, a
        # narrower question, an honest gap) heads off the failure mode
        # of the model just re-asking the same specialist the same
        # question and burning the call budget on a repeat.
        parts.append("no usable result. Do not retry the same question "
                     "with the same agent — ask a different agent, ask a "
                     "narrower question, or report the gap honestly.")

    if note:
        parts.append(f"note: {_clean(note)}")
    return " — ".join(parts)


# ── Middleware ────────────────────────────────────────────────────────

class AgentCallBudgetMiddleware(AgentMiddleware):
    """Caps call_agent calls per turn. Each call is a real network round
    trip to another agent's own LangGraph loop — an unbounded fan-out on
    an ambiguous question is a cost and latency problem, same reasoning
    as the reference Supervisor's own AgentCallBudgetMiddleware.
    """
    state_schema_extra = {"call_count": (int, 0, operator.add)}

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "call_agent":
            return handler(request)

        if request.state.get("call_count", 0) >= CFG.max_agent_calls_per_turn:
            return ToolMessage(
                content=f"CALL BUDGET EXHAUSTED ({CFG.max_agent_calls_per_turn} "
                        "specialist calls this turn). Stop calling specialists "
                        "and produce your decision from what has been found so "
                        "far, or set answerable=false if nothing usable came "
                        "back.",
                tool_call_id=request.tool_call["id"])

        result = handler(request)
        if isinstance(result, Command):
            result.update["call_count"] = 1
            # The tool_call_id placeholders above get filled in properly
            # here, where the real id from the framework is available.
            for msg in result.update.get("messages", []):
                if isinstance(msg, ToolMessage) and not msg.tool_call_id:
                    msg.tool_call_id = request.tool_call["id"]
            return result
        return result


class RequireAgentCallMiddleware(AgentMiddleware):
    """Makes it structurally impossible to produce a SupervisorDecision
    with answerable=True and zero call_agent calls made this turn.

    Ported directly from the reference Supervisor's own middleware,
    which exists because of a real, specific production incident there:
    a question needing a graph traversal, where the model correctly
    identified the need but then emitted its decision on the same turn
    instead of actually calling an agent. The composer then wrote a
    plausible-sounding negative finding — indistinguishable from a real
    search that came back empty. No search ever ran. The reference
    system's own framing is worth repeating exactly: "not an error, not
    a visible blank, but a confident negative finding an analyst would
    reasonably act on. A silent wrong answer is worse than a loud
    failure." Applied here preemptively, before this Supervisor has had
    the chance to reproduce that incident on its own.

    Deliberately checks the tool NAME on any pending tool call, not
    merely whether tool_calls is non-empty — under ToolStrategy, the
    structured decision itself arrives AS a tool call (named after the
    schema). A naive "are there any tool calls" check would treat the
    decision-in-progress as "a pending call, nothing to guard yet,"
    defeating this middleware's own purpose. This was the reference
    guard's own first bug, caught by a regression test built from the
    original trace — checked here from the start rather than
    rediscovered independently.
    """

    def wrap_model_call(self, request, handler):
        response = handler(request)
        message = response.result[0]

        tool_calls = getattr(message, "tool_calls", None) or []
        decision_call = next(
            (c for c in tool_calls if c["name"] == "SupervisorDecision"), None)
        if decision_call is None:
            return response  # mid-loop, or genuinely calling call_agent — fine

        args = decision_call.get("args", {})
        if not args.get("answerable", True):
            return response  # an honest "cannot answer" is not hollow
        if args.get("clarifying_question"):
            return response  # a real clarification is not hollow either

        if request.state.get("call_count", 0) > 0:
            return response  # at least one specialist was actually called

        # Hollow: answerable=True, no clarification, zero specialist
        # calls made. One corrective retry, then give up — a guard that
        # can loop is a worse failure than the one it prevents.
        log.warning("hollow SupervisorDecision detected — retrying once")
        retry_request = request.override(messages=request.messages + [
            HumanMessage(content="You have not called any specialist yet. "
                                 "You must call call_agent before producing "
                                 "a decision — there is no way to answer "
                                 "this question without checking trial_graph "
                                 "or trial_search first.")])
        return handler(retry_request)


# ── Assembly ──────────────────────────────────────────────────────────

def build_react_agent():
    """The probabilistic core: an LLM choosing which specialist to call,
    how many times, and when it has enough to decide. This is the ONE
    node in the outer graph below where the model actually reasons —
    everything else in this file is either delivering that reasoning
    somewhere (route) or plain code (render_decision).
    """
    model = ChatBedrockConverse(
        model=CFG.model_id,
        guardrail_config={"guardrailIdentifier": CFG.guardrail_id,
                          "guardrailVersion": CFG.guardrail_version},
    )
    return create_agent(
        model=model, tools=[call_agent], system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(SupervisorDecision),
        middleware=[AgentCallBudgetMiddleware(), RequireAgentCallMiddleware()],
    )


class GraphState(TypedDict, total=False):
    """State threaded through the outer graph below — route writes the
    first four fields, render_decision writes render_target, compose
    writes composed_answer. Every field here is either the original
    question or something a node actually produced; nothing is guessed
    at construction time.
    """
    question: str
    decision: SupervisorDecision
    agent_calls: list[dict]
    captured_results: dict[str, dict]
    usage: Any
    render_target: str
    composed_answer: str


async def _route(state: GraphState) -> dict:
    """PROBABILISTIC node. Runs the inner ReAct agent (its own
    tool-calling loop, middleware, and structured-output guarantees —
    see build_react_agent()) and lifts the parts of its result the rest
    of this graph needs into this graph's own state.
    """
    agent = build_react_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": state["question"]}]})
    return {
        "decision": result["structured_response"],
        "agent_calls": result.get("agent_calls", []),
        "captured_results": result.get("captured_results", {}),
        "usage": collect_usage(result["messages"], CFG.model_id),
    }


def _render_decision(state: GraphState) -> dict:
    """DETERMINISTIC node. No model call — a plain rule over the shapes
    already sitting in captured_results.

    This is the actual point of building this as an explicit graph
    rather than three sequential function calls: the choice of what to
    render is not something an LLM decides and could forget to act on
    — it's a fixed function of what came back. Same principle as the
    reference Supervisor's own "rendering is code, not a tool" — there
    is no render tool here for a model to remember to call, because the
    decision needs no judgment at all, only the data's own shape.

    graph takes priority over chart if a turn somehow produced both —
    a network worth drawing is rarely also best read as a table.
    """
    shapes = {result.get("result_shape") for result in
             state.get("captured_results", {}).values()}
    if "graph" in shapes:
        target = "graph"
    elif "table" in shapes:
        target = "chart"
    else:
        target = "none"
    return {"render_target": target}


async def _compose(state: GraphState) -> dict:
    """PROBABILISTIC node, deliberately separate from _route.

    Cannot be folded into the same model call that produces
    SupervisorDecision: structured output and token-by-token streaming
    are mutually exclusive modes for a single LLM call, same constraint
    the reference Supervisor documents for its own compose() step. This
    composer gets a bounded view of what each specialist actually
    returned — not the full captured_results, and never anything beyond
    what's already in state — so it can state real findings without
    retyping raw data it was never given in the first place.
    """
    model = ChatBedrockConverse(
        model=CFG.model_id,
        guardrail_config={"guardrailIdentifier": CFG.guardrail_id,
                          "guardrailVersion": CFG.guardrail_version},
    )
    summaries = [_summarize_call(name, result)
                for name, result in state.get("captured_results", {}).items()]
    prompt = (f"Question: {state['question']}\n\n"
             f"What was found:\n" + "\n".join(summaries) + "\n\n"
             "Write a brief, direct answer for the analyst, grounded only "
             "in what was found above. If nothing usable was found, say so "
             "plainly rather than hedging.")
    response = await model.ainvoke([HumanMessage(content=prompt)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    return {"composed_answer": text}


def build_graph():
    """The explicit graph this whole file is organized around:

        route (probabilistic: the model decides, via call_agent)
            |
            v
        render_decision (deterministic: plain code on result_shape)
            |
            v
        compose (probabilistic: a separate model call writes the prose)

    Contrast with build_react_agent() above, whose own internal loop
    (model -> tool -> model -> ... -> structured decision) is ALSO a
    graph, just one LangChain's create_agent() builds and hides behind
    a single .ainvoke() call. Nesting it as this outer graph's "route"
    node keeps its already-tested middleware stack intact rather than
    re-implementing tool-calling and structured output by hand — the
    new, visible structure is the outer three-node shape, not a
    replacement for the inner one.
    """
    graph = StateGraph(GraphState)
    graph.add_node("route", _route)
    graph.add_node("render_decision", _render_decision)
    graph.add_node("compose", _compose)
    graph.add_edge(START, "route")
    graph.add_edge("route", "render_decision")
    graph.add_edge("render_decision", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


async def orchestrate(question: str) -> SupervisorResponse:
    graph = build_graph()
    state = await graph.ainvoke({"question": question})

    return SupervisorResponse(
        calls=[AgentCall(**c) for c in state.get("agent_calls", [])],
        decision=state["decision"],
        render_target=state.get("render_target", "none"),
        composed_answer=state.get("composed_answer", ""),
        usage=state.get("usage"),
    )
