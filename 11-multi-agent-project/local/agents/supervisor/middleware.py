"""The policies the model cannot route around.

    AvailableAgents      inject who exists, from discovery, every turn
    AgentCallBudget      at most N specialist calls per turn
    NoRepeatCall         a second call to an agent that answered must justify itself
    RequireAgentCall     no "answered" decision with zero calls made

Each of these has an equivalent sentence in the prompt. The sentences are
necessary and never sufficient — the model follows them most turns and ignores
them on the turn where the question is hard, which is exactly the turn where it
matters.

`awrap_tool_call` wraps the tool NODE, so a refusal happens whether the model
asked nicely or not. Prompt language is a wish; this is a fact.

SYNC OR ASYNC, NOT BOTH BY ACCIDENT. LangChain dispatches to `wrap_model_call`
when the agent is invoked with `invoke`, and to `awrap_model_call` when invoked
with `ainvoke`. Define only the sync pair and an async run raises
NotImplementedError from inside the graph — which reads as a LangGraph problem
rather than a missing method. The A2A bridge is async and the tool is async, so
these are the async hooks.

ORDER IS LOAD-BEARING. Middleware nests outside-in:

    [A, B]   ->   A( B( the loop ) )

AvailableAgents goes first so its injected block is present inside everything
else — including any corrective retry, which would otherwise ask the model to
pick an agent while showing it none.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage

from .state import SupervisorState

CALL_AGENT = "call_agent"


def _refuse(request, text: str) -> ToolMessage:
    """A refusal the model can act on.

    `tool_call_id` must match, or the conversation contains a tool call with no
    result and the provider rejects the next request.

    A ToolMessage, not a string: a bare string is inserted as the wrong message
    type, and the model reads its own refusal as if a person had said it.
    """
    return ToolMessage(content=text, tool_call_id=request.tool_call["id"],
                       name=request.tool_call["name"], status="error")


class AvailableAgents(AgentMiddleware[SupervisorState]):
    """Inject who exists, discovered this turn.

    The prompt deliberately contains no agent list. A list written there is a
    second copy of what the cards already say, and it is how a prompt ends up
    describing an agent that was disabled last week — or omitting one registered
    yesterday.
    """

    state_schema = SupervisorState

    def __init__(self, registry):
        super().__init__()
        self.registry = registry

    async def awrap_model_call(self, request, handler):
        block = ("## AVAILABLE AGENTS\n\n" + self.registry.as_prompt_block()
                 + "\n\nUse these exact names. There are no others.")
        request = request.override(
            messages=[SystemMessage(content=block), *request.messages])
        return await handler(request)


class AgentCallBudget(AgentMiddleware[SupervisorState]):
    """At most `max_calls` specialist calls per turn.

    The wording of the refusal matters as much as the refusal. A bare error
    makes the model retry; "you have used your budget, answer with what you
    have" gives it a move that is not another call.
    """

    state_schema = SupervisorState

    def __init__(self, max_calls: int):
        super().__init__()
        self.max_calls = max_calls
        self._used = 0

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] != CALL_AGENT:
            return await handler(request)  # asking the user is not a data call
        if self._used >= self.max_calls:
            return _refuse(request,
                           f"Call budget exhausted ({self.max_calls}). Answer "
                           "with the results you already have. Do not call "
                           "another agent.")
        self._used += 1
        return await handler(request)

    def reset(self) -> None:
        """Per turn, not per process. A long-lived agent would otherwise refuse
        everything after its first few conversations."""
        self._used = 0


class NoRepeatCall(AgentMiddleware[SupervisorState]):
    """A second call to an agent that already answered must justify itself.

    The budget caps the total but has no notion of redundancy — four calls to
    one agent asking four phrasings of the same question sit comfortably inside
    it. The prompt says "prefer ONE call"; prompt language is necessary and
    never sufficient.

    Only calls that PRODUCED SOMETHING count as already answered. Counting a
    failed call arms this guard wrongly: the model gets told an agent has
    already answered when in fact the call failed and captured nothing.

    A repeat is allowed when the model states what specifically was not
    answered. That is a next step; a rephrasing is a retry.
    """

    state_schema = SupervisorState

    def __init__(self):
        super().__init__()
        self._answered: dict[str, str] = {}

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] != CALL_AGENT:
            return await handler(request)

        args = request.tool_call.get("args") or {}
        agent = str(args.get("agent_name") or "")

        if agent in self._answered and not args.get("observation"):
            return _refuse(request,
                           f"{agent} already answered this turn: "
                           f"{self._answered[agent][:200]}. If you need "
                           "something it did not cover, say what in the "
                           "`observation` argument. Otherwise ask a different "
                           "agent or answer with what you have.")

        result = await handler(request)
        content = str(getattr(result, "content", result))
        # Only record it as answered if something came back. An empty or failed
        # result is not an answer.
        if "empty" not in content[:60] and "FAILED" not in content[:60]:
            self._answered[agent] = content
        return result

    def reset(self) -> None:
        self._answered.clear()


class RequireAgentCall(AgentMiddleware[SupervisorState]):
    """Refuse a final decision that answers without ever asking an agent.

    THE FAILURE THIS EXISTS FOR:

        tool_calls: []          never attempted a call
        calls: []               no agent ran
        answerable: true        ...and yet reported as answered

    The model works out correctly which agent the question needs, then emits its
    decision on the same turn instead of taking step 2 of its own instructions.
    The answer that reaches the user is prose indistinguishable from a real
    search that genuinely came back empty. No search ran at all.

    That is the worst available shape for a bug: not an error, not a visible
    blank, but a confident negative finding someone would reasonably act on.

    Schema validation cannot catch it — `answerable: true` is a correctly typed
    bool, and a validator on the decision sees only its own fields, never
    `state["calls"]` where the evidence lives. The check has to happen where
    both are visible.

    Deliberately narrow. It fires only when ALL of these hold: zero calls made,
    a final decision produced, answerable is true, and no clarification asked.
    A legitimate zero-call turn is untouched — an honest "no agent can answer
    this" sets answerable false, and a clarification sets clarify.

    ONE retry, then it lets the turn through. A guard that can loop is a worse
    failure than the one it prevents.
    """

    state_schema = SupervisorState

    def __init__(self):
        super().__init__()
        self._retried = False

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        if self._retried or not self._is_hollow(request, response):
            return response

        self._retried = True
        corrected = request.override(messages=[
            *request.messages,
            SystemMessage(content=(
                "You produced a final answer without calling any agent. You "
                "have no data. Either call an agent now, or set answerable to "
                "false and say that no available agent can serve this."))])
        return await handler(corrected)

    def _is_hollow(self, request, response) -> bool:
        """A final decision made with no evidence behind it."""
        state = getattr(request, "state", {}) or {}
        if state.get("calls"):
            return False                        # something ran
        if state.get("clarify"):
            return False                        # asking is a legitimate outcome

        message = getattr(response, "result", response)
        messages = message if isinstance(message, list) else [message]
        for item in messages:
            if getattr(item, "tool_calls", None):
                # Still choosing. Only the final decision is checked, and the
                # decision schema arrives as a tool call named after it.
                names = {c["name"] for c in item.tool_calls}
                return names == {"Decision"}
        return False

    def reset(self) -> None:
        self._retried = False
