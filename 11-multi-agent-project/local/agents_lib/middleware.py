"""Middleware for the agent loops.

`create_agent` runs a loop: the model calls a tool, sees the result, decides
again. Middleware wraps that loop, and it is where anything structural belongs —
budgets, retries, injected context.

WHY THESE ARE MIDDLEWARE AND NOT PROMPT TEXT

A prompt that says "call at most three agents" is a suggestion. The model follows
it most turns and ignores it on the turn where the question is hard — which is
exactly the turn where an unbounded loop costs money.

A counter that refuses the fourth call is a wall. It cannot be talked around, and
the model is told why, so it has a next move that is not another call.

ORDER MATTERS

Middleware nests outside-in, so list order decides what wraps what:

    [A, B]   ->   A( B( the loop ) )

Put the budget outside the repeat check and a refused repeat still consumes a
call — a refusal is not work done. Put it inside and the repeat check never runs
once the budget is spent.

TWO THINGS THAT ARE EASY TO GET WRONG

The request exposes `request.tool_call`, a dict with "name" and "args". There is
no `request.tool_name`. Reading a non-existent attribute with a getattr default
gives every call the same empty key — which makes a distinct call look like a
repeat, and the loop stops after one tool with no error anywhere.

And a middleware must return a `ToolMessage`, not a string. Returning a bare
string puts the wrong message type into the conversation, and the model sees its
own refusal as if a person had said it.
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


def _refuse(request, text: str) -> ToolMessage:
    """A refusal the model can act on.

    `tool_call_id` has to match, or the conversation has a tool call with no
    result and the provider rejects the next request.
    """
    return ToolMessage(content=text,
                       tool_call_id=request.tool_call["id"],
                       name=request.tool_call["name"],
                       status="error")


class AgentCallBudget(AgentMiddleware):
    """Refuse tool calls past a limit, and tell the model why.

    The wording matters as much as the refusal. A bare error makes the model
    retry; "you have used your budget, answer with what you have" gives it
    somewhere else to go.
    """

    def __init__(self, max_calls: int, tool_names: set[str] | None = None):
        super().__init__()
        self.max_calls = max_calls
        # Only count calls that cross the network to another agent. A cheap
        # local tool should not consume a budget that exists to bound fan-out.
        self.tool_names = tool_names
        self._used = 0

    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if self.tool_names and name not in self.tool_names:
            return handler(request)

        if self._used >= self.max_calls:
            return _refuse(request,
                           f"Budget exhausted: {self.max_calls} agent calls used. "
                           "Answer with the results you already have. Do not call "
                           "another agent.")

        self._used += 1
        return handler(request)

    def reset(self) -> None:
        """Budgets are per turn, not per process. A long-lived agent would
        otherwise refuse everything after its first few conversations."""
        self._used = 0


class NoRepeatCalls(AgentMiddleware):
    """Refuse asking the same tool the same question twice in one turn.

    Models loop. Asked something the data cannot answer, a model will re-ask the
    same specialist in slightly different words, get the same empty result, and
    try again — spending the budget on a question already answered.

    Refusing with the previous result attached gives it what it needs to stop:
    the answer has not changed, so try something else or say so.
    """

    def __init__(self, tool_names: set[str] | None = None):
        super().__init__()
        self.tool_names = tool_names
        self._seen: dict[tuple[str, str], str] = {}

    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if self.tool_names and name not in self.tool_names:
            return handler(request)

        # The key is (tool, arguments). Both parts are needed: the same question
        # to two different agents is not a repeat, and two different questions to
        # one agent are not either.
        args = request.tool_call.get("args") or {}
        key = (name, repr(sorted(args.items()))[:300])

        if key in self._seen:
            return _refuse(request,
                           f"You already called {name} with these arguments and "
                           f"got: {self._seen[key][:300]}. Asking again returns "
                           "the same thing. Try a different tool, or answer with "
                           "what you have.")

        result = handler(request)
        self._seen[key] = str(getattr(result, "content", result))[:400]
        return result

    def reset(self) -> None:
        self._seen.clear()
