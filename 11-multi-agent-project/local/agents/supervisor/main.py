"""A2A entrypoint for the supervisor.

    python -m agents.supervisor.main     serves on :9200

The supervisor is itself an A2A agent, so whatever calls it does not know there
are three more behind it.

This module is the BRIDGE and nothing else: it turns one A2A task into one run
of the loop, forwards progress as events, and attaches the results as artifacts.
All the reasoning lives in core.py, all the policy in middleware.py.

Keeping the bridge thin is what lets the same core run on AgentCore Runtime with
a different file here and nothing else changed.
"""

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "10-a2a-protocol"))

from a2a_mini import (A2AClient, A2AServer, AgentCard, Artifact, EventQueue,
                      Message, Part, Skill, Task)
from agents_lib import config, llm

from .context import RequestContext
from .core import Decision, build_agent, compose, render
from .discovery import discover

CARD = AgentCard(
    name="supervisor",
    description="Answers questions about clinical trials using specialist agents.",
    url=config.endpoint("supervisor"),
    capabilities={"streaming": True, "pushNotifications": False},
    skills=[Skill(id="investigate", name="Answer a trial question",
                  description=("Plans across document search and the trial "
                               "graph, charts tabular results, and writes the "
                               "answer."),
                  examples=["which sponsors run more than one trial?",
                            "what are the exclusion criteria?"])],
)

# Which discovered agents are renderers rather than things the model may route
# to. Kept here rather than inferred, because "is this a renderer" is a
# deployment fact, not something to guess from a card.
RENDERERS = {"chart"}


async def execute(task: Task, message: Message, queue: EventQueue) -> None:
    """One turn: discover, plan, render, compose."""

    def note(text: str) -> None:
        queue.put({"kind": "progress", "taskId": task.id, "note": text})

    # Discovery runs per turn. An agent down when this process started should
    # not be invisible forever.
    registry = await discover(
        {name: config.endpoint(name) for name in ("vdb", "cypher", "chart")},
        RENDERERS)
    note(f"discovered {len(registry.routable)} data agents: "
         f"{', '.join(registry.names()) or 'none'}")

    if not registry.routable:
        server.add_artifact(task, Artifact(name="answer", parts=[Part(
            text="No data agents are reachable.")]))
        return

    async def invoke(agent_name: str, text: str, call_id: str) -> dict:
        """One A2A round trip, with the specialist's own events forwarded up.

        Lives here rather than in tools.py so the tool has no knowledge of
        transport — which is what lets the same tool run against AgentCore with
        only this function replaced.
        """
        client = A2AClient(registry.endpoint(agent_name))
        result = None
        async for event in client.stream(text):
            if event.get("kind") == "progress":
                note(f"[{agent_name}] {event.get('note', '')}")
            elif event.get("kind") == "artifact-update":
                result = json.loads(event["artifact"]["parts"][0]["text"])
            if event.get("final"):
                break
        if result is None:
            raise RuntimeError(f"{agent_name} returned no artifact")
        return result

    ctx = RequestContext(session_id=task.contextId, registry=registry,
                         invoke=invoke, note=note)

    agent = build_agent(registry)
    for layer in getattr(agent, "middleware", []) or []:
        if hasattr(layer, "reset"):
            layer.reset()          # budgets are per turn, not per process

    note("planning")
    state = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message.text}]},
        context=ctx)

    calls = state.get("calls") or []
    captured = state.get("captured") or {}
    clarify = state.get("clarify")

    # A clarification is the model's own words. Stream them rather than
    # rewriting them through another model call.
    if clarify:
        server.add_artifact(task, Artifact(name="clarification", parts=[Part(
            text=json.dumps(clarify))]))
        return

    decision: Decision = state.get("structured_response")
    note(f"{len(calls)} call(s), composing")

    # Rendering: code, after the loop. Not a tool the model could forget.
    charts = await render(captured, calls, ctx)

    answer = await asyncio.get_event_loop().run_in_executor(
        None, lambda: compose(message.text, decision, calls, captured))

    server.add_artifact(task, Artifact(name="answer", parts=[Part(text=answer)]))

    # The data travels beside the prose, verbatim. It never passed through the
    # model, so a reader can check the answer against it.
    for call_id, result in captured.items():
        server.add_artifact(task, Artifact(name=f"data-{call_id}",
                                           parts=[Part(text=json.dumps(result))]))
    for n, chart in enumerate(charts):
        server.add_artifact(task, Artifact(name=f"chart-{n}",
                                           parts=[Part(text=json.dumps(chart))]))
    # The audit trail: what was asked, of whom, and what came back.
    server.add_artifact(task, Artifact(name="calls",
                                       parts=[Part(text=json.dumps(calls, indent=2))]))


server = A2AServer(CARD, execute)

if __name__ == "__main__":
    print(f"supervisor -> model {llm.describe()}  budget {config.MAX_AGENT_CALLS}")
    server.run(port=config.PORTS["supervisor"])
