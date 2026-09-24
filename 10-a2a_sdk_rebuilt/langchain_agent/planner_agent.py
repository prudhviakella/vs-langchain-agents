"""
============================================================
A2A PLANNER AGENT  (LangChain agent inside)
============================================================

The A2A server stays the same. Only the brain changed:
a LangChain agent (create_agent) talking to a real LLM,
streamed with Event Streaming (astream_events, version="v3").


FLOW
----

  Client                    Executor                     LangChain agent
    |                          |                               |
    | "plan a trip to Kyoto"   |                               |
    |------------------------->|  new messages --------------->|
    |                          |                               | model: days missing
    |                          |                               | tool: ask_user
    |                          |<------ interrupt (question) --|
    |<-- input-required -------|                               |
    |    "How many days?"      |                               |
    |                          |                               |
    | "3 days"                 |                               |
    |------------------------->|  Command(resume="3 days") --->|
    |<-- artifact chunks ------|<------ stream.messages -------|
    |                          |                               | tool: publish_itinerary
    |                          |<------ interrupt (approval) --|  (HITL middleware)
    |<-- input-required -------|                               |
    |    "Approve this plan?"  |                               |
    |                          |                               |
    | "approve"                |                               |
    |------------------------->|  Command(resume=approve) ---->|
    |                          |                               | publish runs
    |<-- artifact: itinerary --|<------ stream.tool_calls -----|
    |<-- completed ------------|                               |


EVENT STREAMING (v3)
--------------------

  One loop over the raw protocol events:

    async for event in await agent.astream_events(..., version="v3"):

  event["method"]   what we do with it
  ---------------   -----------------------------------------------
  messages          text-delta         -> "response" artifact chunk
                    message-start/finish -> open / close that artifact
  tools             tool-started       -> working status "Running X..."
                    tool-finished      -> publish_itinerary output
                                          = "itinerary" artifact
  values            params["interrupts"] non-empty
                                       -> agent paused for a human

  After the loop:
    interrupts found  -> input-required
    none              -> completed


TWO KINDS OF HUMAN-IN-THE-LOOP
------------------------------

  1. ask_user tool          agent needs a missing fact
                            (tool calls interrupt())

  2. publish_itinerary      agent wants to publish the plan
                            (HumanInTheLoopMiddleware pauses it)

  Both become A2A "input-required".
  The user's next message resumes the paused agent.


HOW THE AGENT REMEMBERS
-----------------------

  checkpointer = InMemorySaver()
  thread_id    = A2A task.id

  The checkpointer keeps the full agent state per task,
  including the paused interrupt. We no longer rebuild
  context from task.history.


WHAT THIS DOES NOT DO
---------------------

  - Does NOT persist state across restarts. InMemorySaver
    lives in process memory. Swap in a Postgres/Redis
    checkpointer for that.
  - Does NOT share memory between tasks. A new task in the
    same context starts a fresh agent thread.
  - Does NOT call real travel APIs. The itinerary comes
    from the LLM's own knowledge.
  - Does NOT support "edit" decisions. The reviewer either
    approves, or replies with feedback (= reject + reason).
  - version="v3" is marked experimental by LangGraph.
    Its API may change between releases.


CONFIG
------

  PLANNER_MODEL   any init_chat_model string
                  default: anthropic:claude-sonnet-5
                  e.g.     bedrock_converse:<model-id>
                           openai:gpt-4.1
"""

import os
import uuid
import warnings
from typing import Any


# ============================================================
# LANGCHAIN / LANGGRAPH
# ============================================================

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_core._api import LangChainBetaWarning
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

# v3 event streaming prints a beta warning on every run. We know.
warnings.filterwarnings("ignore", category=LangChainBetaWarning)


# ============================================================
# A2A
# ============================================================

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater


# ============================================================
# APPLICATION HELPERS
# ============================================================

from common import make_card, run_agent


# ============================================================
# SERVER CONFIGURATION
# ============================================================

PORT = 9103
URL = f"http://127.0.0.1:{PORT}/"

MODEL = os.getenv("PLANNER_MODEL", "anthropic:claude-sonnet-5")


# ============================================================
# AGENT CARD
# ============================================================

CARD = make_card(
    name="planner",
    description=(
        "Plans a trip with an LLM. Asks for missing details "
        "and asks for approval before publishing the plan."
    ),
    url=URL,
    skill_id="plan-trip",
    skill_name="Plan a trip",
    skill_description="Produces a day-by-day itinerary for a destination.",
    examples=[
        "plan a trip to Kyoto",
        "plan 3 days in Lisbon",
    ],
    streaming=True,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a trip planner.

1. You need two facts: the destination and the number of days.
   If either is missing, call ask_user with ONE short question.
   Never guess them.

2. When you have both, write a day-by-day itinerary
   (morning / afternoon / evening, at most 14 days)
   and pass it to publish_itinerary.
   Do not write the itinerary in your chat reply.

3. If publish_itinerary is rejected, the rejection contains the
   traveller's feedback. Revise the plan and call publish_itinerary again.

4. After publish_itinerary succeeds, reply with one short sentence.
"""


# ============================================================
# TOOLS
# ============================================================

@tool
def ask_user(question: str) -> str:
    """Ask the traveller a question when a required detail is missing."""

    # interrupt() pauses the whole agent run right here.
    # The value we pass becomes the pending interrupt.
    # On resume, interrupt() returns the user's answer.
    return interrupt({"kind": "question", "question": question})


@tool
def publish_itinerary(destination: str, days: int, itinerary: str) -> str:
    """Publish the final day-by-day itinerary to the traveller."""

    # This only runs AFTER a human approved it
    # (HumanInTheLoopMiddleware stops it before that).
    # The executor turns this tool's output into the A2A artifact.
    return itinerary


# ============================================================
# BUILD THE AGENT
# ============================================================

def build_agent(model: Any = MODEL):
    """
    model -> create_agent -> compiled LangGraph graph

    model can be a string ("anthropic:claude-sonnet-5")
    or a chat model instance.
    """

    return create_agent(
        model=model,
        tools=[ask_user, publish_itinerary],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            # Pause before publish_itinerary runs.
            # ask_user is not listed: it pauses itself via interrupt().
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "publish_itinerary": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                },
            ),
        ],
        # Required for interrupts: the paused state must be saved somewhere.
        checkpointer=InMemorySaver(),
    )


# ============================================================
# INTERRUPT <-> A2A HELPERS
# ============================================================
#
# A pending interrupt value has one of two shapes:
#
#   from ask_user:
#       {"kind": "question", "question": "How many days?"}
#
#   from HumanInTheLoopMiddleware:
#       {"action_requests": [{"name": "publish_itinerary",
#                             "args": {...}}],
#        "review_configs":  [...]}
#
# ============================================================

APPROVE_WORDS = {"approve", "approved", "yes", "y", "ok", "okay", "lgtm", "looks good"}


def _prompt_for(value: dict) -> str:
    """Turn a pending interrupt into the text shown to the A2A client."""

    if value.get("kind") == "question":
        return value["question"]

    itinerary = value["action_requests"][0]["args"].get("itinerary", "")
    return (
        f"Here is the draft plan:\n\n{itinerary}\n\n"
        "Reply 'approve' to publish it, or tell me what to change."
    )


def _resume_value(value: dict, user_text: str) -> Any:
    """Turn the user's reply into the value the paused agent expects."""

    # ask_user: interrupt() simply returns the answer text.
    if value.get("kind") == "question":
        return user_text

    # HITL middleware: needs one decision per paused tool call.
    if user_text.strip().lower().rstrip(".!") in APPROVE_WORDS:
        decision = {"type": "approve"}
    else:
        # Anything else is feedback. The agent sees it as the rejection reason.
        decision = {"type": "reject", "message": user_text}

    return {"decisions": [decision for _ in value["action_requests"]]}


# ============================================================
# PLANNER EXECUTOR
# ============================================================

class PlannerAgentExecutor(AgentExecutor):

    def __init__(self, model: Any = MODEL) -> None:
        self.agent = build_agent(model)

    # ========================================================
    # EXECUTE
    # ========================================================
    #
    # Called once per incoming A2A message.
    # Same task can call this many times:
    #
    #   1st  "plan a trip to Kyoto"   -> new agent run
    #   2nd  "3 days"                 -> resume ask_user
    #   3rd  "approve"                -> resume publish approval
    #
    # ========================================================

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        # ----------------------------------------------------
        # STEP 1  Read the incoming message.
        # ----------------------------------------------------

        message = context.message
        if message is None:
            raise ValueError("A message is required")

        user_text = get_message_text(message) or ""

        # ----------------------------------------------------
        # STEP 2  Get or create the A2A task.
        # ----------------------------------------------------

        task = context.current_task
        if task is None:
            task = new_task_from_user_message(message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # One agent thread per A2A task.
        config = {"configurable": {"thread_id": task.id}}

        try:
            # ------------------------------------------------
            # STEP 3  New run, or resume a paused one?
            #
            #   pending interrupt  ->  Command(resume=...)
            #   nothing pending    ->  new user message
            # ------------------------------------------------

            state = await self.agent.aget_state(config)

            if state.interrupts:
                pending = state.interrupts[0]
                agent_input = Command(
                    resume={pending.id: _resume_value(pending.value, user_text)}
                )
            else:
                agent_input = {"messages": [{"role": "user", "content": user_text}]}

            await updater.start_work(new_text_message("Planning your trip..."))

            # ------------------------------------------------
            # STEP 4  One loop over the v3 event stream.
            #
            #   event = {"method": ..., "params": {"namespace": [...],
            #                                      "data": ...}}
            #
            #   namespace [] = the agent itself. Anything else would be
            #   a nested subgraph; this agent has none, so we skip them.
            # ------------------------------------------------

            interrupts = []          # set when a values event reports a pause
            tool_names = {}          # tool_call_id -> tool name (finished events lack the name)
            response_id = None       # artifact id of the current LLM message
            sent_any = False         # has this LLM message sent a chunk yet?

            stream = await self.agent.astream_events(
                agent_input,
                config,
                version="v3",
            )

            async for event in stream:

                method = event["method"]
                params = event["params"]

                if params["namespace"]:
                    continue

                # STEP 4a  LLM output. data = (event_dict, metadata)
                #
                #   message-start        -> new "response" artifact id
                #   content-block-delta  -> text chunk (tool-call args are
                #                           separate blocks, not text-delta)
                #   message-finish       -> last_chunk=True
                if method == "messages":
                    data, _meta = params["data"]
                    kind = data["event"]

                    if kind == "message-start":
                        response_id = str(uuid.uuid4())
                        sent_any = False

                    elif kind == "content-block-delta" and data["delta"].get("type") == "text-delta":
                        text = data["delta"]["text"]
                        if text:
                            await updater.add_artifact(
                                parts=[new_text_part(text=text, media_type="text/plain")],
                                artifact_id=response_id,
                                name="response",
                                append=sent_any,
                                last_chunk=False,
                            )
                            sent_any = True

                    elif kind == "message-finish" and sent_any:
                        # Empty parts: this chunk only marks the end of
                        # the artifact, it adds no more text.
                        await updater.add_artifact(
                            parts=[],
                            artifact_id=response_id,
                            name="response",
                            append=True,
                            last_chunk=True,
                        )

                # STEP 4b  Tool execution.
                #
                #   ask_user           no status; its question arrives as
                #                      the input-required message instead
                #   publish_itinerary  only runs after approval, so
                #                      tool-finished = the approved plan.
                #                      A rejected publish never runs.
                elif method == "tools":
                    data = params["data"]
                    kind = data["event"]

                    if kind == "tool-started":
                        tool_names[data["tool_call_id"]] = data["tool_name"]
                        if data["tool_name"] != "ask_user":
                            await updater.start_work(
                                new_text_message(f"Running {data['tool_name']}...")
                            )

                    elif kind == "tool-finished":
                        if tool_names.get(data["tool_call_id"]) == "publish_itinerary":
                            await updater.add_artifact(
                                parts=[new_text_part(
                                    text=str(data["output"].content),
                                    media_type="text/markdown",
                                )],
                                name="itinerary",
                            )

                # STEP 4c  State snapshot. Only the pause info matters here.
                elif method == "values" and params.get("interrupts"):
                    interrupts = list(params["interrupts"])

            # ------------------------------------------------
            # STEP 5  Did the agent pause for a human?
            #
            #   yes -> input-required (task stays alive)
            #   no  -> completed
            # ------------------------------------------------

            if interrupts:
                await updater.requires_input(
                    new_text_message(_prompt_for(interrupts[0].value))
                )
                return

            await updater.complete(new_text_message("Trip plan completed."))

        except Exception as exc:
            # ------------------------------------------------
            # STEP 6  Any LLM / tool error fails the A2A task
            #         instead of leaving it stuck in "working".
            # ------------------------------------------------
            await updater.failed(new_text_message(f"Planner failed: {exc}"))

    # ========================================================
    # CANCEL
    # ========================================================

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        await TaskUpdater(
            event_queue,
            context.task_id or "",
            context.context_id,
        ).cancel(new_text_message("Planning was cancelled."))


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_agent(
        CARD,
        PlannerAgentExecutor(),
        PORT,
    )