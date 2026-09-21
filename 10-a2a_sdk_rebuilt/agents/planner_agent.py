"""
============================================================
A2A PLANNER AGENT
============================================================

Demonstrates:

    1. A2A Agent Card
    2. Task creation
    3. Task history
    4. input-required
    5. Task continuation
    6. Artifact creation
    7. Task completion
    8. Task cancellation


EXAMPLE
-------

First request:

    "plan a trip to Kyoto"


Agent:

    "How many days is the trip?"


Second request:

    "3 days"


Agent:

    Day 1: explore Kyoto
    Day 2: explore Kyoto
    Day 3: explore Kyoto


============================================================
WORKFLOW
============================================================

FIRST REQUEST
-------------

Client
  |
  | message/send
  | "plan a trip to Kyoto"
  v
Planner Agent
  |
  | execute()
  v
Create Task
  |
  v
Check task history
  |
  v
Duration missing
  |
  v
input-required
  |
  v
"How many days?"


CONTINUATION
------------

Client
  |
  | "3 days"
  |
  | SAME TASK / CONTEXT
  v
Planner Agent
  |
  | execute() again
  v
Read task.history
  |
  +-- "plan a trip to Kyoto"
  |
  +-- "3 days"
  |
  v
Extract:
    destination = Kyoto
    days = 3
  |
  v
working
  |
  v
Generate itinerary
  |
  v
Artifact
  |
  v
completed
"""

import asyncio
import re


# ============================================================
# A2A HELPERS
# ============================================================

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)


# ============================================================
# A2A SERVER COMPONENTS
# ============================================================

from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
)

from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater


# ============================================================
# A2A TYPES
# ============================================================
#
# IMPORTANT:
#
# In a2a-sdk 1.x these are protobuf types.
#
# Role is therefore a protobuf enum.
#
# task.history item.role returns an integer enum value.
#
# Therefore:
#
#     item.role.name       ❌
#
# Use:
#
#     item.role == Role.ROLE_USER     ✅
#
# ============================================================

from a2a.types import Role


# ============================================================
# APPLICATION HELPERS
# ============================================================

from common import (
    make_card,
    run_agent,
)


# ============================================================
# SERVER CONFIGURATION
# ============================================================

PORT = 9103

URL = f"http://127.0.0.1:{PORT}/"


# ============================================================
# AGENT CARD
# ============================================================

CARD = make_card(

    name="planner",

    description=(
        "Plans a trip and asks for duration "
        "if it was not supplied."
    ),

    url=URL,

    skill_id="plan-trip",

    skill_name="Plan a trip",

    skill_description=(
        "Produces a day-by-day outline for a destination."
    ),

    examples=[
        "plan a trip to Kyoto",
        "plan 3 days in Lisbon",
    ],

    streaming=True,
)


# ============================================================
# HELPER: EXTRACT NUMBER OF DAYS
# ============================================================

def _days(
    text: str,
) -> int | None:

    match = re.search(
        r"(\d+)\s*day",
        text,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# HELPER: EXTRACT DESTINATION
# ============================================================

def _destination(
    user_messages: list[str],
) -> str:

    # Combine all user messages.
    #
    # Example:
    #
    #     plan a trip to Kyoto
    #     3 days
    #
    # becomes:
    #
    #     plan a trip to Kyoto 3 days

    asked = " ".join(
        user_messages
    )

    # Remove duration.
    destination = re.sub(
        r"\d+\s*days?",
        "",
        asked,
        flags=re.IGNORECASE,
    )

    # Remove common words.
    destination = re.sub(
        r"\b(plan|a|trip|to|for|the)\b",
        "",
        destination,
        flags=re.IGNORECASE,
    )

    return (
        destination.strip()
        or "somewhere"
    )


# ============================================================
# PLANNER EXECUTOR
# ============================================================

class PlannerAgentExecutor(AgentExecutor):


    # ========================================================
    # EXECUTE
    # ========================================================
    #
    # Called by the A2A SDK when:
    #
    #     message/send
    #
    # is received.
    #
    # IMPORTANT:
    #
    # This method can be called multiple times for the same
    # task.
    #
    # First call:
    #
    #     "plan a trip to Kyoto"
    #
    # Second call:
    #
    #     "3 days"
    #
    # ========================================================

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        # ----------------------------------------------------
        # STEP 1
        # Get incoming message.
        # ----------------------------------------------------

        message = context.message

        if message is None:
            raise ValueError(
                "A message is required"
            )


        # ----------------------------------------------------
        # STEP 2
        # Get current task.
        #
        # First request:
        #
        #     current_task may be None
        #
        # Continuation:
        #
        #     current_task contains existing Task
        # ----------------------------------------------------

        task = context.current_task


        # ====================================================
        # CREATE NEW TASK
        # ====================================================

        if task is None:

            task = new_task_from_user_message(
                message
            )

            # Publish the new task.
            await event_queue.enqueue_event(
                task
            )


        # ====================================================
        # STEP 3
        # CREATE TASK UPDATER
        # ====================================================

        updater = TaskUpdater(
            event_queue,
            task.id,
            task.context_id,
        )


        # ====================================================
        # STEP 4
        # READ TASK HISTORY
        # ====================================================
        #
        # This is where we remember the previous conversation.
        #
        # Example:
        #
        #     task.history
        #
        #     User:
        #         plan a trip to Kyoto
        #
        #     Agent:
        #         How many days?
        #
        #     User:
        #         3 days
        #
        # ====================================================

        user_messages = [
            get_message_text(item) or ""
            for item in task.history

            # IMPORTANT:
            #
            # In protobuf, role is represented as an integer
            # enum value.
            #
            # Therefore DON'T do:
            #
            #     item.role.name.lower()
            #
            # Instead:
            #
            #     item.role == Role.ROLE_USER
            #
            if item.role == Role.ROLE_USER
        ]


        # ====================================================
        # STEP 5
        # ADD CURRENT MESSAGE IF NECESSARY
        # ====================================================
        #
        # Usually the SDK will already put the current message
        # into task history.
        #
        # We add it defensively in case it hasn't been persisted
        # yet.
        # ====================================================

        current_text = (
            get_message_text(message)
            or ""
        )

        if (
            current_text
            and current_text not in user_messages
        ):
            user_messages.append(
                current_text
            )


        # ====================================================
        # STEP 6
        # COMBINE USER MESSAGES
        # ====================================================

        asked = " ".join(
            user_messages
        )


        # ====================================================
        # STEP 7
        # FIND DURATION
        # ====================================================

        days = _days(
            asked
        )


        # ====================================================
        # INPUT-REQUIRED
        # ====================================================
        #
        # If the user didn't provide the duration:
        #
        #     Task
        #       |
        #       v
        #     input-required
        #
        # The task stays alive.
        #
        # The user can provide additional information later.
        #
        # ====================================================

        if days is None:

            await updater.requires_input(
                new_text_message(
                    "How many days is the trip?"
                )
            )

            return


        # ====================================================
        # STEP 8
        # WE HAVE ENOUGH INFORMATION
        # ====================================================
        #
        # Now move the task to working.
        # ====================================================

        await updater.start_work(
            new_text_message(
                "Planning your trip..."
            )
        )


        # ====================================================
        # STEP 9
        # RECOVER DESTINATION
        # ====================================================

        destination = _destination(
            user_messages
        )


        # ====================================================
        # STEP 10
        # GENERATE ITINERARY
        # ====================================================
        #
        # This represents the agent's business logic.
        #
        # In a real agent this could call:
        #
        #     LLM
        #     Search
        #     Maps
        #     Travel API
        #     Database
        #
        # ====================================================

        lines: list[str] = []

        for day in range(
            1,
            min(days, 14) + 1,
        ):

            # Simulate work.
            await asyncio.sleep(
                0.4
            )

            lines.append(
                f"Day {day}: "
                f"explore {destination}"
            )


        # Combine itinerary.
        plan = "\n".join(
            lines
        )


        # ====================================================
        # STEP 11
        # CREATE ARTIFACT
        # ====================================================
        #
        # The itinerary is the actual output produced by
        # the agent.
        #
        # ====================================================

        await updater.add_artifact(

            parts=[
                new_text_part(
                    text=plan,
                    media_type="text/plain",
                )
            ],

            name="itinerary",
        )


        # ====================================================
        # STEP 12
        # COMPLETE TASK
        # ====================================================

        await updater.complete(
            new_text_message(
                "Trip plan completed."
            )
        )


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
        ).cancel(
            new_text_message(
                "Planning was cancelled."
            )
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    run_agent(
        CARD,
        PlannerAgentExecutor(),
        PORT,
    )