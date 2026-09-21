"""
============================================================
A2A COUNTER AGENT
============================================================

Purpose
-------
This agent demonstrates a LONG-RUNNING A2A task with
streaming progress updates.

Example user request:

    "count to 10"

The agent will produce:

    counted 1/10
    counted 2/10
    counted 3/10
    ...
    counted 10/10

and finally:

    Finished counting to 10.


============================================================
A2A WORKFLOW
============================================================

                  A2A CLIENT
                       |
                       | message/send
                       | "count to 10"
                       v
              +------------------+
              |   A2A SERVER     |
              |                  |
              | SDK handles      |
              | protocol/HTTP    |
              +--------+---------+
                       |
                       | calls execute()
                       v
              +------------------+
              | CounterExecutor  |
              +--------+---------+
                       |
                       | create/get Task
                       v
                  Task: working
                       |
                       | add_artifact()
                       v
              +------------------+
              | EventQueue       |
              +--------+---------+
                       |
             +---------+---------+
             |         |         |
             v         v         v
           1/10      2/10      3/10 ... 10/10
             |         |         |
             +---------+---------+
                       |
                       v
                Task: completed


============================================================
IMPORTANT A2A CONCEPTS
============================================================

Agent Card
-----------
Describes what the agent is and what it can do.

AgentExecutor
-------------
Contains the actual business logic of the agent.

RequestContext
--------------
Contains information about the current A2A request,
including the incoming message and task.

EventQueue
----------
Used by the executor to publish task/artifact/status
events to the A2A server.

TaskUpdater
-----------
Provides convenient methods for changing task state and
publishing artifacts.

Artifact
--------
Represents a piece of work/output produced by the agent.

Task
----
Represents one unit of work.

============================================================

Run:

    python agents/counter_agent.py

Server:

    http://127.0.0.1:9102/

Agent Card:

    http://127.0.0.1:9102/.well-known/agent-card.json
"""

import asyncio
import re
import uuid


# ============================================================
# A2A HELPERS
# ============================================================

from a2a.helpers import (
    # Extract text from the incoming A2A message.
    get_message_text,

    # Create a new A2A task from the user's message.
    new_task_from_user_message,

    # Create a text message for task/status updates.
    new_text_message,

    # Create a text part for an artifact.
    new_text_part,
)


# ============================================================
# A2A SERVER COMPONENTS
# ============================================================

from a2a.server.agent_execution import (
    # Base class for our agent's business logic.
    AgentExecutor,

    # Information about the current A2A request.
    RequestContext,
)

from a2a.server.events import (
    # Queue through which the executor publishes events.
    EventQueue,
)

from a2a.server.tasks import (
    # Helper used to update task state/artifacts.
    TaskUpdater,
)


# ============================================================
# APPLICATION SERVER HELPERS
# ============================================================

from common import (
    # Creates the Agent Card.
    make_card,

    # Starts the A2A server.
    run_agent,
)


# ============================================================
# SERVER CONFIGURATION
# ============================================================

PORT = 9112

URL = f"http://127.0.0.1:{PORT}/"


# ============================================================
# AGENT CARD
# ============================================================
#
# The Agent Card is how another A2A agent/client discovers
# this agent.
#
# A client first calls:
#
#     GET /.well-known/agent-card.json
#
# and receives this information.
#
# ============================================================

CARD = make_card(
    name="counter",

    description=(
        "Counts to a number, one step per second, "
        "reporting progress."
    ),

    url=URL,

    skill_id="count",

    skill_name="Count slowly",

    skill_description=(
        "Counts to N, emitting progress updates."
    ),

    examples=[
        "count to 10"
    ],

    # The agent produces multiple events while working.
    streaming=True,
)


# ============================================================
# COUNTER AGENT EXECUTOR
# ============================================================
#
# This is the most important class from the application
# developer's perspective.
#
# The A2A SDK takes care of:
#
#   HTTP
#   JSON-RPC
#   request routing
#   task transport
#   event transport
#   streaming
#
# Our executor implements:
#
#   "What should the Counter Agent actually do?"
#
# ============================================================

class CounterAgentExecutor(AgentExecutor):

    # ========================================================
    # EXECUTE
    # ========================================================
    #
    # A2A WORKFLOW
    #
    # Client
    #   |
    #   | message/send
    #   v
    # A2A Server
    #   |
    #   | execute()
    #   v
    # this method
    #
    # ========================================================

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        # ----------------------------------------------------
        # STEP 1
        # Get the user's A2A message.
        #
        # Example:
        #
        #     "count to 10"
        # ----------------------------------------------------

        message = context.message

        if message is None:
            raise ValueError("A message is required")


        # ----------------------------------------------------
        # STEP 2
        # Get the existing Task.
        #
        # For a brand-new request, current_task may be None.
        #
        # Therefore we create a Task if necessary.
        # ----------------------------------------------------

        task = context.current_task


        # ----------------------------------------------------
        # NEW TASK WORKFLOW
        #
        # No existing task
        #       |
        #       v
        # Create Task
        #       |
        #       v
        # Publish Task to EventQueue
        #
        # ----------------------------------------------------

        if task is None:

            task = new_task_from_user_message(
                message
            )

            # Tell the A2A framework:
            #
            # "A new task has been created."
            #
            # The SDK/server can now expose this task
            # to the client.
            await event_queue.enqueue_event(task)


        # ----------------------------------------------------
        # STEP 3
        # Create a TaskUpdater.
        #
        # TaskUpdater is our interface for updating the task.
        #
        # We can use it to:
        #
        #   start_work()
        #   add_artifact()
        #   complete()
        #   cancel()
        #
        # ----------------------------------------------------

        updater = TaskUpdater(
            event_queue,
            task.id,
            task.context_id,
        )


        # ----------------------------------------------------
        # STEP 4
        # Move the task into WORKING state.
        #
        # Client will know:
        #
        #     Task → working
        #
        # and receive:
        #
        #     "Counting..."
        #
        # ----------------------------------------------------

        await updater.start_work(
            new_text_message(
                "Counting..."
            )
        )


        # ----------------------------------------------------
        # STEP 5
        # Extract the text from the user's message.
        #
        # Example:
        #
        #     "count to 10"
        #
        # becomes:
        #
        #     "count to 10"
        # ----------------------------------------------------

        text = get_message_text(message) or ""


        # ----------------------------------------------------
        # STEP 6
        # Extract the target number.
        #
        # Regex:
        #
        #     \d+
        #
        # finds numbers in the message.
        #
        # Example:
        #
        #     "count to 10"
        #
        # gives:
        #
        #     10
        #
        # If no number is found, use 5.
        #
        # Maximum allowed value is 60.
        # ----------------------------------------------------

        match = re.search(
            r"\d+",
            text,
        )

        target = min(
            int(match.group()) if match else 5,
            60,
        )


        # ====================================================
        # STEP 7
        # CREATE THE ARTIFACT ID
        # ====================================================
        #
        # This is VERY important for streaming artifacts.
        #
        # We want ONE artifact:
        #
        #     progress
        #
        # containing multiple chunks:
        #
        #     counted 1/10
        #     counted 2/10
        #     counted 3/10
        #     ...
        #
        # Therefore every update must use the SAME artifact_id.
        #
        # ====================================================

        artifact_id = str(
            uuid.uuid4()
        )


        # ====================================================
        # STEP 8
        # LONG-RUNNING BUSINESS LOGIC
        # ====================================================
        #
        # Now the actual agent work begins.
        #
        # Every iteration:
        #
        #     1. Generate progress
        #     2. Publish artifact update
        #     3. Wait one second
        #
        # ====================================================

        try:

            for n in range(
                1,
                target + 1,
            ):

                # ------------------------------------------------
                # STREAM PROGRESS
                # ------------------------------------------------
                #
                # First iteration:
                #
                #     append=False
                #
                # This CREATES the artifact.
                #
                # Following iterations:
                #
                #     append=True
                #
                # These APPEND to the same artifact.
                #
                # ------------------------------------------------

                await updater.add_artifact(

                    parts=[
                        new_text_part(
                            text=f"counted {n}/{target}",
                            media_type="text/plain",
                        )
                    ],

                    # Reuse the SAME artifact ID.
                    artifact_id=artifact_id,

                    # Create on first iteration.
                    # Append on subsequent iterations.
                    append=n > 1,

                    # Tell the SDK when this is the final
                    # chunk of the artifact.
                    last_chunk=n == target,

                    # Human-readable artifact name.
                    name="progress",
                )


                # ------------------------------------------------
                # Simulate long-running work.
                #
                # In a real application this could be:
                #
                #   - database processing
                #   - ML inference
                #   - file processing
                #   - API calls
                #   - data pipeline execution
                #
                # ------------------------------------------------

                await asyncio.sleep(1)


            # ====================================================
            # STEP 9
            # TASK COMPLETED
            # ====================================================
            #
            # After all work is finished:
            #
            #     Task → completed
            #
            # ====================================================

            await updater.complete(
                new_text_message(
                    f"Finished counting to {target}."
                )
            )


        # ====================================================
        # STEP 10
        # HANDLE ASYNC CANCELLATION
        # ====================================================
        #
        # If the underlying async operation is cancelled,
        # Python raises asyncio.CancelledError.
        #
        # We update the A2A task before propagating the
        # cancellation.
        # ====================================================

        except asyncio.CancelledError:

            await updater.cancel(
                new_text_message(
                    "Counting was cancelled."
                )
            )

            # Let the cancellation continue.
            raise


    # ========================================================
    # CANCEL
    # ========================================================
    #
    # This method is associated with the A2A cancellation flow.
    #
    # Client:
    #
    #     tasks/cancel
    #
    #       |
    #       v
    #
    # A2A Server:
    #
    #       |
    #       v
    #
    # executor.cancel()
    #
    #       |
    #       v
    #
    # TaskUpdater.cancel()
    #
    #       |
    #       v
    #
    # Task → canceled
    #
    # ========================================================

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        await TaskUpdater(
            event_queue,

            # Which task should be cancelled?
            context.task_id or "",

            # Which context does the task belong to?
            context.context_id or "",

        ).cancel(
            new_text_message(
                "Counting was cancelled."
            )
        )


# ============================================================
# START A2A SERVER
# ============================================================
#
# When this file is executed directly:
#
#     python agents/counter_agent.py
#
# the A2A server starts.
#
# ============================================================

if __name__ == "__main__":

    run_agent(
        CARD,
        CounterAgentExecutor(),
        PORT,
    )