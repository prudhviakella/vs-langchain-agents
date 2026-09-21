"""A2A SDK Agent 1: immediate response + artifact.

Run:
    python agents/echo_agent.py

This replaces the custom aiohttp server, JSON-RPC implementation, Task model,
and EventQueue from the original mini implementation with the official A2A SDK.
"""

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

from common import make_card, run_agent

PORT = 9111
URL = f"http://127.0.0.1:{PORT}/"

CARD = make_card(
    name="echo",
    description="Reverses text. Exists to demonstrate the A2A protocol.",
    url=URL,
    skill_id="reverse",
    skill_name="Reverse text",
    skill_description="Returns the input text backwards.",
    examples=["hello world"],
    streaming=False,
)


class EchoAgentExecutor(AgentExecutor):
    """
        A2A Client
            │
            │ message/send
            │
            ▼
        A2A Server
            │
            │ No existing task
            ▼
        Create TASK-001
            │
            ▼
        execute(context, event_queue)
            │
            ▼
        Agent business logic
            │
            ▼
        Artifact / status
            │
            ▼
        completed
    """
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        {
          "method": "message/send",
          "params": {
            "message": {
              "role": "user",
              "parts": [
                {
                  "text": "Reverse this text"
                }
              ]
            }
          }
        }
        A2A Client
            │
            │  message/send
            │
            ▼
        A2A Server
            │
            ▼
        AgentExecutor.execute()


        :param context:
        :param event_queue:
        :return:
        """

        # Get the incoming A2A message from the RequestContext.
        # This is the message sent by the A2A client/user to this agent.
        message = context.message

        # An agent cannot process a request without a message.
        # Fail early if the message is missing.
        if message is None:
            raise ValueError("A message is required")

        # Get the task currently associated with this request.
        #
        # If this is a continuation of an existing task, the SDK provides
        # that task through context.current_task.
        #
        # Example:
        #
        # TASK-001
        #   ├── "Reverse this text"
        #   ├── input-required
        #   └── "Continue with this text"
        #
        # In that case, we should continue using TASK-001 rather than
        # creating another task.
        task = context.current_task

        # If there is NO existing task, this is a new piece of work.
        #
        # Create a new A2A Task from the incoming user message.
        if task is None:
            task = new_task_from_user_message(message)

            # Publish the newly created Task to the A2A event stream.
            #
            # The client can now see that the agent has created a task
            # and can track its lifecycle.
            await event_queue.enqueue_event(task)

        # TaskUpdater is a convenient SDK helper used to update the task.
        #
        # We provide:
        #   - event_queue : where task events are published
        #   - task.id     : which task we are updating
        #   - task.context_id : which A2A context the task belongs to
        #
        # The updater will be used below to publish the artifact and
        # mark the task as completed.
        updater = TaskUpdater(
            event_queue,
            task.id,
            task.context_id
        )

        # Extract the text from the incoming A2A message.
        #
        # If the message doesn't contain text, use an empty string
        # instead of None.
        text = get_message_text(message) or ""

        # This is the actual BUSINESS LOGIC of our Echo Agent.
        #
        # Reverse the text.
        #
        # Example:
        #   Input  -> "Hello A2A"
        #   Output -> "A2A olleH"
        reversed_text = text[::-1]

        # Publish the result as an A2A Artifact.
        #
        # Artifact represents the OUTPUT produced by the agent.
        #
        # Here:
        #   name       = "reversed"
        #   parts      = one text part containing the reversed text
        #   media_type = text/plain
        #
        # The artifact becomes part of the task's output.
        await updater.add_artifact(
            parts=[
                new_text_part(
                    text=reversed_text,
                    media_type="text/plain"
                )
            ],
            name="reversed",
        )

        # Mark the task as COMPLETED.
        #
        # This tells the A2A framework/client that the agent has
        # finished processing this task.
        #
        # At this point the lifecycle is essentially:
        #
        #   submitted
        #       ↓
        #   working
        #       ↓
        #   completed
        #
        # and the artifact containing the reversed text is available.
        await updater.complete()

    async def cancel(
            self,
            context: RequestContext,
            event_queue: EventQueue,
    ) -> None:
        """
        A2A Client
            │
            │ tasks/cancel
            │
            ▼
        A2A Server
            │
            │ Find existing task
            ▼
        TASK-001
            │
            ▼
        cancel(context, event_queue)
            │
            ▼
        TaskUpdater(...)
            │
            ▼
        updater.cancel()
            │
            ▼
        TASK-001 → CANCELED
            │
            ▼
        EventQueue
            │
            ▼
        A2A Client
        """

        # context contains information about the A2A request,
        # including the task ID and the A2A context ID.
        #
        # For example:
        #
        # context.task_id    = "TASK-001"
        # context.context_id = "CTX-001"

        # TaskUpdater is an A2A SDK helper used to update
        # the lifecycle state of an existing task.
        updater = TaskUpdater(
            event_queue,

            # Identify WHICH task should be cancelled.
            #
            # If task_id is missing, use an empty string.
            context.task_id or "",

            # Identify WHICH A2A context this task belongs to.
            #
            # If context_id is missing, use an empty string.
            context.context_id or "",
        )

        # Tell the A2A SDK:
        #
        #   "Change this task's status to CANCELED
        #    and publish the corresponding event."
        #
        # The client listening to the task's event stream
        # can then receive the cancellation status.
        await updater.cancel()


if __name__ == "__main__":
    run_agent(CARD, EchoAgentExecutor(), PORT)
