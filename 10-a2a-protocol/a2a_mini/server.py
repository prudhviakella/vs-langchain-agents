"""A2A server: JSON-RPC over HTTP, plus SSE for streaming.

No SDK. Around 150 lines, and it is the whole protocol — because A2A is
deliberately built on things that already existed.

    GET  /.well-known/agent-card.json   discovery
    POST /                              JSON-RPC 2.0
         message/send                   run it, wait, return the task
         message/stream                 run it, stream events as SSE
         tasks/get                      what state is task X in
         tasks/resubscribe              reattach to task X's stream
         tasks/cancel                   stop task X

You write one function — the executor — and this serves it over A2A.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, AsyncIterator, Awaitable, Callable

from aiohttp import web

from .queue import EventQueue, QueueRegistry
from .types import TERMINAL, AgentCard, Artifact, Message, Part, Task

# An executor receives the task, the incoming message, and the task's queue.
# It reports progress by putting events on the queue; it does not return them.
# That separation is what lets the work continue when nobody is listening.
Executor = Callable[[Task, Message, EventQueue], Awaitable[None]]


class A2AServer:
    """Serves one agent over A2A."""

    def __init__(self, card: AgentCard, executor: Executor):
        self.card = card
        self.executor = executor
        self.tasks: dict[str, Task] = {}
        self.queues = QueueRegistry()
        # Handles to running work, so tasks/cancel has something to cancel and
        # so a task is not garbage collected mid-run.
        self._running: dict[str, asyncio.Task] = {}

    # ── events ───────────────────────────────────────────────────────────────

    def _status_event(self, task: Task, note: str = "") -> dict[str, Any]:
        """One status change, as an event.

        `final` ends the client's stream. It is true for the terminal states and
        also for `input-required` — because at that point the agent has stopped
        and will produce nothing further until the caller replies.

        Leaving it false there is a real bug and an easy one to write: the task
        is not finished, so `final` feels wrong. But a client waiting for `final`
        then waits forever on work that has already stopped, and the hang looks
        like a network problem rather than a protocol one.

        The task itself stays alive in `input-required`. Only the stream ends.
        """
        stream_ends = task.state in TERMINAL or task.state == "input-required"
        return {"kind": "status-update", "taskId": task.id, "state": task.state,
                "note": note, "final": stream_ends}

    def set_state(self, task: Task, state: str, note: str = "") -> None:
        """Change a task's state and tell anyone watching.

        Every state change goes through here, so no transition can happen without
        an event. A task that silently moves to `completed` leaves a streaming
        client waiting for an ending that already happened.
        """
        task.state = state                                   # type: ignore[assignment]
        queue = self.queues.get(task.id)
        if queue:
            queue.put(self._status_event(task, note))

    def add_artifact(self, task: Task, artifact: Artifact) -> None:
        """Attach an output and emit it, so streaming clients see it immediately."""
        task.artifacts.append(artifact)
        queue = self.queues.get(task.id)
        if queue:
            queue.put({"kind": "artifact-update", "taskId": task.id,
                       "artifact": asdict(artifact)})

    # ── running work ─────────────────────────────────────────────────────────

    async def _run(self, task: Task, message: Message) -> None:
        """Execute the agent, and make sure the queue always gets closed.

        The `finally` matters. An executor that raises leaves subscribers waiting
        on a queue nothing will ever write to — the client hangs, and the traceback
        is on the server where the client cannot see it.
        """
        queue = self.queues.get(task.id)
        try:
            self.set_state(task, "working")
            await self.executor(task, message, queue)
            if task.state not in TERMINAL and task.state != "input-required":
                self.set_state(task, "completed")
        except asyncio.CancelledError:
            self.set_state(task, "canceled")
            raise
        except Exception as exc:
            self.set_state(task, "failed", note=str(exc)[:200])
        finally:
            # Only close on a terminal state. A task parked in `input-required`
            # keeps its queue open, because it will produce more events when the
            # caller answers — on the same task id, into the same queue.
            if queue and task.state in TERMINAL:
                queue.close()

    def _start(self, message: Message, task_id: str | None = None) -> Task:
        """Create or continue a task, and start the work in the background.

        Started as a background task rather than awaited, so the HTTP handler can
        return or start streaming straight away. This is the line that makes a
        task outlive its request.
        """
        if task_id and task_id in self.tasks:
            task = self.tasks[task_id]           # continuing after input-required
        else:
            task = Task()
            self.tasks[task.id] = task
            self.queues.create(task.id)
        task.history.append(message)
        self._running[task.id] = asyncio.create_task(self._run(task, message))
        return task

    # ── HTTP ─────────────────────────────────────────────────────────────────

    async def _card(self, request: web.Request) -> web.Response:
        return web.json_response(self.card.to_dict())

    async def _rpc(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        method, params, rpc_id = body.get("method"), body.get("params", {}), body.get("id")

        if method == "message/send":
            task = self._start(_message_from(params["message"]),
                               params.get("taskId"))
            # Wait for it, because this method promises a finished task.
            await self._running[task.id]
            return _result(rpc_id, task.to_dict())

        if method == "message/stream":
            task = self._start(_message_from(params["message"]),
                               params.get("taskId"))
            return await self._sse(request, task.id, replay=True)

        if method == "tasks/get":
            task = self.tasks.get(params["id"])
            return _result(rpc_id, task.to_dict()) if task else _error(rpc_id, "no such task")

        if method == "tasks/resubscribe":
            # The whole point of the queue. The task has been running and queueing
            # this entire time; the client just reattaches and catches up.
            if params["id"] not in self.tasks:
                return _error(rpc_id, "no such task")
            return await self._sse(request, params["id"], replay=True)

        if method == "tasks/cancel":
            running = self._running.get(params["id"])
            if running and not running.done():
                running.cancel()
            return _result(rpc_id, {"id": params["id"], "canceled": True})

        return _error(rpc_id, f"unknown method {method}")

    async def _sse(self, request: web.Request, task_id: str,
                   replay: bool) -> web.StreamResponse:
        """Stream a task's events as Server-Sent Events.

        SSE rather than WebSocket because it is one-way, it is plain HTTP, and it
        reconnects on its own. A2A chose the boring option deliberately —
        proxies, load balancers and corporate networks all handle it already.
        """
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        })
        await response.prepare(request)

        queue = self.queues.get(task_id)
        if queue is None:
            await response.write(b"data: {\"error\":\"no queue\"}\n\n")
            return response

        try:
            async for event in queue.subscribe(replay=replay):
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
                if event.get("final"):
                    break
        except (ConnectionResetError, asyncio.CancelledError):
            # The client went away. The task keeps running, the queue keeps
            # filling, and the client can resubscribe later. Nothing is lost.
            pass
        return response

    def app(self) -> web.Application:
        application = web.Application()
        application.router.add_get("/.well-known/agent-card.json", self._card)
        application.router.add_post("/", self._rpc)
        return application

    def run(self, host: str = "127.0.0.1", port: int = 9101) -> None:
        print(f"{self.card.name} on http://{host}:{port}")
        web.run_app(self.app(), host=host, port=port, print=None)


# ── JSON-RPC helpers ────────────────────────────────────────────────────────

def _result(rpc_id: Any, result: Any) -> web.Response:
    return web.json_response({"jsonrpc": "2.0", "id": rpc_id, "result": result})


def _error(rpc_id: Any, message: str, code: int = -32000) -> web.Response:
    return web.json_response(
        {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})


def _message_from(raw: dict[str, Any]) -> Message:
    return Message(role=raw.get("role", "user"),
                   parts=[Part(kind=p.get("kind", "text"), text=p.get("text", ""))
                          for p in raw.get("parts", [])],
                   messageId=raw.get("messageId", ""))
