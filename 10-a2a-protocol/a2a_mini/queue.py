"""The event queue. This is the part most A2A tutorials skip.

A task outlives the HTTP request that started it. Everything else follows from
that one fact.

The agent is producing events — status changes, partial results — while a client
may or may not be listening. So the events cannot be written straight to an HTTP
response. They go into a queue, and whoever is listening drains it.

    executor  --put()-->  EventQueue  --subscribe()-->  SSE response
                              |
                              +-- replay: every event so far
                              +-- subscribers: one asyncio.Queue each

Three things this buys you, none of which are possible without it:

  the connection can drop        the task keeps running and keeps queueing
  a client can reattach          replay the history, then continue live
  two clients can watch          each subscriber gets its own queue

If you build A2A without this, `tasks/resubscribe` cannot work and a dropped
network connection kills work that was still running fine on the server.
"""

from __future__ import annotations

import asyncio
from typing import Any


class EventQueue:
    """Fan-out queue for one task's events.

    Each subscriber gets its own `asyncio.Queue`. A single shared queue would
    mean two watchers each receive half the events, which is worse than either
    of them getting none — the bug looks like flaky streaming rather than a
    design error.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._subscribers: list[asyncio.Queue] = []
        # Every event, kept so a late or reattaching subscriber can catch up.
        # A real deployment caps this or moves it to Redis; the mechanism is the
        # same and the reason for it is the same.
        self._replay: list[dict[str, Any]] = []
        self._closed = False

    # ── producing ────────────────────────────────────────────────────────────

    def put(self, event: dict[str, Any]) -> None:
        """Record an event and hand it to every current subscriber.

        Never blocks and never awaits. The agent doing the work must not be
        slowed down, or stopped, by whether anyone happens to be listening.
        """
        if self._closed:
            return
        self._replay.append(event)
        for queue in self._subscribers:
            queue.put_nowait(event)

    def close(self) -> None:
        """Signal that no more events will arrive.

        Subscribers get a None sentinel so their `async for` loops end. Without
        it, a client streaming a finished task waits forever on a queue nobody
        will ever write to again.
        """
        self._closed = True
        for queue in self._subscribers:
            queue.put_nowait(None)

    # ── consuming ────────────────────────────────────────────────────────────

    async def subscribe(self, replay: bool = True):
        """Yield events for one listener, oldest first, then live.

        `replay=True` is what makes reattaching work. A client whose connection
        dropped at event 4 resubscribes, receives 1 through 4 immediately, then
        continues from 5 — with no gap and no duplicate handling on its side.

        The subscriber's queue is created before the replay is sent, so an event
        arriving mid-replay is queued rather than lost. Getting that order wrong
        produces a race that only appears under load.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            if replay:
                for event in list(self._replay):
                    yield event
            if self._closed:
                return
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            # A client that disconnects must not leave its queue attached, or
            # put() keeps filling a queue nobody reads and memory grows for the
            # life of the process.
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    # ── inspection, for the notebooks ────────────────────────────────────────

    @property
    def event_count(self) -> int:
        return len(self._replay)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


class QueueRegistry:
    """One queue per task, looked up by task id.

    This is the whole reason a task id exists in the protocol. `tasks/resubscribe`
    is a dictionary lookup here — the client sends an id, and the server finds the
    queue that has been filling up in the meantime.
    """

    def __init__(self):
        self._queues: dict[str, EventQueue] = {}

    def create(self, task_id: str) -> EventQueue:
        queue = EventQueue(task_id)
        self._queues[task_id] = queue
        return queue

    def get(self, task_id: str) -> EventQueue | None:
        return self._queues.get(task_id)

    def drop(self, task_id: str) -> None:
        """Forget a finished task. A real server does this on a TTL."""
        self._queues.pop(task_id, None)

    def __len__(self) -> int:
        return len(self._queues)
