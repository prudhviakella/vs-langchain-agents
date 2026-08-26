"""A2A client: discover an agent, send it work, read the results.

The client knows nothing about the agent until it fetches the card. That is the
design — no shared library, no imported schema, no agreed-on Python class. Just
a URL and a JSON document.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import aiohttp

from .types import AgentCard, Skill


class A2AClient:
    """Talks to one remote agent."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.card: AgentCard | None = None
        self._rpc_id = 0

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    # ── discovery ────────────────────────────────────────────────────────────

    async def discover(self) -> AgentCard:
        """Fetch the agent card.

        Always the first call. Until this returns you do not know what the agent
        does, where to POST, or whether you may stream to it.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{self.base_url}/.well-known/agent-card.json") as response:
                raw = await response.json()
        self.card = AgentCard(
            name=raw["name"], description=raw["description"], url=raw["url"],
            version=raw.get("version", "1.0.0"),
            capabilities=raw.get("capabilities", {}),
            skills=[Skill(**s) for s in raw.get("skills", [])],
        )
        return self.card

    # ── sending work ─────────────────────────────────────────────────────────

    async def send(self, text: str, task_id: str | None = None) -> dict[str, Any]:
        """Send a message and wait for the finished task.

        Simplest of the three, and the wrong choice for anything slow: the HTTP
        connection is held open for the whole run, so a long task hits a proxy
        timeout and you lose the result even though the server finished it.
        """
        payload = {
            "jsonrpc": "2.0", "id": self._next_id(), "method": "message/send",
            "params": {"message": {"role": "user",
                                   "parts": [{"kind": "text", "text": text}]}},
        }
        if task_id:
            payload["params"]["taskId"] = task_id
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload) as response:
                body = await response.json()
        if "error" in body:
            raise RuntimeError(body["error"]["message"])
        return body["result"]

    async def stream(self, text: str,
                     task_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        """Send a message and yield events as they happen.

        What you want for anything that takes longer than a second — the user
        sees progress instead of a spinner, and each event arrives as the server
        produces it rather than all at once at the end.
        """
        payload = {
            "jsonrpc": "2.0", "id": self._next_id(), "method": "message/stream",
            "params": {"message": {"role": "user",
                                   "parts": [{"kind": "text", "text": text}]}},
        }
        if task_id:
            payload["params"]["taskId"] = task_id
        async for event in self._sse(payload):
            yield event

    async def resubscribe(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """Reattach to a task already running.

        This is the method that justifies the whole queue. Your stream dropped,
        or you are a different process entirely — send the task id and you get
        the full history followed by live events, with no gap.
        """
        payload = {"jsonrpc": "2.0", "id": self._next_id(),
                   "method": "tasks/resubscribe", "params": {"id": task_id}}
        async for event in self._sse(payload):
            yield event

    async def get(self, task_id: str) -> dict[str, Any]:
        """Read a task's current state without streaming. Polling, for cron jobs."""
        payload = {"jsonrpc": "2.0", "id": self._next_id(),
                   "method": "tasks/get", "params": {"id": task_id}}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload) as response:
                body = await response.json()
        if "error" in body:
            raise RuntimeError(body["error"]["message"])
        return body["result"]

    async def cancel(self, task_id: str) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": self._next_id(),
                   "method": "tasks/cancel", "params": {"id": task_id}}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload) as response:
                return (await response.json())["result"]

    # ── SSE parsing ──────────────────────────────────────────────────────────

    async def _sse(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Read an SSE response line by line.

        The format is deliberately trivial: lines beginning `data: `, events
        separated by a blank line. No framing, no handshake, no library.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload) as response:
                async for raw in response.content:
                    line = raw.decode().strip()
                    if line.startswith("data: "):
                        yield json.loads(line[6:])
