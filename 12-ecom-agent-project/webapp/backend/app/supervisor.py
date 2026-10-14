"""One turn from the supervisor, as a stream of its envelopes.

    stream_turn({question, history, conversation_id})
      ├─ SUPERVISOR_ARN: boto3 invoke_agent_runtime (SigV4, accept text/event-stream) — read in a thread
      └─ SUPERVISOR_URL: httpx POST (a local supervisor A2A server)
    yields {"type": status | reasoning | token | final, …}; raises SupervisorError on a failed task

WHAT THIS DOES NOT DO
    It does not translate or store anything — turns.py does.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid

from .settings import S


class SupervisorError(Exception):
    pass


def _envelope(payload: dict) -> bytes:
    cid = payload.get("conversation_id") or uuid.uuid4().hex
    return json.dumps({"jsonrpc": "2.0", "id": "1", "method": "message/stream", "params": {"message": {
        "role": "user", "messageId": f"msg-{uuid.uuid4().hex}", "contextId": cid,
        "parts": [{"kind": "text", "text": json.dumps(payload, default=str)}]}}}).encode()


def frame_envelopes(frame: dict) -> list[dict]:
    """One JSON-RPC frame -> the supervisor envelopes it carries."""
    if "error" in frame:
        raise SupervisorError(str(frame["error"]))
    r = frame.get("result") or {}
    status = r.get("status") or {}
    parts = (status.get("message") or {}).get("parts") or (r.get("artifact") or {}).get("parts") or []
    texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
    if status.get("state") == "failed":
        detail = next((json.loads(t).get("error") for t in texts if t.startswith("{")), None)
        raise SupervisorError(detail or " ".join(texts) or "the supervisor failed")
    out = []
    for t in texts:
        try:
            out.append(json.loads(t))
        except ValueError:
            pass
    return out


def _lines(payload: dict):
    if S.supervisor_arn:
        import boto3
        client = boto3.client("bedrock-agentcore", region_name=S.region)
        sid = f"{payload.get('conversation_id') or uuid.uuid4().hex}:supervisor"
        resp = client.invoke_agent_runtime(agentRuntimeArn=S.supervisor_arn, qualifier="DEFAULT",
                                           runtimeSessionId=sid if len(sid) >= 33 else (sid + "-" + "0" * 33)[:40],
                                           contentType="application/json", accept="text/event-stream",
                                           payload=_envelope(payload))
        for raw in resp["response"].iter_lines():
            yield raw.decode("utf-8") if isinstance(raw, bytes) else raw
    else:
        import httpx
        with httpx.Client(timeout=S.turn_timeout_s) as c:
            with c.stream("POST", S.supervisor_url, content=_envelope(payload),
                          headers={"Content-Type": "application/json", "Accept": "text/event-stream"}) as resp:
                resp.raise_for_status()
                yield from resp.iter_lines()


async def stream_turn(payload: dict):
    if not (S.supervisor_arn or S.supervisor_url):
        raise SupervisorError("no supervisor configured (SUPERVISOR_ARN or SUPERVISOR_URL)")
    loop, q = asyncio.get_running_loop(), asyncio.Queue()

    def pump():
        try:
            for line in _lines(payload):
                if line and line.startswith("data:"):
                    loop.call_soon_threadsafe(q.put_nowait, ("frame", json.loads(line[5:])))
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, ("error", exc))
        loop.call_soon_threadsafe(q.put_nowait, ("end", None))

    threading.Thread(target=pump, daemon=True).start()
    while True:
        kind, item = await q.get()
        if kind == "frame":
            for env in frame_envelopes(item):
                yield env
        elif kind == "error":
            raise SupervisorError(str(item)) from item
        else:
            return
