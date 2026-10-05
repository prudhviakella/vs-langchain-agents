"""Calling NLQ and chart_gen — A2A message/stream, their progress relayed while they work.

    call(agent, payload, on_progress)
      ├─ deployed: boto3 invoke_agent_runtime(accept=text/event-stream)  SigV4, the supervisor's role
      └─ local:    httpx POST <url>  Accept: text/event-stream
    each SSE frame:  status-update  -> progress JSON -> on_progress(event)   (live)
                     artifact-update -> the result JSON                        (returned)
                     state failed    -> SpecialistError(the agent's error text)

THE SESSION ID
    "<conversation>:<agent>", padded to AgentCore's 33-character minimum: the same
    conversation reaches the same warm specialist microVM; a shorter id fails
    validation before reaching the runtime (the trial supervisor hit exactly that).

WHAT THIS DOES NOT DO
    It does not retry a failed specialist: the supervisor reports the failure in
    the answer rather than silently asking twice.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid

from .config import CFG


class SpecialistError(Exception):
    pass


def session_id(conversation_id: str, agent: str) -> str:
    sid = f"{conversation_id or uuid.uuid4().hex}:{agent}"
    return sid if len(sid) >= 33 else (sid + "-" + "0" * 33)[:40]


def envelope(payload: dict, conversation_id: str) -> dict:
    message = {"role": "user", "messageId": f"msg-{uuid.uuid4().hex}",
               "parts": [{"kind": "text", "text": json.dumps(payload, default=str)}]}
    if conversation_id:
        message["contextId"] = conversation_id
    return {"jsonrpc": "2.0", "id": "1", "method": "message/stream", "params": {"message": message}}


def _texts(parts) -> list[str]:
    out = []
    for p in parts or []:
        t = p.get("text") if isinstance(p, dict) else None
        if not t and isinstance(p, dict) and isinstance(p.get("root"), dict):
            t = p["root"].get("text")
        if t:
            out.append(t)
    return out


def handle_frame(frame: dict, on_progress, agent: str):
    """One JSON-RPC frame -> progress (relayed) or the result (returned)."""
    if "error" in frame:
        raise SpecialistError(f"{agent}: {frame['error']}")
    r = frame.get("result") or {}
    status = r.get("status") or {}
    if status.get("state") == "failed":
        texts = _texts((status.get("message") or {}).get("parts"))
        detail = next((json.loads(t).get("error") for t in texts if t.startswith("{")), None) or " ".join(texts)
        raise SpecialistError(f"{agent} failed: {detail or 'unknown error'}")
    if r.get("kind") == "status-update":
        for t in _texts((status.get("message") or {}).get("parts")):
            try:
                on_progress(json.loads(t))
            except (ValueError, TypeError):
                pass
    elif r.get("kind") == "artifact-update":
        for t in _texts((r.get("artifact") or {}).get("parts")):
            return json.loads(t)
    return None


def _lines_deployed(arn: str, payload: dict, conversation_id: str, agent: str):
    import boto3
    client = boto3.client("bedrock-agentcore", region_name=CFG.region)
    resp = client.invoke_agent_runtime(agentRuntimeArn=arn, qualifier="DEFAULT",
                                       runtimeSessionId=session_id(conversation_id, agent),
                                       contentType="application/json", accept="text/event-stream",
                                       payload=json.dumps(envelope(payload, conversation_id)).encode())
    for raw in resp["response"].iter_lines():
        yield raw.decode("utf-8") if isinstance(raw, bytes) else raw


def _lines_local(url: str, payload: dict, conversation_id: str):
    import httpx
    with httpx.Client(timeout=CFG.specialist_timeout_s) as client:
        with client.stream("POST", url, json=envelope(payload, conversation_id),
                           headers={"Accept": "text/event-stream"}) as resp:
            resp.raise_for_status()
            yield from resp.iter_lines()


async def call(agent: str, payload: dict, on_progress, conversation_id: str = "") -> dict:
    """Run the specialist; relay its progress as it arrives; return its result."""
    arn, url = (CFG.nlq_arn, CFG.nlq_url) if agent == "nlq" else (CFG.chart_arn, CFG.chart_url)
    if not (arn or url):
        raise SpecialistError(f"{agent} is not configured (set {agent.upper()}_ARN or _URL)")
    loop, queue = asyncio.get_running_loop(), asyncio.Queue()

    def pump():                                   # the blocking read runs in a thread
        try:
            lines = _lines_deployed(arn, payload, conversation_id, agent) if arn else _lines_local(url, payload, conversation_id)
            for line in lines:
                if line and line.startswith("data:"):
                    loop.call_soon_threadsafe(queue.put_nowait, ("frame", json.loads(line[5:])))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

    threading.Thread(target=pump, daemon=True).start()
    result = None
    while True:
        kind, item = await asyncio.wait_for(queue.get(), timeout=CFG.specialist_timeout_s)
        if kind == "frame":
            got = handle_frame(item, on_progress, agent)
            result = got if got is not None else result
        elif kind == "error":
            raise SpecialistError(f"{agent}: {item}") from item
        else:
            break
    if result is None:
        raise SpecialistError(f"{agent} returned no result")
    return result
