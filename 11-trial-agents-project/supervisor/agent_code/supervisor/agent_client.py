"""Calls trial_graph and trial_search through AgentCore Runtime.

    call_agent tool (core.py)
        │  conversation_id, agent_name, question
        v
    session_id(conversation_id, agent_name)      ≥ 33 chars, stable per pair
        │
        v
    build_a2a_envelope(question, conversation_id)   A2A JSON-RPC message/send
        │
        v
    boto3 bedrock-agentcore.invoke_agent_runtime   SigV4 under the
        │                                          Supervisor's own role
        v
    the specialist's main.py executor -> one Message -> response bytes
        │
        v
    parse_a2a_response(raw)  ->  the specialist's structured response dict

THE SESSION ID

invoke_agent_runtime requires runtimeSessionId of 33 to 256 characters
(checked against the service model). An earlier version passed the agent
name — "trial_graph", 11 characters — so every specialist call failed
validation before reaching the Runtime.

It is now "<conversation_id>:<agent_name>":

    same conversation, same specialist   -> same session: the Runtime
                                            routes to the warm microVM
    same conversation, other specialist  -> a different session
    no conversation id supplied          -> a fresh uuid4, so the id is
                                            always long enough

The same conversation_id is sent as the A2A contextId, so the specialist
sees one conversation across turns.

WHY boto3 AND NOT A HAND-SIGNED HTTP CALL

The reference backend signs its own HTTPS call because it relays
message/stream SSE frames to a browser. The specialists here enqueue
exactly one final Message, so message/send is the right method and
boto3 signs it. The specialists authenticate callers by IAM (the
Runtime default); there is no end-user token to propagate because this
corpus has no per-user access control (every chunk's access = public).

WHAT THIS DOES NOT DO

    It does not retry. A failed specialist call raises AgentCallError; the
    call_agent tool turns that into a result the model can act on.
"""
from __future__ import annotations

import json
import uuid

import boto3

_client = None
MIN_SESSION_ID = 33          # InvokeAgentRuntime.runtimeSessionId min length


class AgentCallError(Exception):
    """The specialist did not produce a usable answer — a network, service
    or protocol failure. Distinct from an honest "unanswerable", which is
    a valid result and is routed normally."""


def _get_client():
    """Created on first use. A module-level client fails with NoRegionError
    at import if the region is not yet set, taking the container down."""
    global _client
    if _client is None:
        _client = boto3.client("bedrock-agentcore")
    return _client


def session_id(conversation_id: str | None, agent_name: str) -> str:
    base = conversation_id or str(uuid.uuid4())
    sid = f"{base}:{agent_name}"
    if len(sid) < MIN_SESSION_ID:            # short caller-supplied id
        sid = f"{sid}:{uuid.uuid5(uuid.NAMESPACE_URL, sid)}"
    return sid[:256]


def build_a2a_envelope(question: str, context_id: str) -> bytes:
    return json.dumps({
        "jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "message/send",
        "params": {"message": {
            "role": "user", "messageId": f"msg-{uuid.uuid4().hex}",
            "parts": [{"kind": "text", "text": question}],
            "contextId": context_id}},
    }).encode()


def _extract_text(parts: list) -> str:
    """A part is {"text": ...} or {"root": {"text": ...}} depending on which
    side serialized it."""
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if not text and isinstance(part.get("root"), dict):
            text = part["root"].get("text")
        if text:
            return text
    return ""


def parse_a2a_response(raw: bytes) -> dict:
    """The specialist's structured response, JSON-decoded.

    The specialists enqueue one bare Message, so result["parts"] holds the
    answer — confirmed against a real serve_a2a() server, not assumed. The
    Task / TaskArtifactUpdateEvent branch uses field names checked against
    a2a.types, for a future specialist that enqueues a Task instead.
    """
    rpc = json.loads(raw)
    if "error" in rpc:
        raise AgentCallError(f"specialist returned a JSON-RPC error: {rpc['error']}")

    result = rpc.get("result") or {}
    parts = result.get("parts")
    if parts is None:
        artifacts = list(result.get("artifacts") or [])
        if isinstance(result.get("artifact"), dict):
            artifacts.append(result["artifact"])
        parts = next((a["parts"] for a in artifacts
                      if isinstance(a, dict) and a.get("parts")), None)

    text = _extract_text(parts or [])
    if not text:
        raise AgentCallError("specialist response had no readable text part")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # The specialist's main.py sends a plain error string when its own
        # orchestrate() raised.
        raise AgentCallError(f"specialist returned a non-JSON response: {text[:300]}")


def call_specialist(agent_runtime_arn: str, agent_name: str, question: str,
                    conversation_id: str | None) -> dict:
    context_id = conversation_id or str(uuid.uuid4())
    try:
        response = _get_client().invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn, qualifier="DEFAULT",
            runtimeSessionId=session_id(context_id, agent_name),
            contentType="application/json", accept="application/json",
            payload=build_a2a_envelope(question, context_id))
        raw = response["response"].read()
    except Exception as exc:
        raise AgentCallError(f"failed to invoke {agent_name}: {exc}") from exc
    return parse_a2a_response(raw)
