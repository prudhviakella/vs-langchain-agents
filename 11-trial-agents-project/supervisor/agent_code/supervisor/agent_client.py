"""Calls trial_graph and trial_search via AgentCore Runtime's data-plane
invoke_agent_runtime API — a genuinely different mechanism from how the
Supervisor's own tools reach the Gateway (SigV4-signed MCP).

    Supervisor's call_agent tool
        |
        v
    build_a2a_envelope(question)   one A2A JSON-RPC "message/send" request
        |
        v
    boto3 bedrock-agentcore client
        .invoke_agent_runtime(agentRuntimeArn=..., payload=envelope bytes)
        |
        v
    trial_graph or trial_search's own A2A executor (main.py) handles it,
    enqueues ONE Message event with its structured response as JSON text
        |
        v
    parse_a2a_response(raw bytes)  ->  the specialist's own response dict

WHY invoke_agent_runtime, NOT A DIRECT SIGV4-SIGNED HTTP CALL

The reference ACT Xerebro backend hand-signs a direct HTTPS POST to each
runtime's invocation URL, specifically because it needs message/stream's
SSE frames to relay live progress to an end-user's chat UI as it happens.
This Supervisor has no such requirement — trial_graph and trial_search
each enqueue exactly one final Message, not a stream of progress events,
so message/send (non-streaming) is the correct method, and boto3's own
invoke_agent_runtime already handles SigV4 signing internally. Manually
signing a raw request here would be replicating complexity this project
does not need, not matching the reference for its own sake.

WHY message/send, NOT message/stream

Confirmed directly from each specialist's own main.py: TrialGraphExecutor
and TrialSearchExecutor call event_queue.enqueue_event() exactly once,
with the complete final response — there is no intermediate progress
event to stream. message/stream exists in the A2A protocol for agents
that emit multiple events over time; these two don't.
"""
from __future__ import annotations

import json
import uuid

import boto3

_client = None


def _get_client():
    """Created lazily, on first use — not at module import time.

    Caught by testing this module directly: a module-level
    boto3.client("bedrock-agentcore") call fails outright with
    NoRegionError if AWS_REGION/AWS_DEFAULT_REGION isn't already set in
    the environment at import time. In the deployed container this
    would mean the whole agent fails to start on any environment
    ordering quirk, not just a test inconvenience — same reason
    lambda_tools/handler.py in trial_graph/trial_search initializes
    boto3 clients lazily rather than at module scope.
    """
    global _client
    if _client is None:
        _client = boto3.client("bedrock-agentcore")
    return _client


class AgentCallError(Exception):
    """Raised when a specialist agent's own invocation fails outright —
    a network/service error, not the specialist honestly answering
    'unanswerable'. Callers distinguish these: an AgentCallError is a
    real failure to surface; an unanswerable result_shape is a valid,
    honest answer to route through normally.
    """


def build_a2a_envelope(question: str, context_id: str) -> bytes:
    return json.dumps({
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {"message": {
            "role": "user",
            "messageId": f"msg-{uuid.uuid4().hex[:8]}",
            "parts": [{"kind": "text", "text": question}],
            "contextId": context_id,
        }},
    }).encode()


def _extract_text(parts: list) -> str:
    """A part may be {"text": ...} or {"root": {"text": ...}} depending
    on which side serialized it (Part(root=TextPart(...)).model_dump()
    produces the nested shape) — handle both rather than assume one.
    """
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
    """Parse the A2A JSON-RPC response and return the specialist's own
    structured response as a dict (already JSON-decoded, not a string).

    trial_graph and trial_search both enqueue a single, bare Message
    (see their own main.py) — the "immediate response" path the A2A
    AgentExecutor contract explicitly allows, as opposed to enqueuing a
    Task for long-running/resumable work. Confirmed this is genuinely
    what comes back, not assumed: started a real local A2A server using
    the exact same serve_a2a() this project's agents use, sent it a real
    message/send request, and inspected the actual JSON. The result is
    result["kind"]=="message" with parts directly on it — exactly the
    first branch below.

    The second branch (Task.artifacts / TaskArtifactUpdateEvent.artifact)
    is not dead-guess code — checked directly against a2a.types that
    these are real, distinct response shapes a differently-built
    specialist could return (Task.artifacts is a list; the streaming
    event's artifact is singular). trial_graph/trial_search don't use
    this path today, but a future specialist enqueuing a Task instead of
    a bare Message would, and this fallback means that specialist would
    still parse correctly without another round of guessing.
    """
    rpc = json.loads(raw)
    if "error" in rpc:
        raise AgentCallError(f"specialist returned a JSON-RPC error: {rpc['error']}")

    result = rpc.get("result") or {}

    # Bare Message: the confirmed, actual shape trial_graph/trial_search
    # produce today.
    parts = result.get("parts")

    # Task / TaskArtifactUpdateEvent: not yet produced by any agent in
    # this project, kept for a future specialist that enqueues one of
    # these instead — real, confirmed field names, not a guess.
    if parts is None:
        artifacts = result.get("artifacts") or []          # Task (list)
        single = result.get("artifact")                    # streaming event (singular)
        if isinstance(single, dict):
            artifacts = [*artifacts, single]
        for artifact in artifacts:
            if isinstance(artifact, dict) and artifact.get("parts"):
                parts = artifact["parts"]
                break

    text = _extract_text(parts or [])
    if not text:
        raise AgentCallError("specialist response had no readable text part")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # The specialist's own main.py sends an error string, not JSON,
        # when orchestrate() itself raised (see TrialGraphExecutor's
        # except block) — surface that plainly rather than crash here.
        raise AgentCallError(f"specialist returned a non-JSON response: {text[:300]}")


def call_specialist(agent_runtime_arn: str, question: str, context_id: str) -> dict:
    """Invoke one specialist agent and return its parsed structured
    response. Raises AgentCallError for anything that isn't a genuine,
    complete answer from the specialist itself.
    """
    payload = build_a2a_envelope(question, context_id)
    try:
        response = _get_client().invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            runtimeSessionId=context_id,
            payload=payload,
        )
        raw = response["response"].read()
    except Exception as exc:
        # A boto3/network-level failure — the specialist never got a
        # chance to answer at all. Distinct from AgentCallError raised
        # inside parse_a2a_response, but the caller treats both the same
        # way: a real failure to surface, not an honest empty result.
        raise AgentCallError(f"failed to invoke {agent_runtime_arn}: {exc}") from exc

    return parse_a2a_response(raw)
