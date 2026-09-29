"""One turn: the backend asks the supervisor, signed with SigV4.

    ask(question, conversation_id, history, username)
        │  A2A message/send
        │    parts[0].text      the question
        │    contextId          the conversation — keeps the supervisor's and
        │                       specialists' microVMs warm across turns
        │    metadata.history   earlier messages, oldest first (≤ 20)
        │    metadata.user_id   the signed-in username (scopes memory)
        v
    bedrock-agentcore.invoke_agent_runtime     SigV4 from the ECS task role —
        │                                      no token to mint, rotate or leak
        v
    SupervisorResponse (JSON in the reply's text part)

WHY NO RETRIES

A turn can take a minute and may already have written memory; replaying it
could run every specialist twice and record an episode twice. A failure is
shown to the user, who can ask again.

WHAT THIS DOES NOT DO

    It does not stream. The supervisor returns one final message; chat.py
    keeps the browser's connection alive with progress events meanwhile.
"""
import json
import uuid
from functools import lru_cache

from . import settings


class SupervisorError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config
    return boto3.client("bedrock-agentcore", region_name=settings.REGION, config=Config(
        read_timeout=settings.SUPERVISOR_TIMEOUT_S, connect_timeout=10,
        retries={"max_attempts": 1, "mode": "standard"}))


def session_id(conversation_id: str) -> str:
    """AgentCore requires 33-256 characters; one per conversation."""
    return f"app-conversation-{conversation_id}"[:256].ljust(33, "0")


def envelope(question: str, conversation_id: str, history: list[dict], username: str) -> bytes:
    return json.dumps({
        "jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "message/send",
        "params": {"message": {
            "role": "user", "messageId": f"msg-{uuid.uuid4().hex}",
            "parts": [{"kind": "text", "text": question}], "contextId": conversation_id,
            "metadata": {"history": history, "user_id": username}}}}).encode()


def _text(parts: list) -> str:
    texts = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        text = part.get("text") or (part.get("root") or {}).get("text")
        if text:
            texts.append(text)
    return "".join(texts)


def parse(raw: bytes) -> dict:
    """The supervisor's JSON response — a bare Message, or a Task's artifact."""
    rpc = json.loads(raw)
    if "error" in rpc:
        raise SupervisorError(f"supervisor returned an error: {rpc['error']}")
    result = rpc.get("result") or {}
    parts = result.get("parts")
    if parts is None:
        artifacts = list(result.get("artifacts") or [])
        parts = next((a["parts"] for a in artifacts if isinstance(a, dict) and a.get("parts")),
                     None)
    text = _text(parts or [])
    if not text:
        raise SupervisorError("supervisor reply had no text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise SupervisorError(text[:500])               # the supervisor's own error string


def ask(question: str, conversation_id: str, history: list[dict], username: str) -> dict:
    if not settings.SUPERVISOR_ARN:
        raise SupervisorError("SUPERVISOR_ARN is not configured")
    try:
        response = _client().invoke_agent_runtime(
            agentRuntimeArn=settings.SUPERVISOR_ARN, qualifier="DEFAULT",
            runtimeSessionId=session_id(conversation_id),
            contentType="application/json", accept="application/json",
            payload=envelope(question, conversation_id, history, username))
        raw = response["response"].read()
    except Exception as exc:
        if isinstance(exc, SupervisorError):
            raise
        raise SupervisorError(f"{type(exc).__name__}: {exc}") from exc
    return parse(raw)
