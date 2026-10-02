"""Bedrock Guardrail, applied through the ApplyGuardrail API.

    analyst question ──► before_agent ──► ApplyGuardrail(source=INPUT)
                                              │  intervened -> GuardrailBlocked
                                              v
                              agent loop (model <-> tools)
                                              │
    each model turn    ──► after_model  ──► ApplyGuardrail(source=OUTPUT)
                                              on model-authored text only

WHY A MIDDLEWARE AND NOT guardrail_config

guardrail_config is a parameter of Bedrock's Converse API. The chat model is
OpenAI, which never sees it: the guardrail would silently stop applying,
with no error anywhere. ApplyGuardrail evaluates text independently of any
model, so the same Bedrock guardrail works in front of any provider.

WHAT IS CHECKED, AND WHAT IS NOT

    checked     the analyst's question (all filters, including
                PROMPT_ATTACK, which exists only on input)
    checked     text the model writes: message content, and the `note` /
                `clarifying_question` fields of its final decision
    NOT checked retrieved passages and query results. Trial protocols
                describe deaths, overdoses and adverse events; a VIOLENCE
                filter applied to them blocks legitimate clinical evidence.
                They are evidence, fenced as <untrusted_data>, not output.

WHAT THIS DOES NOT DO

    It does not rewrite text. On intervention it raises GuardrailBlocked,
    carrying the guardrail's own configured message; orchestrate() turns
    that into an unanswerable result the caller can show.
"""
from __future__ import annotations

import asyncio
import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from .tracing import set_span_attrs, span

log = logging.getLogger("agent.guardrail")

_client = None


def _bedrock():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("bedrock-runtime")
    return _client


class GuardrailBlocked(Exception):
    def __init__(self, source: str, message: str):
        super().__init__(f"{source} blocked by guardrail")
        self.source, self.message = source, message


def _blocked(node) -> bool:
    """True if any policy in the assessments BLOCKED — as opposed to only
    masking. ApplyGuardrail reports each finding with an action: BLOCKED,
    ANONYMIZED or NONE, nested per policy (content filters, denied topics,
    words, personal data)."""
    if isinstance(node, dict):
        return node.get("action") == "BLOCKED" or any(_blocked(v) for v in node.values())
    if isinstance(node, list):
        return any(_blocked(v) for v in node)
    return False


def check(text: str, source: str, guardrail_id: str, guardrail_version: str,
          client=None) -> str:
    """The text to use: unchanged, or with personal data masked.

    Raises GuardrailBlocked only when a policy BLOCKED. A guardrail that only
    masked (an email -> {EMAIL}) also reports GUARDRAIL_INTERVENED; treating
    that as a block would refuse a question for containing an email, and
    refuse every answer quoting a protocol's contact phone number.
    """
    if not text or not text.strip():
        return text
    with span(f"guardrail.{source.lower()}", chars=len(text)) as sp:
        response = (client or _bedrock()).apply_guardrail(
            guardrailIdentifier=guardrail_id, guardrailVersion=guardrail_version,
            source=source, content=[{"text": {"text": text}}])
        set_span_attrs(sp, action=response.get("action"))
    if response.get("action") != "GUARDRAIL_INTERVENED":
        return text
    output = next((o.get("text") for o in response.get("outputs", []) if o.get("text")), None)
    if _blocked(response.get("assessments", [])):
        log.warning("guardrail blocked %s", source)
        raise GuardrailBlocked(source, output or "This request was blocked by a content policy.")
    log.info("guardrail masked personal data in %s", source)
    return output or text


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return ""


class GuardrailMiddleware(AgentMiddleware):
    """INPUT once per turn, OUTPUT after every model call."""

    def __init__(self, guardrail_id: str, guardrail_version: str,
                 decision_tool: str, client=None):
        super().__init__()
        self.args = (guardrail_id, guardrail_version)
        self.decision_tool = decision_tool
        self.client = client

    # ── INPUT ───────────────────────────────────────────────────────────
    def _question(self, state):
        human = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        return human[-1] if human else None

    def _masked(self, question, checked: str):
        """If personal data was masked, the masked question REPLACES the
        original — same message id, so the messages reducer overwrites it and
        the model never sees the email or phone number."""
        if question is None or checked == _text(question.content):
            return None
        return {"messages": [HumanMessage(content=checked, id=question.id)]}

    def before_agent(self, state, runtime):
        question = self._question(state)
        text = _text(question.content) if question else ""
        return self._masked(question, check(text, "INPUT", *self.args, client=self.client))

    async def abefore_agent(self, state, runtime):
        question = self._question(state)
        text = _text(question.content) if question else ""
        checked = await asyncio.to_thread(check, text, "INPUT", *self.args, client=self.client)
        return self._masked(question, checked)

    # ── OUTPUT ──────────────────────────────────────────────────────────
    def _authored(self, state) -> str:
        """Text the model wrote in the turn that just finished.

        The newest AIMessage, NOT messages[-1]. When the model emits its
        structured decision, create_agent appends the AIMessage AND a
        ToolMessage acknowledging it in the same step, so messages[-1] is the
        acknowledgement. An earlier version read messages[-1] and never
        checked the final decision at all — found by tracing what after_model
        actually receives in a real loop.
        """
        message = next((m for m in reversed(state.get("messages", []))
                        if isinstance(m, AIMessage)), None)
        if message is None:
            return ""
        parts = [_text(message.content)]
        for call in message.tool_calls or []:
            if call["name"].endswith(self.decision_tool):
                args = call.get("args", {})
                parts += [str(args.get("note") or ""), str(args.get("clarifying_question") or "")]
        return " ".join(p for p in parts if p)

    def after_model(self, state, runtime):
        check(self._authored(state), "OUTPUT", *self.args, client=self.client)
        return None

    async def aafter_model(self, state, runtime):
        await asyncio.to_thread(check, self._authored(state), "OUTPUT", *self.args,
                                client=self.client)
        return None
