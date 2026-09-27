"""The guardrail middleware, inside a real create_agent loop under ainvoke.

Checked: the question (INPUT) and model-authored text (OUTPUT).
NOT checked: retrieved evidence — clinical text about deaths and adverse
events would trip VIOLENCE filters and block legitimate answers.
"""
import asyncio
import json
from typing import Optional

import pytest
from pydantic import BaseModel
from langchain_core.tools import StructuredTool

import fakes
from fakes import Fake, GuardrailClient, call
from trial_search import core, guardrail

P = "trial-search-tools___"


class SearchArgs(BaseModel):
    query: str
    top_k: Optional[int] = 8


def search_returning(text):
    async def run(**kw):
        payload = {"passages": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.9, "text": text,
                                 "origin": "search", "position": 1}]}
        return [{"type": "text", "text": json.dumps(payload)}], None
    return StructuredTool.from_function(coroutine=run, name=P + "semantic_search",
                                        description="s", args_schema=SearchArgs,
                                        response_format="content_and_artifact")


def run(question, script, passage_text="neutral clinical text"):
    client = GuardrailClient()
    guardrail._client = client
    model = Fake(messages=iter(script))
    agent = core.build_agent([search_returning(passage_text)], model=model,
                             cfg=fakes.settings_for("trial_search"))
    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": question}]}))
    return result, client


DECIDE = call("ModelDecision", entities=["d1"], answerable=True, note="grounded")


def test_clean_turn_checks_input_once_and_every_model_turn():
    result, client = run("what are the eligibility criteria?",
                         [call(P + "semantic_search", query="eligibility"), DECIDE])
    sources = [s for s, _ in client.calls]
    assert sources[0] == "INPUT" and sources.count("INPUT") == 1
    # The tool-call turn wrote no text, so nothing is sent for it; the final
    # decision's note is checked. Before the fix, the note was never checked:
    # after_model's messages[-1] is the structured-response acknowledgement.
    outputs = [text for source, text in client.calls if source == "OUTPUT"]
    assert outputs == ["grounded"]
    assert result["structured_response"].answerable


def test_blocked_question_stops_before_any_model_call():
    script = iter([DECIDE])
    with pytest.raises(guardrail.GuardrailBlocked) as blocked:
        client = GuardrailClient(); guardrail._client = client
        agent = core.build_agent([search_returning("x")], model=Fake(messages=script),
                                 cfg=fakes.settings_for("trial_search"))
        asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "BLOCKME now"}]}))
    assert blocked.value.source == "INPUT" and blocked.value.message == "blocked-input"
    assert next(script, None) is DECIDE, "the model must never have been called"


def test_model_authored_note_is_checked():
    note = call("ModelDecision", entities=[], answerable=True, note="BLOCKME in my note")
    with pytest.raises(guardrail.GuardrailBlocked) as blocked:
        run("q", [call(P + "semantic_search", query="q"), note])
    assert blocked.value.source == "OUTPUT"


def test_retrieved_evidence_is_not_guardrailed():
    """A passage containing the trigger reaches the model untouched."""
    result, client = run("q", [call(P + "semantic_search", query="q"), DECIDE],
                         passage_text="adverse events: BLOCKME deaths reported")
    assert result["structured_response"].answerable
    assert not any("deaths reported" in text for _, text in client.calls)


def test_orchestrate_turns_a_block_into_an_unanswerable_result(monkeypatch):
    async def blocked_tools():
        raise guardrail.GuardrailBlocked("INPUT", "blocked-input")

    class Ctx:
        async def __aenter__(self): return await blocked_tools()
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(core, "connect_tools", lambda: Ctx())
    response = asyncio.run(core.orchestrate("anything"))
    assert response.result_shape == "unanswerable"
    assert "Blocked by guardrail (INPUT): blocked-input" == response.result_note
