"""The REAL ChatOpenAI, as Settings.chat_model() builds it, through the real
create_agent + ToolStrategy + middleware. Only the network is replaced — by an
HTTP transport that answers in the Responses API's own wire format.

Proves: every call goes to /v1/responses (never /chat/completions), the tools
are sent, a function_call comes back as a LangChain tool call, and the loop
ends in a structured decision.
"""
import asyncio
import json

import httpx
import openai
from langchain_core.tools import tool

import fakes
from fakes import GuardrailClient


def responses_body(call_id, name, arguments):
    return {"id": f"resp_{call_id}", "object": "response", "created_at": 0, "status": "completed",
            "model": "gpt-6-sol", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [{"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id,
                        "name": name, "arguments": json.dumps(arguments), "status": "completed"}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}}


def test_agent_loop_runs_over_the_responses_api():
    from trial_search import core, guardrail
    guardrail._client = GuardrailClient()
    s = fakes.settings_for("trial_search", openai_model="gpt-6-sol")
    script = iter([
        responses_body("c1", "trial-search-tools___semantic_search", {"query": "neonates"}),
        responses_body("c2", "ModelDecision", {"entities": ["d1"], "answerable": True, "note": ""})])
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append((request.url.path, body.get("model"), [t.get("name") for t in body.get("tools", [])]))
        return httpx.Response(200, json=next(script))

    model = s.chat_model()
    assert model.use_responses_api is True
    model.root_async_client = openai.AsyncOpenAI(
        api_key="sk-test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    @tool("trial-search-tools___semantic_search")
    async def search(query: str) -> str:
        """search"""
        return json.dumps({"passages": [{"chunk_id": "c", "doc_id": "d1", "text": "Neonates are "
                           "not enrolled.", "origin": "search", "position": 1, "score": 0.9}]})

    result = asyncio.run(core.build_agent([search], model=model, cfg=s).ainvoke(
        {"messages": [{"role": "user", "content": "Are neonates enrolled?"}]}))

    assert [path for path, _, _ in seen] == ["/v1/responses", "/v1/responses"]
    assert all(m == "gpt-6-sol" for _, m, _ in seen)
    assert "trial-search-tools___semantic_search" in seen[0][2] and "ModelDecision" in seen[0][2]
    assert result["search_calls"] == 1
    assert result["structured_response"].entities == ["d1"]
