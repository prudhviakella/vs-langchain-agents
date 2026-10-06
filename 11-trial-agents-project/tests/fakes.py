"""Shared test doubles. Every fake here mirrors the REAL signature of what it
replaces — a fake with a looser signature hides exactly the bugs these tests
exist to catch (a Cypher parameter named `query` colliding with
Session.run(query, ...) was found only because a fake mirrored the driver).

    Fake        a chat model that plays a scripted list of AI messages
    call        one scripted tool call
    mcp_tool    a tool shaped like langchain_mcp_adapters output: async,
                content_and_artifact, content as a LIST of text blocks,
                Gateway-prefixed name — dispatching to a real lambda_handler
    settings_for  a Settings object for an agent, no AWS involved
    GuardrailClient  a bedrock-runtime stand-in with the real ApplyGuardrail shape
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("TRIAL_DATA_DIR", "/mnt/user-data/uploads"))
for agent in ("trial_graph", "trial_search", "supervisor"):
    path = str(ROOT / agent / "agent_code")
    if path not in sys.path:
        sys.path.insert(0, path)


class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(_tool: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": _tool, "args": args,
                                              "id": f"{_tool[-10:]}-{id(args)}"}])


def load_lambda(agent: str):
    """Each Lambda is a module named `handler`; load under a unique name.

    In the deployed zip every lambda_tools/*.py sits at the root, so the
    handler imports its siblings by bare name ("from rerank import rerank").
    The folder goes on sys.path here to reproduce that layout."""
    folder = str(ROOT / agent / "lambda_tools")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(
        f"{agent}_handler", ROOT / agent / "lambda_tools" / "handler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mcp_tool(prefix: str, name: str, lambda_handler, schema, sent=None):
    ctx = type("Ctx", (), {"client_context": type("CC", (), {
        "custom": {"bedrockAgentCoreToolName": prefix + name}})()})()

    async def run(**kwargs):
        if sent is not None:
            sent.append((name, kwargs))
        return [{"type": "text", "text": json.dumps(lambda_handler(kwargs, ctx))}], None
    return StructuredTool.from_function(coroutine=run, name=prefix + name, description=name,
                                        args_schema=schema,
                                        response_format="content_and_artifact")


class GuardrailClient:
    """ApplyGuardrail stand-in, shaped like the real response.

    a trigger word   -> GUARDRAIL_INTERVENED, a content filter BLOCKED,
                        outputs = the blocked message
    an email address -> GUARDRAIL_INTERVENED, a PII entity ANONYMIZED,
                        outputs = the text with the email replaced by {EMAIL}
    otherwise        -> NONE
    Both interventions carry "assessments", as the real API does: only an
    assessment with action BLOCKED makes it a block.
    """
    EMAIL = __import__("re").compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")   # never ends on a dot

    def __init__(self, triggers=("BLOCKME",)):
        self.triggers, self.calls = triggers, []

    def apply_guardrail(self, guardrailIdentifier, guardrailVersion, source, content):
        text = content[0]["text"]["text"]
        self.calls.append((source, text))
        if any(t in text for t in self.triggers):
            return {"action": "GUARDRAIL_INTERVENED",
                    "outputs": [{"text": f"blocked-{source.lower()}"}],
                    "assessments": [{"contentPolicy": {"filters": [
                        {"type": "MISCONDUCT", "confidence": "HIGH", "action": "BLOCKED"}]}}]}
        if self.EMAIL.search(text):
            return {"action": "GUARDRAIL_INTERVENED",
                    "outputs": [{"text": self.EMAIL.sub("{EMAIL}", text)}],
                    "assessments": [{"sensitiveInformationPolicy": {"piiEntities": [
                        {"type": "EMAIL", "match": self.EMAIL.search(text).group(), "action": "ANONYMIZED"}]}}]}
        return {"action": "NONE", "outputs": []}


def settings_for(agent: str, **overrides):
    common = dict(region="us-east-1", openai_api_key="sk-test", openai_model="gpt-test",
                  system_prompt="test prompt", prompt_version="1",
                  guardrail_id="gr-test", guardrail_version="1")
    specific = {
        "trial_graph": dict(gateway_url="https://gw", row_cap=500, graph_node_cap=500,
                            max_repairs=3),
        "trial_search": dict(gateway_url="https://gw", max_resolve_calls=3, max_searches_per_turn=5,
                             max_neighbor_calls=3, max_table_calls=3, max_window=10,
                             expansion_token_budget=6000),
        "supervisor": dict(compose_template="Q: {{question}}\nE: {{evidence}}",
                           specialists={
                               "trial_graph": {"arn": "arn:aws:bedrock-agentcore:us-east-1:1:runtime/tg",
                                               "description": "graph"},
                               "trial_search": {"arn": "arn:aws:bedrock-agentcore:us-east-1:1:runtime/ts",
                                                "description": "search"}},
                           max_agent_calls_per_turn=3),
    }[agent]
    config = importlib.import_module(f"{agent}.config")
    return config.Settings(**{**common, **specific, **overrides})


_otel = None


def otel():
    """One process-wide OpenTelemetry SDK provider with an in-memory exporter.
    set_tracer_provider can be called only once per process, so every test
    shares this one and clears the exporter first."""
    global _otel
    if _otel is None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        provider, exporter = TracerProvider(), InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _otel = exporter
    _otel.clear()
    return _otel
