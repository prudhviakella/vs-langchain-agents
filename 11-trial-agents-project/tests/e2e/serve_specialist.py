"""Run one specialist as a real A2A server: its real main.py executor, real
agent loop and middleware, real Lambda handler — with a scripted model and
fake data stores. Usage: python serve_specialist.py <trial_graph|trial_search> <port>
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fakes                                                   # noqa: E402
from fakes import Fake, GuardrailClient, call, mcp_tool        # noqa: E402
from pydantic import BaseModel                                  # noqa: E402

agent, port = sys.argv[1], int(sys.argv[2])

# Record this server's finished spans, one JSON line each, so the test can
# check them against the supervisor's trace from another process.
if os.environ.get("E2E_SPAN_FILE"):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class ToFile(SpanExporter):
        def export(self, spans):
            with open(os.environ["E2E_SPAN_FILE"], "a") as f:
                for sp in spans:
                    f.write(json.dumps({
                        "name": sp.name, "trace": format(sp.context.trace_id, "032x"),
                        "parent": format(sp.parent.span_id, "016x") if sp.parent else None}) + "\n")
            return SpanExportResult.SUCCESS
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ToFile()))
    trace.set_tracer_provider(provider)
sys.path.insert(0, str(fakes.ROOT / agent / "agent_code"))
import importlib                                                # noqa: E402
main = importlib.import_module("main")
core = importlib.import_module(f"{agent}.core")
importlib.import_module(f"{agent}.guardrail")._client = GuardrailClient()
settings = fakes.settings_for(agent)
handler = fakes.load_lambda(agent)


class Session:
    def __enter__(self): return self
    def __exit__(self, *a): pass

    def run(self, query, parameters=None, **kwargs):        # real driver signature
        if "fulltext" in query:
            return [{"labels": ["Sponsor"], "score": 3.0, "props": {"name": "Pfizer Inc."}}]
        if query.startswith("EXPLAIN"):
            return []
        return [{"nctId": "NCT02951156", "phase": "PHASE3"}]


class Q(BaseModel):
    query: str


class N(BaseModel):
    name: str
    entity_type: str = "any"


if agent == "trial_graph":
    handler._driver = type("D", (), {"session": lambda self: Session()})()
    P = "trial-graph-tools___"
    tools = [mcp_tool(P, "find_entity_by_name", handler.lambda_handler, N),
             mcp_tool(P, "execute_cypher", handler.lambda_handler, Q)]
    script = lambda: [call(P + "find_entity_by_name", name="Pfizer", entity_type="sponsor"),
                      call(P + "execute_cypher", query='MATCH (s:Sponsor {name: "Pfizer Inc."})'
                           '<-[:SPONSORED_BY]-(t:Trial) RETURN t.nctId AS nctId, t.phase AS phase'),
                      call("ModelDecision", entities=["NCT02951156"], answerable=True, note="")]
else:
    class Idx:
        def query(self, vector, top_k, include_metadata, filter=None):
            meta = {"doc_id": "nct02014597-glaucoma-optokinetic", "content_type": "text",
                    "headings": ["E4. Neonates"], "page": 6, "position": 24,
                    "text": "Neonates will not be enrolled."}
            m = type("M", (), {"id": "c24", "score": 0.9, "metadata": meta})()
            return type("R", (), {"matches": [m]})()

    class Embed:
        class embeddings:
            @staticmethod
            def create(model, input):
                return type("R", (), {"data": [type("D", (), {"embedding": [0.0] * 4})()]})()
    handler._index, handler._openai = Idx(), Embed
    P = "trial-search-tools___"

    class S(BaseModel):
        query: str
        top_k: int = 8
    tools = [mcp_tool(P, "semantic_search", handler.lambda_handler, S)]
    script = lambda: [call(P + "semantic_search", query="neonates"),
                      call("ModelDecision", entities=["nct02014597-glaucoma-optokinetic"],
                           answerable=True, note="")]


async def orchestrate(question):
    if os.environ.get("E2E_FAIL") == agent:          # proves the crash check can fail
        raise RuntimeError("deliberate failure for the e2e crash check")
    result = await core.build_agent(tools, model=Fake(messages=iter(script())),
                                    cfg=settings).ainvoke(
        {"messages": [{"role": "user", "content": question}]})
    return core.assemble(result, cfg=settings)

main.orchestrate = orchestrate
from bedrock_agentcore.runtime import serve_a2a                 # noqa: E402
serve_a2a(getattr(main, {"trial_graph": "TrialGraphExecutor",
                         "trial_search": "TrialSearchExecutor"}[agent])(),
          port=port, host="127.0.0.1")
