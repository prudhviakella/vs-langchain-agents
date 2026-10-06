"""trial_graph: Lambda + agent loop, real create_agent under ainvoke."""
import asyncio

from pydantic import BaseModel

import fakes
from fakes import Fake, GuardrailClient, call, mcp_tool
from trial_graph import core, guardrail

P = "trial-graph-tools___"
handler = fakes.load_lambda("trial_graph")


class Session:
    """Mirrors neo4j's real Session.run(query, parameters=None, **kwargs)."""
    rows = [{"nctId": "NCT02951156", "phase": "PHASE3"}]

    def __enter__(self): return self
    def __exit__(self, *a): pass

    def run(self, query, parameters=None, **kwargs):
        if "fulltext" in query:
            assert parameters["search"] == "Pfizer"
            return [{"labels": ["Sponsor"], "score": 3.1,
                     "props": {"name": "Pfizer Inc.", "key": "pfizer inc"}}]
        return [] if query.startswith("EXPLAIN") else self.rows


handler._driver = type("D", (), {"session": lambda self: Session()})()


class Q(BaseModel):
    query: str


class N(BaseModel):
    name: str
    entity_type: str = "any"


def run(script, sent):
    guardrail._client = GuardrailClient()
    tools = [mcp_tool(P, "find_entity_by_name", handler.lambda_handler, N, sent),
             mcp_tool(P, "validate_cypher", handler.lambda_handler, Q, sent),
             mcp_tool(P, "execute_cypher", handler.lambda_handler, Q, sent)]
    s = fakes.settings_for("trial_graph")
    agent = core.build_agent(tools, model=Fake(messages=iter(script)), cfg=s)
    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "q"}]}))
    return result, core.assemble(result, cfg=s)


DECIDE = call("ModelDecision", entities=["NCT02951156"], answerable=True, note="")


def test_resolver_gives_the_exact_identity_anchor():
    sent = []
    result, _ = run([call(P + "find_entity_by_name", name="Pfizer", entity_type="sponsor"),
                     DECIDE], sent)
    view = next(m.content for m in result["messages"]
                if type(m).__name__ == "ToolMessage" and "candidate" in str(m.content))
    assert '(:Sponsor {name: "Pfizer Inc."})' in view
    assert "pfizer inc" not in view, "the lower-cased `key` must not be offered"


def test_writes_blocked_and_limits_enforced():
    sent = []
    result, response = run([
        call(P + "execute_cypher", query="MATCH (t:Trial) DETACH DELETE t"),
        call(P + "execute_cypher", query="MATCH (t:Trial) RETURN t.nctId"),
        call(P + "execute_cypher", query="MATCH (t:Trial) RETURN t LIMIT 99999"),
        call(P + "execute_cypher", query="MATCH (t:Trial) RETURN t LIMIT 5"),
        DECIDE], sent)
    queries = [a["query"] for n, a in sent if n == "execute_cypher"]
    assert not any("DELETE" in q for q in queries), "a write reached the Lambda"
    assert queries == ["MATCH (t:Trial) RETURN t.nctId LIMIT 500",
                       "MATCH (t:Trial) RETURN t LIMIT 500",
                       "MATCH (t:Trial) RETURN t LIMIT 5"]
    assert response.result_shape == "table" and response.rows == [["NCT02951156", "PHASE3"]]


def test_repair_budget_refuses_after_three_failures():
    Session.rows, sent = None, []
    original = handler.execute_cypher
    handler._TOOLS["execute_cypher"] = lambda a: {"error": True, "error_class": "SyntaxError",
                                                 "detail": "bad"}
    try:
        result, response = run([call(P + "execute_cypher", query=f"BAD {i} LIMIT 1")
                                for i in range(5)] + [DECIDE], sent)
    finally:
        handler._TOOLS["execute_cypher"] = original
        Session.rows = [{"nctId": "NCT02951156", "phase": "PHASE3"}]
    assert sum(1 for n, _ in sent if n == "execute_cypher") == 3
    assert any("REFUSED" in str(m.content) for m in result["messages"])
    assert response.result_shape == "not_executed"


def test_decision_without_a_query_is_not_executed():
    _, response = run([DECIDE], [])
    assert response.result_shape == "not_executed"


def test_two_queries_in_one_turn_run_one_at_a_time():
    """Production crash: the model put two execute_cypher calls in one message.
    Both ran in one graph step and both wrote `captured` ->
    InvalidUpdateError. Now the first runs, the second is refused with a
    message (not counted as a repair), and the model can send it again."""
    from langchain_core.messages import AIMessage
    first, second = "MATCH (t:Trial) RETURN t.nctId", "MATCH (s:Sponsor) RETURN s.name"
    both = AIMessage(content="", tool_calls=[
        {"name": P + "execute_cypher", "args": {"query": first}, "id": "q1"},
        {"name": P + "find_entity_by_name", "args": {"name": "Pfizer"}, "id": "r1"},
        {"name": P + "execute_cypher", "args": {"query": second}, "id": "q2"}])
    sent = []
    result, response = run([both, call(P + "execute_cypher", query=second), DECIDE], sent)

    queries = [a["query"] for n, a in sent if n == "execute_cypher"]
    assert queries == [first + " LIMIT 500", second + " LIMIT 500"], "in order, one per turn"
    assert any(n == "find_entity_by_name" for n, _ in sent), "resolving stays parallel"
    refused = [m for m in result["messages"]
               if type(m).__name__ == "ToolMessage" and m.tool_call_id == "q2"]
    assert refused and "REFUSED" in refused[0].content
    assert result.get("repair_count", 0) == 0
    assert result["execute_calls"] == 2
    assert response.cypher == second + " LIMIT 500" and response.result_shape == "table"
