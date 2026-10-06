"""Supervisor memory and history — phase 1 of the application build.

    OVERVIEW   what memory holds is shown to the routing model on every
               model call, and the stored facts to the composer
    EPISODE    SupervisorDecision.episode is written after compose — only
               for a turn where a specialist call succeeded
    STORE      one record in DynamoDB and Pinecone under one memory_id;
               recall is scoped to the user; a restated fact is updated,
               not duplicated
    TOOLS      no user -> "unavailable", no store call; a store failure
               never raises into the loop; recalls are fenced and captured
    GUARD      a memory-only turn is accepted; a from_conversation answer
               is accepted only when history was really passed; a bare
               decision is still retried
    BUDGET     memory calls past the per-turn limit are refused
    HISTORY    earlier turns reach the model before the question, bounded;
               the question stays the last human message (the guardrail
               checks it)
    RESPONSE   tool calls, full results, history count, resolved question
    INFRA      the memory table payload passes botocore's own validator;
               the runtime role gets exactly the table and the secret
"""
import asyncio
import hashlib
import math

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import fakes
from fakes import Fake, GuardrailClient, call
from supervisor import agent_client, config, core, guardrail, memory

S = fakes.settings_for("supervisor")


# ── an in-memory stand-in for DynamoDB + Pinecone with real cosine search ──
def embed(text: str) -> list[float]:
    vec = [0.0] * 64
    for word in text.lower().replace(".", " ").split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Table:
    """put_item and query, shaped like the boto3 Table resource. query honours
    the one key condition memory.py uses, pages of 2 items, and Select=COUNT —
    so the pagination loop is exercised."""
    def __init__(self): self.items = {}
    def put_item(self, Item): self.items[(Item["user_id"], Item["memory_id"])] = Item

    def query(self, KeyConditionExpression, ExpressionAttributeValues, Select=None,
              ExclusiveStartKey=None):
        assert KeyConditionExpression == "user_id = :u AND begins_with(memory_id, :k)"
        user, prefix = ExpressionAttributeValues[":u"], ExpressionAttributeValues[":k"]
        rows = sorted((v for (u, m), v in self.items.items()
                       if u == user and m.startswith(prefix)), key=lambda i: i["memory_id"])
        start = ExclusiveStartKey["memory_id"] if ExclusiveStartKey else ""
        rows = [r for r in rows if r["memory_id"] > start]
        page, more = rows[:2], len(rows) > 2
        out = {"Count": len(page)}
        if Select != "COUNT":
            out["Items"] = page
        if more:
            out["LastEvaluatedKey"] = {"user_id": user, "memory_id": page[-1]["memory_id"]}
        return out


class Index:
    def __init__(self): self.vectors = {}          # (namespace, id) -> (values, metadata)

    def upsert(self, vectors, namespace):
        for v in vectors:
            self.vectors[(namespace, v["id"])] = (v["values"], v["metadata"])

    def query(self, vector, top_k, namespace, filter, include_metadata):
        user = filter["user_id"]["$eq"]
        scored = [(sum(a * b for a, b in zip(vector, values)), mid, meta)
                  for (ns, mid), (values, meta) in self.vectors.items()
                  if ns == namespace and meta["user_id"] == user]
        scored.sort(reverse=True)
        match = lambda s, i, m: type("M", (), {"id": i, "score": s, "metadata": m})()
        return type("R", (), {"matches": [match(*x) for x in scored[:top_k]]})()


@pytest.fixture(autouse=True)
def fresh():
    original = (agent_client.call_specialist, agent_client._client)
    config.use(S)
    guardrail._client = GuardrailClient()
    table, index = Table(), Index()
    memory.use_store(memory.MemoryStore(table, index, embed))
    token = memory.USER.set("prudhvi")
    yield table, index
    memory.USER.reset(token)
    memory.use_store(None)
    agent_client.call_specialist, agent_client._client = original


def run_tool(tool, **args):
    return tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": "t1"})


# ── STORE ────────────────────────────────────────────────────────────────
def test_one_record_in_both_stores_scoped_to_the_user(fresh):
    table, index = fresh
    out = run_tool(memory.remember_fact, fact="Focuses on phase 3 trials.", topic="focus")
    [(user, memory_id)] = table.items
    assert user == "prudhvi" and memory_id.startswith("semantic-")
    assert ("semantic", memory_id) in index.vectors
    assert out.update["tool_calls"][0]["succeeded"] and out.update["memory_calls"] == 1

    memory.USER.set("someone-else")
    other = run_tool(memory.recall_facts, query="phase 3 trials")
    assert "no semantic memories" in other.update["messages"][0].content


def test_a_restated_fact_updates_instead_of_duplicating(fresh):
    table, _ = fresh
    run_tool(memory.remember_fact, fact="Prefers results as tables.")
    second = run_tool(memory.remember_fact, fact="Prefers results as tables.")
    run_tool(memory.remember_fact, fact="Works on hepatocellular carcinoma trials.")
    assert len(table.items) == 2
    assert "updated the existing fact" in second.update["messages"][0].content


def test_episodes_carry_the_conversation_and_recall_is_fenced(fresh):
    table, _ = fresh
    record = memory.write_episode("Checked IMbrave150 (NCT03434379) exclusion "
                                  "criteria; hepatic encephalopathy excluded.",
                                  "answered", "conv-123")
    assert record["tool"] == "record_episode" and record["succeeded"]
    [item] = table.items.values()
    assert item["conversation_id"] == "conv-123" and item["kind"] == "episodic"

    out = run_tool(memory.recall_episodes, query="IMbrave150 exclusion criteria")
    text = out.update["messages"][0].content
    assert text.count("<untrusted_data>") == 1 and "NCT03434379" in text
    captured = out.update["captured_results"]["memory:episodic"]
    assert captured["result_shape"] == "memory" and len(captured["items"]) == 1


# ── TOOLS never guess an identity and never raise ────────────────────────
def test_no_user_means_no_memory_and_no_store_call(fresh):
    table, _ = fresh
    memory.USER.set(None)
    out = run_tool(memory.remember_fact, fact="anything")
    assert "no user identity" in out.update["messages"][0].content
    assert out.update["tool_calls"][0]["succeeded"] is False and table.items == {}


def test_a_store_failure_becomes_a_tool_result(fresh):
    class Broken:
        def query(self, **kw): raise ConnectionError("pinecone down")
    memory.use_store(memory.MemoryStore(Table(), Broken(), embed))
    out = run_tool(memory.recall_facts, query="format")
    assert "memory unavailable: ConnectionError" in out.update["messages"][0].content


# ── GUARD and BUDGET in the real agent loop ──────────────────────────────
DECISION = dict(entities=[], answerable=True, clarifying_question="", note="")


def react(script, history_turns=0):
    agent_client.call_specialist = lambda *a: pytest.fail("no specialist should be called")
    return asyncio.run(core.build_react_agent(model=Fake(messages=iter(script)), cfg=S)
                       .ainvoke({"messages": [{"role": "user", "content": "q"}],
                                 "history_turns": history_turns}))


def test_a_memory_only_turn_is_accepted_without_a_specialist(fresh):
    """A scripted model has no third message: had the guard retried, the
    loop would have failed asking for one."""
    out = react([call("remember_fact", fact="Focuses on phase 3 trials."),
                 call("SupervisorDecision", **DECISION)])
    assert out["memory_calls"] == 1 and out["call_count"] == 0
    assert out["structured_response"].answerable


def test_from_conversation_needs_history_that_was_really_passed(fresh):
    accepted = react([call("SupervisorDecision", **DECISION, from_conversation=True)],
                     history_turns=4)
    assert accepted["structured_response"].from_conversation

    with pytest.raises(Exception):          # no history -> guard retries -> script ends
        react([call("SupervisorDecision", **DECISION, from_conversation=True)])


def test_memory_calls_past_the_limit_are_refused(fresh):
    table, _ = fresh
    script = [call("remember_fact", fact=f"Fact number {i} about topic {i}.") for i in range(6)]
    out = react(script + [call("SupervisorDecision", **DECISION)])
    assert len(table.items) == S.max_memory_calls_per_turn == 4
    refused = [m for m in out["messages"] if "REFUSED" in str(getattr(m, "content", ""))]
    assert len(refused) == 2


# ── HISTORY reaches the model, bounded, question last ────────────────────
def test_history_is_bounded_and_the_question_stays_last():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i} " + "x" * 3000}
               for i in range(30)]
    msgs = core.history_messages(history, limit=20)
    assert len(msgs) == 20 and msgs[0].content.startswith("turn 10")
    assert all(len(m.content) <= 2000 for m in msgs)
    assert isinstance(msgs[0], HumanMessage) and isinstance(msgs[1], AIMessage)


def test_orchestrate_passes_history_and_returns_the_full_response(fresh):
    seen = []

    class Recording(Fake):
        def _generate(self, messages, *args, **kwargs):
            seen.append(messages)
            return super()._generate(messages, *args, **kwargs)

    def graph_specialist(arn, agent, question, conversation):
        return {"result_shape": "table", "columns": ["nctId"], "rows": [["NCT03548935"]],
                "cypher": "MATCH (t:Trial) RETURN t.nctId"}
    agent_client.call_specialist = graph_specialist
    models = iter([
        Recording(messages=iter([
            call("call_agent", agent_name="trial_graph", question="STEP 1 endpoint",
                 rationale="resolve"),
            call("SupervisorDecision", **{**DECISION, "entities": ["NCT03548935"],
                                          "resolved_question": "Primary endpoint of STEP 1?"})])),
        Recording(messages=iter([AIMessage(content="The primary endpoint is ...")]))])
    real = S.__class__.chat_model
    S.__class__.chat_model = lambda self: next(models)
    try:
        response = asyncio.run(core.orchestrate(
            "and its primary endpoint?", "c" * 36,
            history=[{"role": "user", "text": "Which trials does Novo Nordisk sponsor?"},
                     {"role": "assistant", "text": "PIONEER 4 and STEP 1."}],
            user_id="prudhvi"))
    finally:
        S.__class__.chat_model = real

    route_input = seen[0]
    humans = [m for m in route_input if isinstance(m, HumanMessage)]
    assert humans[0].content == "Which trials does Novo Nordisk sponsor?"
    assert humans[-1].content == "and its primary endpoint?"
    assert any(isinstance(m, AIMessage) and m.content == "PIONEER 4 and STEP 1."
               for m in route_input)
    assert "Primary endpoint of STEP 1?" in seen[-1][0].content     # composer used it

    assert response.history_turns == 2
    assert response.results["trial_graph"]["cypher"].startswith("MATCH")
    assert response.decision.resolved_question == "Primary endpoint of STEP 1?"
    assert memory.USER.get() == "prudhvi"


# ── INFRA ────────────────────────────────────────────────────────────────
def test_memory_table_payload_passes_botocore_validation():
    from test_payloads import Recording, infra
    ms = infra("supervisor", "memory_store")
    state = {"exists": False}

    def describe(**kw):
        if not state["exists"]:
            state["exists"] = True
            raise ms_client.exceptions.ResourceNotFoundException()
        return {"Table": {"TableArn": "arn:aws:dynamodb:us-east-1:1:table/trial-agents-memory"}}
    ms_client = Recording("dynamodb", {"describe_table": describe})
    ms.boto3 = type("B", (), {"client": staticmethod(lambda *a, **k: ms_client)})
    assert ms.ensure_table("us-east-1").endswith("table/trial-agents-memory")
    assert "create_table" in ms_client.calls


def test_runtime_role_grants_exactly_the_table_and_the_secret(monkeypatch):
    from test_payloads import infra
    iam = infra("supervisor", "runtime_iam")
    captured = {}
    monkeypatch.setattr(iam, "_account_region", lambda: ("1", "us-east-1"))
    monkeypatch.setattr(iam, "_ensure_role", lambda account: "arn:aws:iam::1:role/r")
    monkeypatch.setattr(iam, "_converge", lambda policies: captured.update(policies))
    iam.runtime_role(specialist_arns=["arn:s"], ecr_repo_arn="arn:e", guardrail_arn="arn:g",
                     prompt_arns=["arn:p"], secret_arn="arn:o", param_prefix="/p",
                     registry_path="/r", memory_table_arn="arn:table",
                     pinecone_secret_arn="arn:pinecone")
    assert captured["memory-table"][0]["Resource"] == "arn:table"
    assert "dynamodb:Scan" not in captured["memory-table"][0]["Action"]
    assert captured["read-pinecone-secret"][0]["Resource"] == "arn:pinecone"


def test_pinecone_calls_carry_a_ca_bundle_that_does_not_depend_on_the_os(monkeypatch):
    """Regression, from a real deploy: python.org Python on macOS has no CA
    bundle until "Install Certificates.command" runs, so urllib failed with
    CERTIFICATE_VERIFY_FAILED while boto3 worked. Every Pinecone call must
    carry its own bundle — certifi's, or botocore's when certifi is absent."""
    import builtins
    import io
    import json as _json
    import ssl
    from test_payloads import infra
    ms = infra("supervisor", "memory_store")
    assert ms._TLS.cert_store_stats()["x509_ca"] > 100

    real_import = builtins.__import__
    def no_certifi(name, *args, **kwargs):
        if name == "certifi":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_certifi)
    fallback = ms.ca_bundle()
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert fallback.endswith("botocore/cacert.pem")
    assert ssl.create_default_context(cafile=fallback).cert_store_stats()["x509_ca"] > 100

    seen = {}
    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def urlopen(request, timeout, context=None):
        seen["context"] = context
        return Response(_json.dumps({"status": {"ready": True}}).encode())
    monkeypatch.setattr(ms.urllib.request, "urlopen", urlopen)
    assert ms.ensure_index("pc-key") == ms.INDEX
    assert seen["context"] is ms._TLS


# ── OVERVIEW: the model sees what memory holds ───────────────────────────
def test_overview_shows_facts_in_full_and_counts_episodes(fresh):
    for i in range(3):
        run_tool(memory.remember_fact, fact=f"Distinct preference number {i} about area {i}.",
                 topic="preference")
    for i in range(5):
        memory.write_episode(f"episode {i}", "answered", "c")
    view = memory.overview()
    assert (view.available, view.fact_count, view.episode_count) == (True, 3, 5)
    text = view.for_router()
    assert "Stored facts: 3. Recorded episodes: 5." in text
    assert text.count("<untrusted_data>") == 1 and "Distinct preference number 2" in text


def test_overview_caps_facts_and_says_more_exist(fresh):
    for i in range(memory.MAX_FACTS_IN_CONTEXT + 3):
        run_tool(memory.remember_fact, fact=f"Unrelated fact {i} zz{i} qq{i}.")
    view = memory.overview()
    assert view.fact_count == memory.MAX_FACTS_IN_CONTEXT + 3
    assert len(view.facts) == memory.MAX_FACTS_IN_CONTEXT
    assert "recall_facts searches the rest" in view.for_router()


def test_overview_without_a_user_says_memory_is_unavailable(fresh):
    memory.USER.set(None)
    view = memory.overview()
    assert not view.available and "Do not call memory tools" in view.for_router()
    assert view.for_composer() == "No stored preferences."


def _orchestrate(script_route, compose_text="answer", specialist=None, template=None):
    """One full turn. Returns (response, every message list each model saw)."""
    seen = []

    class Recording(Fake):
        def _generate(self, messages, *args, **kwargs):
            seen.append(messages)
            return super()._generate(messages, *args, **kwargs)

    agent_client.call_specialist = specialist or (lambda *a: {
        "result_shape": "table", "columns": ["nctId"], "rows": [["NCT1"]]})
    models = iter([Recording(messages=iter(script_route)),
                   Recording(messages=iter([AIMessage(content=compose_text)]))])
    cfg = S if template is None else S.__class__(**{**S.__dict__, "compose_template": template})
    config.use(cfg)
    real = cfg.__class__.chat_model
    cfg.__class__.chat_model = lambda self: next(models)
    try:
        response = asyncio.run(core.orchestrate("q", "c" * 36, history=[], user_id="prudhvi"))
    finally:
        cfg.__class__.chat_model = real
        config.use(S)
    return response, seen


def test_router_sees_the_memory_overview_on_its_system_message(fresh):
    run_tool(memory.remember_fact, fact="Focuses on oncology trials.", topic="research focus")
    _, seen = _orchestrate([
        call("call_agent", agent_name="trial_graph", question="q", rationale="r"),
        call("SupervisorDecision", **DECISION)])
    from langchain_core.messages import SystemMessage
    system = [m for m in seen[0] if isinstance(m, SystemMessage)]
    assert system and "THIS ANALYST'S MEMORY" in system[0].content
    assert "Focuses on oncology trials." in system[0].content
    assert "test prompt" in system[0].content, "the managed prompt is kept, not replaced"


def test_composer_sees_the_stored_preferences(fresh):
    run_tool(memory.remember_fact, fact="Prefers results as tables.", topic="format")
    _, seen = _orchestrate([
        call("call_agent", agent_name="trial_graph", question="q", rationale="r"),
        call("SupervisorDecision", **DECISION)],
        template="Q: {{question}}\nA: {{analyst}}\nE: {{evidence}}")
    assert "Prefers results as tables." in seen[-1][0].content


def test_the_decisions_episode_is_recorded_after_a_successful_call(fresh):
    table, _ = fresh
    response, _ = _orchestrate([
        call("call_agent", agent_name="trial_graph", question="q", rationale="r"),
        call("SupervisorDecision", **DECISION, episode="Listed trials NCT1 for the analyst.")])
    episodes = [i for i in table.items.values() if i["kind"] == "episodic"]
    assert len(episodes) == 1 and episodes[0]["conversation_id"] == "c" * 36
    assert episodes[0]["outcome"] == "answered"
    assert [t.tool for t in response.tool_calls] == ["record_episode"]


def test_no_episode_without_a_successful_specialist_call(fresh):
    """A memory-only turn, or one whose every call failed, established nothing."""
    table, _ = fresh
    _orchestrate([call("remember_fact", fact="Focuses on phase 3 trials."),
                  call("SupervisorDecision", **DECISION, episode="Stored a preference.")])

    def failing(*a):
        raise agent_client.AgentCallError("down")
    _orchestrate([call("call_agent", agent_name="trial_graph", question="q", rationale="r"),
                  call("SupervisorDecision", **DECISION, episode="Tried the graph.")],
                 specialist=failing)
    assert not [i for i in table.items.values() if i["kind"] == "episodic"]


def test_record_episode_is_no_longer_a_tool():
    assert memory.TOOL_NAMES == {"remember_fact", "recall_facts", "recall_episodes"}


def test_no_trial_ids_in_supervisor_prompts():
    import re
    for name in ("system.md", "compose.md"):
        text = (fakes.ROOT / "supervisor" / "prompts" / name).read_text()
        assert not re.search(r"NCT\d{8}", text), f"an NCT number is hard-coded in {name}"
