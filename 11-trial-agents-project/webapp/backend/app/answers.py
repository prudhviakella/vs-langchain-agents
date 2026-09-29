"""SupervisorResponse -> what the analyst sees beside an answer, and what
AgentOps lists.

    supervisor response                        AnswerDetails (models.py)
    ───────────────────                        ─────────────────────────
    calls[]  + results[agent].cypher/stats ─►  agents[]     who ran, why, the query
    tool_calls[]                           ─►  tools[]      memory reads and writes
    results[*].passages                    ─►  citations[]  best re-ranked first, ≤ 12
    render_target + results table/graph    ─►  artifact     capped rows / nodes
    results["memory:*"].items              ─►  memory[]     what was recalled
    decision                               ─►  decision     note, resolved question
    usage + prices                         ─►  usage, cost_usd
    trace_id                               ─►  trace_url    CloudWatch X-Ray trace

WHAT IS SHOWN IS THE AGENT'S DECISION TRAIL, NOT ITS PRIVATE REASONING

Each agent step carries the `rationale` the supervisor wrote as a tool
argument — a statement written for the analyst — and the query that actually
ran. The model's hidden reasoning is never part of the response and is not
shown.

ONE QUERY PER AGENT

The supervisor keeps each specialist's LAST result. When it calls the same
specialist twice in a turn, the query is attached to the last call; the
earlier call shows its question and rationale only.

WHAT THIS DOES NOT DO

    It does not show searches trial_search did not run: `searches` is
    recorded by trial_search's middleware from each executed call.
"""
import json

from . import settings

MAX_CITATIONS, SNIPPET_CHARS = 12, 400
MAX_ROWS, MAX_NODES = 200, 150
MAX_DETAILS_BYTES = 300_000            # DynamoDB items are capped at 400 KB


def trace_url(trace_id: str) -> str:
    """CloudWatch X-Ray console link. X-Ray ids are the W3C id split 8|24."""
    if len(trace_id) != 32:
        return ""
    xray = f"1-{trace_id[:8]}-{trace_id[8:]}"
    region = settings.REGION
    return (f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
            f"#xray:traces/{xray}")


def cost_usd(usage: dict) -> float:
    cached = usage.get("cached_input_tokens", 0) or 0
    fresh = max((usage.get("input_tokens", 0) or 0) - cached, 0)
    return round((fresh * settings.PRICE_INPUT_PER_MTOK
                  + cached * settings.PRICE_CACHED_INPUT_PER_MTOK
                  + (usage.get("output_tokens", 0) or 0) * settings.PRICE_OUTPUT_PER_MTOK)
                 / 1_000_000, 6)


def _agents(calls: list[dict], results: dict) -> list[dict]:
    last = {c["agent_name"]: i for i, c in enumerate(calls)}
    steps = []
    for i, c in enumerate(calls):
        result = results.get(c["agent_name"], {}) if last[c["agent_name"]] == i else {}
        steps.append({"agent": c["agent_name"], "question": c.get("question", ""),
                      "rationale": c.get("rationale", ""),
                      "result_shape": c.get("result_shape", ""),
                      "succeeded": bool(c.get("succeeded")),
                      "query": result.get("cypher") or None,
                      "stats": result.get("stats") or None,
                      "searches": result.get("searches") or None})
    return steps


def _citations(results: dict) -> list[dict]:
    seen, passages = set(), []
    for key, result in results.items():
        if result.get("result_shape") != "passages":
            continue
        for p in result.get("passages", []):
            if p.get("chunk_id") in seen:
                continue
            seen.add(p.get("chunk_id"))
            passages.append(p)
    passages.sort(key=lambda p: (p.get("rerank_score") is None,
                                 -(p.get("rerank_score") or p.get("score") or 0)))
    return [{"n": n, "doc_id": p.get("doc_id", ""), "page": p.get("page"),
             "headings": list(p.get("headings") or []),
             "snippet": str(p.get("text", ""))[:SNIPPET_CHARS],
             "rerank_score": p.get("rerank_score"), "origin": p.get("origin", "search")}
            for n, p in enumerate(passages[:MAX_CITATIONS], start=1)]


def _artifact(render_target: str, results: dict) -> dict | None:
    for agent, result in results.items():
        shape = result.get("result_shape")
        if render_target == "graph" and shape == "graph":
            nodes = result.get("nodes", [])[:MAX_NODES]
            kept = {n.get("element_id") for n in nodes}
            rels = [r for r in result.get("relationships", [])
                    if r.get("start") in kept and r.get("end") in kept]
            return {"kind": "graph", "agent": agent, "nodes": nodes, "relationships": rels,
                    "total_nodes": len(result.get("nodes", []))}
        if render_target == "chart" and shape == "table":
            rows = result.get("rows", [])
            return {"kind": "table", "agent": agent, "columns": result.get("columns", []),
                    "rows": rows[:MAX_ROWS], "total_rows": len(rows)}
    return None


def _memory(results: dict) -> list[dict]:
    return [{"kind": r.get("kind"), "text": i.get("text", ""),
             "created_at": i.get("created_at", ""), "score": i.get("score")}
            for key, r in results.items() if key.startswith("memory:")
            for i in r.get("items", [])]


def _fit(details: dict) -> dict:
    """Shrink until the stored JSON is safely under the item limit."""
    for step in range(6):
        if len(json.dumps(details)) <= MAX_DETAILS_BYTES:
            return details
        if details.get("artifact") and details["artifact"].get("rows"):
            details["artifact"]["rows"] = details["artifact"]["rows"][: max(10, MAX_ROWS >> (step + 1))]
        if details.get("artifact") and details["artifact"].get("nodes"):
            details["artifact"]["nodes"] = details["artifact"]["nodes"][: max(10, MAX_NODES >> (step + 1))]
        for c in details["citations"]:
            c["snippet"] = c["snippet"][:200]
    details["artifact"] = None
    return details


def details(response: dict, latency_ms: int) -> dict:
    results = response.get("results") or {}
    usage = response.get("usage") or {}
    trace_id = response.get("trace_id", "")
    return _fit({
        "agents": _agents(response.get("calls") or [], results),
        "tools": response.get("tool_calls") or [],
        "citations": _citations(results),
        "artifact": _artifact(response.get("render_target", "none"), results),
        "memory": _memory(results),
        "decision": response.get("decision") or {},
        "usage": usage, "cost_usd": cost_usd(usage), "latency_ms": latency_ms,
        "history_turns": response.get("history_turns", 0),
        "trace_id": trace_id, "trace_url": trace_url(trace_id)})


def interaction(message: dict, question: str, username: str) -> dict:
    """The AgentOps row for one answered (or failed) turn."""
    d = message.get("details") or {}
    usage = d.get("usage") or {}
    return {"conversation_id": message["conversation_id"],
            "message_id": message["message_id"], "username": username,
            "question": question[:500], "status": message["status"],
            "created_at": message["created_at"],
            "agents": [a["agent"] for a in d.get("agents", [])],
            "tools": [t["tool"] for t in d.get("tools", [])],
            **{k: int(usage.get(k, 0) or 0) for k in
               ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
                "llm_calls")},
            "cost_usd": d.get("cost_usd", 0.0), "latency_ms": d.get("latency_ms", 0),
            "trace_id": d.get("trace_id", ""), "trace_url": d.get("trace_url", "")}
