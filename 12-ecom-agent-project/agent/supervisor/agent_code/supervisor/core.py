"""Supervisor core — one turn in, a stream of events out, ending with `final`.

    question + history
      ─► STEP 1 route (gpt-6-sol, structured RouteDecision)         reasoning: the plan
           ├─ answer  ─► reply streamed as tokens ─► final
           ├─ clarify ─► final kind=clarification
           └─ query
      ─► STEP 2 NLQ per sub-question (A2A stream)                    status + reasoning (the SQL, row counts)
      ─► STEP 3 chart_gen on the main table (≥ chart_min_rows rows)  status: charting
      ─► STEP 4 compose (gpt-6-sol, streamed)                        token × n
      ─► STEP 5 final  {kind, text, reasoning, entities, sources, artifacts{table, charts, graphs, exports},
                        artifact_summary, agents_used, usage_by_agent, plan_record, trace_id, options}

    sources — the CITATION, one record per NLQ call that ran a query (ACT's SourceRecord):
        {call_id, agent, store, query_language, query, question, row_count, sources: [tables], filters}
        The tables are read from the SQL that RAN, so a citation can never name a
        table the query did not read. A failed call is not cited.

    event types (what the backend relays as SSE):  status · reasoning · token · final

FROM ACT
    - A specialist's QUERY becomes a REASONING line, persisted with the turn —
      "here is the SQL that answered this" outlives the spinner.
    - The table in `final` is the data agent's rows, attached by CODE — never
      re-typed by a model. chart_gen's decision rides as is in artifacts.charts.
    - Entities are built from the data agent's entity labels, not invented here.

WHAT THIS DOES NOT DO
    It keeps no conversation: history arrives with each turn (the backend owns it).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from . import specialists
from .config import CFG, api_key
from .prompt import COMPOSE_PROMPT, ROUTE_PROMPT
from .schemas import RouteDecision

log = logging.getLogger("supervisor.core")


def chat_model(streaming: bool = False):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CFG.model, use_responses_api=True, api_key=api_key(), max_retries=4,
                      streaming=streaming, stream_usage=True)


def _history(history: list[dict]) -> list:
    kept, used = [], 0
    for m in reversed(history or []):                          # newest first, within budget
        text = str(m.get("text") or m.get("content") or "")
        if used + len(text) > CFG.history_chars and kept:
            break
        kept.append(HumanMessage(content=f"[{m.get('role', 'user')}] {text}"))
        used += len(text)
    return list(reversed(kept))


def _usage(meta: dict | None) -> dict:
    meta = meta or {}
    return {"input_tokens": int(meta.get("input_tokens") or 0), "output_tokens": int(meta.get("output_tokens") or 0),
            "total_tokens": int(meta.get("total_tokens") or 0)}


def _add(a: dict, b: dict) -> dict:
    return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in ("input_tokens", "output_tokens", "total_tokens")}


def _entities(labels: list[str]) -> list[dict]:
    """'customer 4817' / 'product 2297 (RAN-FIC-1296)' -> {eid, type, name}."""
    out, seen = [], set()
    for label in labels or []:
        m = re.match(r"\s*([a-z_]+)\s+([\w-]+)", str(label), re.I)
        if not m:
            continue
        eid = f"{m.group(1).lower()}:{m.group(2)}"
        if eid not in seen:
            seen.add(eid)
            out.append({"eid": eid, "type": m.group(1).lower(), "name": str(label).strip()})
    return out


def _source(q: str, r: dict) -> dict | None:
    """ACT's SourceRecord for one NLQ call — None when no query ran.
    Tables come from the SQL that RAN: NLQ always qualifies them as ecom.<table>."""
    sql = r.get("sql") or ""
    if "error" in r or not sql:
        return None
    tables = sorted({f"ecom.{t}" for t in re.findall(r"\becom\.(\w+)", sql)})
    return {"call_id": uuid.uuid4().hex, "agent": "nlq", "store": "PostgreSQL ecom", "query_language": "SQL",
            "query": sql, "question": q, "row_count": r.get("row_count"), "sources": tables, "filters": []}


def _nlq_line(ev: dict, attempt: dict) -> str | None:
    """One NLQ progress event -> one reasoning line (None for events that only drive the status line).

        grounding (with tables)  Schema: tables · business terms · joins · similar solved examples
        executing                Query n: the SQL, as a ```sql block
        error                    ↳ query n failed: <the error> — rewriting
        validated                ↳ plan check: <valid / invalid — why>
        result                   ↳ rows × columns (names) in ms, capped or not
        lookup                   Looked up '<term>' in the schema
    """
    phase = ev.get("phase")
    if phase == "grounding" and ev.get("tables"):
        parts = [f"Schema: {', '.join(ev['tables'])}"]
        if ev.get("terms"):
            parts.append(f"business terms: {', '.join(ev['terms'])}")
        if ev.get("joins"):
            parts.append(f"joins: {'; '.join(ev['joins'])}")
        if ev.get("examples"):
            parts.append("similar solved examples: " + "; ".join(f"\u201c{x}\u201d" for x in ev["examples"][:3]))
        return "\n".join(parts[:1] + [f"  {p}" for p in parts[1:]])
    if phase == "executing" and ev.get("sql"):
        attempt["n"] += 1
        return f"Query {attempt['n']}:\n```sql\n{ev['sql'].strip()}\n```"
    if phase == "error":
        return f"  \u21b3 query {max(attempt['n'], 1)} failed: {ev.get('detail', 'error')} — rewriting"
    if phase == "validated":
        return f"  \u21b3 plan check: {ev.get('detail', '')}"
    if phase == "result":
        cols = ev.get("columns") or []
        shown = ", ".join(cols[:8]) + (f", +{len(cols) - 8} more" if len(cols) > 8 else "")
        return (f"  \u21b3 {int(ev.get('rows') or 0):,} rows \u00d7 {len(cols)} columns ({shown})"
                + (f" in {ev['elapsed_ms']} ms" if ev.get("elapsed_ms") is not None else "")
                + (" — capped, more rows exist" if ev.get("truncated") else ""))
    if phase == "lookup":
        return f"Looked up {ev.get('detail', 'a term')} in the schema"
    return None


def _evidence(i: int, q: str, r: dict) -> str:
    if "error" in r:
        return f"### Evidence {i}: {q}\nERROR: {r['error']}"
    cols, rows = r.get("columns") or [], r.get("rows") or []
    lines = [f"### Evidence {i}: {q}", f"shape: {r.get('result_shape')} · {r.get('row_count', len(rows))} rows"
             + (" (CAPPED — more exist)" if r.get("truncated") else "")]
    if cols:
        lines.append("| " + " | ".join(map(str, cols)) + " |")
        lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows[:CFG.rows_to_composer]]
        if len(rows) > CFG.rows_to_composer:
            lines.append(f"(+{len(rows) - CFG.rows_to_composer} more rows in the table the user sees)")
    for key, label in (("result_note", "note"), ("unmet_parts", "not covered")):
        if r.get(key):
            lines.append(f"{label}: {r[key]}")
    return "\n".join(lines)


async def orchestrate(question: str, history: list[dict] | None = None, conversation_id: str = "",
                      route_model=None, compose_model=None):
    trace_id = uuid.uuid4().hex
    reasoning: list[str] = []
    usage_by_agent: dict[str, dict] = {}

    def reason(text: str) -> dict:
        reasoning.append(text if text.endswith("\n") else text + "\n")
        return {"type": "reasoning", "text": reasoning[-1]}

    # STEP 1 — route
    yield {"type": "status", "agent": "supervisor", "phase": "planning"}
    structured = (route_model or chat_model()).with_structured_output(RouteDecision, method="function_calling",
                                                                      include_raw=True)
    out = await structured.ainvoke([SystemMessage(content=ROUTE_PROMPT), *_history(history), HumanMessage(content=question)])
    decision: RouteDecision = out["parsed"]
    if decision is None:
        raise RuntimeError(f"routing failed: {out.get('parsing_error')}")
    usage_by_agent["supervisor"] = _usage(getattr(out.get("raw"), "usage_metadata", None))
    yield reason(f"Plan: {decision.rationale}")

    base = {"type": "final", "kind": "answer", "trace_id": trace_id, "clarifying_question": "", "options": [],
            "entities": [], "sources": [], "artifacts": {"table": None, "charts": [], "graphs": [], "exports": []}}

    def final(text: str, agents: list[str], extra: dict | None = None) -> dict:
        arts = base["artifacts"]
        return {**base, "text": text, "reasoning": "".join(reasoning), **(extra or {}),
                "artifact_summary": {"charts": len(arts["charts"]), "tables": 1 if arts["table"] else 0,
                                     "graphs": 0, "exports": 0},
                "agents_used": agents,
                "usage_by_agent": [{"agent": a, "usage": u} for a, u in usage_by_agent.items()],
                "plan_record": {"action": decision.action, "questions": decision.questions, "rationale": decision.rationale}}

    if decision.action == "clarify":
        yield final(decision.clarifying_question, ["supervisor"],
                    {"kind": "clarification", "clarifying_question": decision.clarifying_question,
                     "options": decision.options})
        return
    if decision.action == "answer" or not decision.questions:
        yield {"type": "token", "text": decision.reply}
        yield final(decision.reply, ["supervisor"])
        return

    # STEP 2 — NLQ, one call per sub-question; every step it takes becomes a reasoning line
    results: list[tuple[str, dict]] = []
    for q in decision.questions[:CFG.max_nlq_calls]:
        yield {"type": "status", "agent": "nlq", "phase": "invoking", "detail": q}
        yield reason(f"Asked the data agent: \u201c{q}\u201d")
        events: list[dict] = []
        attempt = {"n": 0}
        t0 = time.monotonic()
        try:
            fut = asyncio.ensure_future(
                specialists.call("nlq", {"question": q, "history": []}, events.append, conversation_id))
            while not fut.done() or events:               # relay while it runs
                while events:
                    ev = events.pop(0)
                    yield {"type": "status", "agent": "nlq", "phase": ev.get("phase", "working"),
                           **({"detail": ev["detail"]} if ev.get("detail") else {})}
                    line = _nlq_line(ev, attempt)
                    if line:
                        yield reason(line)
                if not fut.done():
                    await asyncio.sleep(0.05)
            r = fut.result()
            usage_by_agent["nlq"] = _add(usage_by_agent.get("nlq", {}), (r.get("usage") or {}))
            took = f"{time.monotonic() - t0:.1f} s"
            if r.get("result_shape") == "unanswerable":
                yield reason(f"The data agent could not answer from this data ({took}): {r.get('result_note', '')}")
            else:
                yield reason(f"Data agent done in {took}" + (f" — note: {r['result_note']}" if r.get("result_note") else "")
                             + (f" — not covered: {'; '.join(r['unmet_parts'])}" if r.get("unmet_parts") else ""))
        except specialists.SpecialistError as exc:
            r = {"error": str(exc)}
            yield reason(f"The data agent failed after {time.monotonic() - t0:.1f} s: {exc}")
        results.append((q, r))

    tables = [r for _, r in results if r.get("rows")]
    if tables:
        main = tables[0]
        base["artifacts"]["table"] = {"columns": main["columns"], "rows": main["rows"],
                                      "row_count": main.get("row_count", len(main["rows"])),
                                      "truncated": bool(main.get("truncated")), "sql": main.get("sql", "")}
    base["entities"] = _entities([e for _, r in results for e in (r.get("entities") or [])])
    base["sources"] = [src for q, r in results if (src := _source(q, r))]

    # STEP 3 — chart the main table when it has something to show
    agents = ["supervisor", "nlq"]
    if tables and len(tables[0]["rows"]) >= CFG.chart_min_rows and len(tables[0]["columns"]) >= 2:
        yield {"type": "status", "agent": "chart_gen", "phase": "charting"}
        t0 = time.monotonic()
        try:
            chart = await specialists.call("chart_gen", {"question": question, "columns": tables[0]["columns"],
                                                         "rows": tables[0]["rows"], "note": tables[0].get("result_note", "")},
                                           lambda ev: None, conversation_id)
            usage_by_agent["chart_gen"] = chart.pop("usage", {}) or {}
            agents.append("chart_gen")
            took = f"{time.monotonic() - t0:.1f} s"
            if chart.get("figures"):
                base["artifacts"]["charts"].append(chart)
                n = len(chart["figures"])
                yield reason(f"Chart: {chart['chart_type']}, {n} figure{'s' if n > 1 else ''} ({took}) — {chart.get('insight', '')}")
            elif chart.get("error"):
                yield reason(f"No chart ({took}): {chart['error']}")
            else:
                yield reason(f"No chart: the chart agent judged that a chart adds nothing here ({took}) — "
                             f"{chart.get('insight', '')}".rstrip(" —"))
        except specialists.SpecialistError as exc:
            yield reason(f"No chart: {exc}")
    elif tables:
        yield reason(f"No chart: {len(tables[0]['rows'])} row(s) read better as text")

    # STEP 4 — compose, streamed
    yield {"type": "status", "agent": "supervisor", "phase": "composing"}
    rows = sum(len(r.get("rows") or []) for _, r in results)
    failed = sum("error" in r for _, r in results)
    yield reason(f"Writing the answer from {rows:,} row{'s' if rows != 1 else ''} of evidence"
                 + (f" ({failed} part{'s' if failed > 1 else ''} failed — said so in the answer)" if failed else "")
                 + "; numbers are copied from the rows, not recomputed.")
    evidence = "\n\n".join(_evidence(i + 1, q, r) for i, (q, r) in enumerate(results))
    parts: list[str] = []
    comp_usage: dict = {}
    async for chunk in (compose_model or chat_model(streaming=True)).astream(
            [SystemMessage(content=COMPOSE_PROMPT), *_history(history),
             HumanMessage(content=f"Question: {question}\n\n# EVIDENCE\n{evidence}")]):
        text = chunk.text() if callable(getattr(chunk, "text", None)) else str(chunk.content or "")
        if text:
            parts.append(text)
            yield {"type": "token", "text": text}
        if getattr(chunk, "usage_metadata", None):
            comp_usage = _add(comp_usage, _usage(chunk.usage_metadata))
    usage_by_agent["supervisor"] = _add(usage_by_agent["supervisor"], comp_usage)

    # STEP 5 — final
    yield final("".join(parts), agents)
