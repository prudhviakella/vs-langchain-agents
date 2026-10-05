"""Grounding — the schema, joins, metrics and examples for ONE question, from Pinecone ecom-kb.

    question ─► embed once (text-embedding-3-small)
        ├─ nlq-schema   type=table   top k_tables            ─┐ matched tables =
        ├─ nlq-schema   type=column  top k_columns            ─┘ tables ∪ tables of top columns
        ├─ nlq-schema   type=column  table ∈ matched (ALL)    every column of every matched table
        ├─ nlq-schema   type=join    both sides ∈ matched
        ├─ nlq-examples              top k_examples           (verified only, if configured)
        └─ common       glossary top k_glossary + ALL capability rules
                            │
                            ▼
        Grounding.context_block  ──► appended to the system prompt by GroundingMiddleware

FROM ACT
    - Retrieved ONCE per question (before_agent), not per model call.
    - Every column of a matched table is included: a column missing from the
      prompt reads to the model as a column that does not exist.
    - Field values are normalised defensively (_s): one malformed record must
      not crash grounding — ACT found that bug three times, one at a time.
    - retrieve() and assemble() are split, so assemble is testable offline.

WHAT THIS DOES NOT DO
    It does not decide which table answers the question — the model does, from
    what is here. It does not cache across questions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from .config import CFG, api_key


@dataclass
class Grounding:
    context_block: str
    record_ids: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


def _s(v, cap: int = 0) -> str:
    """A field as one clean line: lists joined, None empty, newlines flattened."""
    if v is None:
        return ""
    if isinstance(v, list):
        v = "; ".join(str(x) for x in v)
    text = " ".join(str(v).split())
    return text[:cap].rstrip() + "…" if cap and len(text) > cap else text


def _section(markdown: str, title: str) -> str:
    m = re.search(rf"## {title}\n(.*?)(?:\n## |\Z)", markdown or "", re.S)
    return _s(m.group(1)) if m else ""


# ── retrieval ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _index():
    from pinecone import Pinecone
    return Pinecone(api_key=api_key("PINECONE_API_KEY", "PINECONE_SECRET_ID")).Index(CFG.pinecone_index)


def embed(text: str) -> list[float]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key("OPENAI_API_KEY", "OPENAI_SECRET_ID"), max_retries=4)
    return client.embeddings.create(model=CFG.embed_model, input=[text]).data[0].embedding


def _query(vec, namespace: str, top_k: int, flt: dict) -> list[dict]:
    res = _index().query(vector=vec, top_k=top_k, namespace=namespace, filter=flt, include_metadata=True)
    return [{"id": m.id, "score": m.score, **(m.metadata or {})} for m in res.matches]


def retrieve(question: str) -> Grounding:
    vec = embed(question)
    tables = _query(vec, "nlq-schema", CFG.k_tables, {"type": {"$eq": "table"}})
    top_cols = _query(vec, "nlq-schema", CFG.k_columns, {"type": {"$eq": "column"}})
    matched = list(dict.fromkeys([_s(t.get("table")) for t in tables] + [_s(c.get("table")) for c in top_cols]))
    matched = [t for t in matched if t]
    columns = _query(vec, "nlq-schema", CFG.max_columns,
                     {"type": {"$eq": "column"}, "table": {"$in": matched}}) if matched else []
    joins = _query(vec, "nlq-schema", 20, {"type": {"$eq": "join"}}) if matched else []
    ex_filter = {"type": {"$eq": "example"}}
    if CFG.examples_verified_only:
        ex_filter["verified"] = {"$eq": True}
    examples = _query(vec, "nlq-examples", CFG.k_examples, ex_filter)
    glossary = _query(vec, "common", CFG.k_glossary, {"type": {"$eq": "glossary"}})
    rules = _query(vec, "common", 10, {"type": {"$eq": "capability_card"}})
    # tables that were reached only through a column still need their own record
    known = {_s(t.get("table")) for t in tables}
    missing = [t for t in matched if t not in known]
    if missing:
        tables += _query(vec, "nlq-schema", len(missing), {"type": {"$eq": "table"}, "table": {"$in": missing}})
    return assemble(tables, columns, joins, examples, glossary, rules)


# ── assembly (pure: testable without Pinecone) ───────────────────────────
def assemble(tables, columns, joins, examples, glossary, rules) -> Grounding:
    ids: list[str] = []
    names = [_s(t.get("table")) for t in tables if _s(t.get("table"))]
    lines = ["# SCHEMA — PostgreSQL, schema `ecom`. Use ONLY these tables and columns."]

    by_table: dict[str, list[dict]] = {}
    for c in columns:
        by_table.setdefault(_s(c.get("table")), []).append(c)

    for t in tables:
        name = _s(t.get("table"))
        if not name:
            continue
        ids.append(t.get("id", name))
        desc = _section(t.get("description", ""), "BUSINESS DESCRIPTION") or _s(t.get("description"), 400)
        guidance = _section(t.get("description", ""), "QUERY GUIDANCE")
        lines.append(f"\nTABLE {name} [{_s(t.get('table_type'))}, {_s(t.get('row_count'))} rows] — {_s(desc, 500)}")
        if _s(t.get("grain")):
            lines.append(f"  grain: {_s(t.get('grain'))}")
        if guidance:
            lines.append(f"  guidance: {_s(guidance, 500)}")
        for c in by_table.get(name, []):
            ids.append(c.get("id", f"{name}.{c.get('column')}"))
            line = f"  .{_s(c.get('column'))} ({_s(c.get('data_type'))}) — {_s(c.get('description'), 260)}"
            if _s(c.get("where_to_use")):
                line += f"  Use: {_s(c.get('where_to_use'), 120)}"
            syn = c.get("analyst_synonyms")
            if syn:
                line += f"  aka: {_s(syn if isinstance(syn, str) else syn[:5], 100)}"
            enums = c.get("enum_values")
            vmap = c.get("value_map")
            if vmap:
                try:
                    pairs = json.loads(vmap) if isinstance(vmap, str) else vmap
                    line += "  codes: " + ", ".join(f"{k}={v}" for k, v in list(pairs.items())[:60])
                except (ValueError, AttributeError):
                    pass
            elif enums:
                line += f"  values: {_s(enums, 300)}"
            if c.get("is_date") and _s(c.get("date_format")):
                line += f"  format: {_s(c.get('date_format'))}"
            elif _s(c.get("sample_values")) and not enums:
                line += f"  e.g. {_s(c.get('sample_values'), 80)}"
            lines.append(line)

    wanted = set(names)
    rel = [j for j in joins if _s(j.get("left_table")) in wanted and _s(j.get("right_table")) in wanted]
    if rel:
        lines.append("\n# JOINS between these tables")
        for j in rel:
            ids.append(j.get("id", ""))
            lines.append(f"  {_s(j.get('left_table'))} JOIN {_s(j.get('right_table'))} ON {_s(j.get('join_on'))} "
                         f"({_s(j.get('cardinality'))}) — {_s(j.get('description'), 200)}")

    if glossary:
        lines.append("\n# BUSINESS TERMS — use these definitions exactly")
        for g in glossary:
            ids.append(g.get("id", ""))
            lines.append(f"  {_s(g.get('term'))} — {_s(g.get('meaning'), 300)}  [maps to: {_s(g.get('maps_to'), 150)}]")

    if rules:
        lines.append("\n# RULES FOR THIS DATA")
        for r in rules:
            ids.append(r.get("id", ""))
            lines.append(f"  {_s(r.get('description'), 1200)}")

    if examples:
        lines.append("\n# EXAMPLES — similar questions with SQL that ran correctly on this data")
        for e in examples:
            ids.append(e.get("id", ""))
            lines.append(f"  Q: {_s(e.get('question'))}\n  SQL:\n{e.get('query', '').strip()}\n")

    return Grounding(context_block="\n".join(lines), record_ids=[i for i in ids if i], tables=names)
