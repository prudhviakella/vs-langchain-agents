"""trial_graph's result-summarization logic.

This file used to also define find_entity_by_name, validate_cypher, and
execute_cypher as LangChain @tool functions, calling Neo4j in-process.
That design was superseded by a deliberate redirect: tools now execute
in lambda_tools/handler.py, reached through the AgentCore Gateway over
SigV4-signed MCP (see core.py's connect_tools()). This file kept the
dead @tool functions and their now-broken imports (a neo4j_io module
that was planned but never built once the redirect happened) until a
real test run caught it — core.py's ResultSummaryMiddleware imports
_summarize from here, and importing this whole module was failing on
an unrelated dead import at the top. Rewritten clean rather than
patched, keeping only what's still real: _summarize itself.

_summarize lives here, not inline in core.py, so it can be unit tested
without any MCP/Gateway/LangGraph machinery in the loop — it is pure
data-shaping logic over a plain dict, nothing about how that dict
arrived.
"""
from __future__ import annotations


def _summarize(captured: dict, warnings: list[str] | None) -> str:
    """The COMPACT view — the only thing the model ever sees of a query
    result. Enough to reason about (did I get anything? is the shape
    right?), never enough to retype.

    Every guard below is ported from a real NLC production trace, not a
    hypothetical — see ACT Xerebro's own nlc/tools.py for the exact
    incidents each one fixed. Applied here preemptively rather than
    waiting for this graph to reproduce them independently.
    """
    shape = captured.get("result_shape", "empty")
    parts: list[str] = []

    if shape == "graph":
        nodes = captured.get("nodes", [])
        rels = captured.get("relationships", [])
        parts.append(f"graph: {len(nodes)} nodes, {len(rels)} relationships")
        labels = sorted({label for node in nodes for label in node.get("labels", [])})
        if labels:
            parts.append(f"labels: {', '.join(labels[:8])}")
        # A graph with nodes but no relationships is almost never a
        # truncation artifact — it is a RETURN clause that projected bare
        # node variables. Say so explicitly, or the model invents a cause.
        if nodes and not rels:
            parts.append(
                "NOTE: 0 relationships. This result establishes NO "
                "connections between the nodes — it is a node list, not a "
                "network. The usual cause is a RETURN of bare node "
                "variables; return the PATH variable (or the relationship "
                "variable) instead. Do NOT describe these nodes as "
                "connected, and do not attribute the missing edges to a "
                "cap unless the cap actually fired.")

    elif shape == "table":
        rows = captured.get("rows", [])
        columns = captured.get("columns", [])
        parts.append(f"table: {len(rows)} rows x {len(columns)} columns")
        parts.append(f"columns: {', '.join(str(c) for c in columns[:12])}")

        # Identifier-shaped columns must never be silently truncated —
        # their whole purpose is to be copied verbatim into a follow-up
        # filter. A production trace elsewhere showed a 64-char id cut to
        # 40 with no marker, copied into 8 follow-up queries, all matching
        # zero, read as "no connections exist."
        def _is_id_like(col: str) -> bool:
            c = col.lower()
            return c in ("nctid", "docid", "chunkid", "sectionkey") or c.endswith("id")

        def _clean_value(value) -> str:
            """Neutralize one cell before it reaches the model. Adapted
            from the reference Supervisor's own _clean() — these values
            are PDF/registry text authored by whoever submitted the
            protocol, not by this project, so a sponsor's free-text
            field is exactly the kind of place a stray newline or a
            fence-breaking attempt at <untrusted_data> could show up,
            deliberate or not.
            """
            text = str(value).replace("\n", " ").replace("\r", " ")
            text = text.replace("<untrusted_data>", "").replace("</untrusted_data>", "")
            return " ".join(text.split())

        # The sample's column cap MUST match the columns: line above (12),
        # not some smaller number — a mismatch here is exactly what let a
        # real, populated column disappear from the sample while still
        # being reported as present in the summary line, which the model
        # read as "this data doesn't exist."
        _SAMPLE_COL_CAP = 12
        dropped = [str(c) for c in columns[_SAMPLE_COL_CAP:]]
        if rows:
            parts.append("<untrusted_data>")
        for row in rows[:3]:                       # 3 rows, not 300
            cells = []
            for col, val in list(zip(columns, row))[:_SAMPLE_COL_CAP]:
                s = _clean_value(val)
                cap = 200 if _is_id_like(col) else 40
                if len(s) > cap:
                    s = f"{s[:cap]}...[TRUNCATED, {len(str(val))} chars total — " \
                        "do not reuse this value verbatim; re-query for it]"
                cells.append(f"{col}={s}")
            if dropped:
                cells.append(f"[+{len(dropped)} more columns not shown]")
            parts.append(f"  sample: {', '.join(cells)}")
        if rows:
            parts.append("</untrusted_data>")
        if dropped:
            parts.append(
                f"  NOTE: sample rows show the first {_SAMPLE_COL_CAP} of "
                f"{len(columns)} columns. NOT SHOWN: {', '.join(dropped)}. "
                "These columns WERE returned and ARE captured — their "
                "absence here is a display limit, not missing data.")
        if len(rows) > 3:
            parts.append(f"  ... and {len(rows) - 3} more rows (captured, not shown)")

    else:
        parts.append("empty: the query ran successfully but matched nothing. "
                     "This may well be the correct answer — do not invent "
                     "data, and do not loosen the filters unless the "
                     "question allows it.")

    if warnings:
        parts.append("warnings: " + "; ".join(warnings[:3]))
    return "\n".join(parts)
