"""trial_graph's result summarization — the only view of a query result
the model ever sees.

    execute_cypher payload (rows/nodes in full)
        │
        v
    summarize(captured, cfg)  ->  counts + shape + at most 3 sample rows
        │
        v
    the model

This file holds NO tool definitions. The three tools execute in
lambda_tools/handler.py and reach this agent over MCP; core.py's
CypherMiddleware calls summarize() when a result comes back. Keeping it
here, as a pure function over a dict, means it is testable without any
MCP, Gateway or LangGraph machinery.

EVERY GUARD BELOW EXISTS BECAUSE OF A REAL FAILURE

    identifier truncation   a 64-char id shortened to 40 with no marker
                            was copied into eight follow-up queries, all
                            matched zero rows, and was read as "no
                            connections exist". Id-shaped columns are
                            never silently cut.
    column cap mismatch     the sample showed fewer columns than the
                            "columns:" line listed, so a real, populated
                            column looked absent. Both use the same cap.
    0 relationships         a node list is not a network. Said plainly,
                            or the model invents a cause for the missing
                            edges.
    untrusted fencing       values come from protocol PDFs written by
                            sponsors, not by this project. Newlines are
                            flattened and fence tags stripped so a value
                            cannot close the fence it sits inside.
"""
from __future__ import annotations

_SAMPLE_ROWS = 3
_COL_CAP = 12
_ID_CHARS, _VALUE_CHARS = 200, 40


def _clean(value) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.replace("<untrusted_data>", "").replace("</untrusted_data>", "")
    return " ".join(text.split())


def _is_id(column: str) -> bool:
    c = column.lower()
    return c.endswith("id") or c in ("nctid", "docid", "chunkid", "sectionkey", "key")


def summarize(captured: dict, cfg) -> str:
    shape = captured.get("result_shape", "empty")
    if shape == "graph":
        return _graph(captured, cfg)
    if shape == "table":
        return _table(captured, cfg)
    return ("empty: the query ran and matched nothing. This may well be the "
            "correct answer — do not invent data, and do not loosen the "
            "filters unless the question allows it.")


def _graph(captured: dict, cfg) -> str:
    nodes = captured.get("nodes", [])
    rels = captured.get("relationships", [])
    out = [f"graph: {len(nodes)} nodes, {len(rels)} relationships"]
    labels = sorted({l for n in nodes for l in n.get("labels", [])})
    if labels:
        out.append(f"labels: {', '.join(labels[:8])}")
    if nodes and not rels:
        out.append(
            "NOTE: 0 relationships. This result establishes NO connections "
            "between these nodes — it is a node list, not a network. The usual "
            "cause is RETURNing bare node variables; return the path or the "
            "relationship variable instead. Do not describe these nodes as "
            "connected, and do not blame a cap that did not fire.")
    if captured.get("truncated"):
        out.append(f"NOTE: truncated at the {cfg.graph_node_cap}-node cap.")
    return "\n".join(out)


def _table(captured: dict, cfg) -> str:
    rows = captured.get("rows", [])
    columns = [str(c) for c in captured.get("columns", [])]
    out = [f"table: {len(rows)} rows x {len(columns)} columns",
           f"columns: {', '.join(columns[:_COL_CAP])}"]

    hidden = columns[_COL_CAP:]
    if rows:
        out.append("<untrusted_data>")
    for row in rows[:_SAMPLE_ROWS]:
        cells = []
        for column, value in list(zip(columns, row))[:_COL_CAP]:
            text = _clean(value)
            cap = _ID_CHARS if _is_id(column) else _VALUE_CHARS
            if len(text) > cap:
                text = (f"{text[:cap]}...[TRUNCATED, {len(_clean(value))} chars — "
                        "do not reuse this value; re-query for it]")
            cells.append(f"{column}={text}")
        if hidden:
            cells.append(f"[+{len(hidden)} more columns not shown]")
        out.append("  " + ", ".join(cells))
    if rows:
        out.append("</untrusted_data>")

    if hidden:
        out.append(f"NOTE: samples show the first {_COL_CAP} of {len(columns)} "
                   f"columns. NOT SHOWN: {', '.join(hidden)}. These WERE returned "
                   "and ARE captured — their absence here is a display limit.")
    if len(rows) > _SAMPLE_ROWS:
        out.append(f"... and {len(rows) - _SAMPLE_ROWS} more rows (captured, not shown). "
                   "Do not retype rows into your answer.")
    if captured.get("truncated"):
        out.append(f"NOTE: truncated at the {cfg.row_cap}-row cap.")
    return "\n".join(out)
