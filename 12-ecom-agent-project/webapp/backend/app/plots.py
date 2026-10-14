"""The fallback chart — when the user ASKED for one and none came back.

    question says plot / chart / graph / visualise / trend   AND   a table exists   AND   no chart
        ─► label column = the first text-like column; value column = the first numeric one
        ─► line when the labels are dates or months (2024-03, 2024-03-01, Mar 2024), else bar
        ─► sanitize_figures (the same scrubbing as chart_gen) ─► one chart artifact

From ACT's backend (_plots.py): a user who says "plot it" should not get prose alone
because the chart agent judged a chart unnecessary.

WHAT THIS DOES NOT DO
    It never runs on a question that did not ask for a chart, never replaces a chart
    chart_gen produced, and plots at most 60 points (the first 60 rows).
"""
from __future__ import annotations

import re

from .sanitize import sanitize_figures

_WANTS = re.compile(r"\b(plot|chart|graph|visuali[sz]e|visual|trend\s*line|bar\s*chart|pie|histogram)\b", re.I)
_DATE = re.compile(r"^(\d{4}-\d{2}(-\d{2})?|\d{4}-\d{2}-\d{2}[ T][\d:]+|[A-Za-z]{3,9}\s+\d{4}|\d{4})$")
_MAX_POINTS = 60


def wants_plot(question: str) -> bool:
    return bool(_WANTS.search(question or ""))


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


def chart_from_table(table: dict, title: str = "") -> dict | None:
    cols, rows = table.get("columns") or [], (table.get("rows") or [])[:_MAX_POINTS]
    if len(cols) < 2 or len(rows) < 2:
        return None
    numeric = [i for i in range(len(cols)) if all(_num(r[i]) is not None for r in rows if r[i] is not None)]
    labels = [i for i in range(len(cols)) if i not in numeric]
    if not numeric:
        return None
    li = labels[0] if labels else 0
    vi = next((i for i in numeric if i != li), None)
    if vi is None:
        return None
    x = [str(r[li]) for r in rows]
    y = [_num(r[vi]) for r in rows]
    is_time = all(_DATE.match(v.split(" 00:00:00")[0]) for v in x)
    trace = ({"type": "scatter", "mode": "lines+markers", "x": x, "y": y, "name": cols[vi]} if is_time
             else {"type": "bar", "x": x, "y": y, "name": cols[vi]})
    fig = {"data": [trace], "layout": {"title": {"text": title or f"{cols[vi]} by {cols[li]}"},
                                       "xaxis": {"title": {"text": cols[li]}}, "yaxis": {"title": {"text": cols[vi]}}}}
    safe = sanitize_figures([fig])
    if not safe:
        return None
    return {"kind": "chart", "chart_type": "line" if is_time else "bar", "insight": "",
            "source": "fallback", "spec": safe[0]}
