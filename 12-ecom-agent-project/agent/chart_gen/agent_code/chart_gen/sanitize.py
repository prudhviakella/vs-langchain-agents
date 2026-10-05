"""Defense-in-depth sanitization of the model-authored Plotly spec.

WHY THIS EXISTS
---------------
`PlotlyFigure.data` and `.layout` are `dict[str, Any]` — entirely authored by
the model. On the web, the backend renders that spec with `Plotly.newPlot`
(the web app renders it with Plotly.newPlot), and Plotly renders a limited set of HTML in text
fields (titles, axis titles, trace `text`, hovertemplate, annotations). That
makes a model-authored string like `<img src=x onerror=alert(1)>` a potential
DOM-injection vector the moment it reaches the browser renderer.

The data the model charts includes text other people wrote: product names,
review titles, customer names. Whoever writes a review controls its title. So
"the model would never write that" is not a guarantee — the model faithfully
echoing a user-written string INTO a chart label is the realistic path, not
model misbehavior.

We fix it at the SOURCE (here, in chart_gen) rather than in the frontend, so
the guarantee holds for EVERY consumer of the spec — the web renderer AND the
server-side PNG/Word export — and does not depend on each consumer remembering
to sanitize. Defense in depth: even though the PNG path happens to be safe
today (matplotlib reads only recognized numeric/string keys), a spec that is
clean at the source protects paths that do not exist yet.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does NOT strip all HTML. Plotly LEGITIMATELY supports `<br>`, `<b>`, `<i>`,
`<sub>`, `<sup>`, and `<span style=...>` in text fields, and analysts rely on
them for readable multi-line titles. A blanket strip would break real charts.
The sanitizer is surgical: it neutralizes the DANGEROUS constructs (script/
iframe/object/embed tags, event-handler attributes, javascript:/vbscript:/
data: URLs) and leaves benign formatting intact.
"""
from __future__ import annotations

import re
import logging
from typing import Any

# Plotly trace types this agent is designed to emit. A trace whose `type` is
# not here is either a mistake or an attempt to reach an unintended renderer
# path; we drop it rather than forward an unrecognized type to Plotly. `line`
# is included because although Plotly has no `line` trace (it is scatter +
# mode='lines'), the schema/prompt name it and the PNG path maps it — keeping
# it here avoids dropping a legitimate figure over a naming nuance.
log = logging.getLogger("chart_gen.sanitize")

ALLOWED_TRACE_TYPES = frozenset({
    "bar", "line", "pie", "scatter", "histogram", "heatmap", "treemap",
    "sankey", "waterfall", "choropleth", "scattergeo", "box", "funnel",
    "sunburst", "scatterpolar", "scattermapbox",
})

# Dangerous HTML elements — never legitimate in a Plotly text label.
_DANGEROUS_TAG = re.compile(
    r"<\s*/?\s*(script|iframe|object|embed|link|meta|base|form|svg|math)\b[^>]*>",
    re.IGNORECASE,
)
# Any leftover tag carrying an on*= event handler (onerror, onclick, onload...).
_EVENT_HANDLER_TAG = re.compile(r"<[^>]*\bon\w+\s*=[^>]*>", re.IGNORECASE)
# Dangerous URL schemes anywhere in a value (href/src or bare).
_DANGEROUS_URL = re.compile(r"(?:javascript|vbscript|data)\s*:", re.IGNORECASE)


def _scrub_str(s: str) -> str:
    """Neutralize XSS vectors in one string while preserving benign Plotly
    formatting tags. Order matters: remove whole dangerous elements first, then
    any tag still carrying an event handler, then dangerous URL schemes."""
    s = _DANGEROUS_TAG.sub("", s)
    s = _EVENT_HANDLER_TAG.sub("", s)
    s = _DANGEROUS_URL.sub("", s)
    return s


def _scrub(value: Any) -> Any:
    """Recursively scrub every string in an arbitrarily nested dict/list."""
    if isinstance(value, str):
        return _scrub_str(value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _plotted_points(trace: dict) -> int:
    """How many points this trace actually draws.

    Plotly carries the values under different keys by trace type -- x/y for
    scatter and bar, `values` for pie, `z` for heatmaps -- so the longest of
    them is the honest count.
    """
    best = 0
    for key in ("x", "y", "values", "z", "labels"):
        v = trace.get(key)
        if isinstance(v, (list, tuple)):
            best = max(best, len(v))
    return best


def _is_too_thin(fig: dict) -> bool:
    """A figure with too few points to be worth drawing.

    WHY THIS IS ENFORCED IN CODE. The prompt already says a two-point line is
    not a trend, and a real turn produced one anyway: a 45-row result over
    2022-2026 yielded a third figure titled "Flagged Transactions: Count vs
    Amount (2025-2026)" -- two years, two series, dual axes, one line rising
    and one falling. It reads as a diverging trend and it is four numbers.

    A prompt rule is a request. This is the same reasoning as MODE_BOUNDARY
    being enforced in code rather than asked for.

    WHAT THIS DOES NOT DO: it does not drop short BAR traces. Two bars side by
    side are a legitimate comparison -- the reader sees two quantities, not a
    direction. It is the LINE between two points that invents a trend, so only
    line-shaped traces are held to the 3-point floor.
    """
    traces = [t for t in (fig.get("data") or []) if isinstance(t, dict)]
    if not traces:
        return True

    for tr in traces:
        ttype = str(tr.get("type") or "scatter").lower()
        pts = _plotted_points(tr)
        if pts == 0:
            continue
        # A single point is never worth a figure, whatever the type.
        if pts == 1:
            return True
        # Lines imply continuity between points; two points imply a trend that
        # two numbers cannot support. Bars, pies and heatmaps do not.
        line_like = ttype in ("scatter", "scattergl", "line") and \
            "lines" in str(tr.get("mode") or "lines").lower()
        if line_like and pts < 3:
            return True
    return False


def sanitize_figures(figures: list[dict]) -> list[dict]:
    """Return a sanitized copy of the figure list.

    - Drops any trace whose `type` is not in ALLOWED_TRACE_TYPES (a trace with
      no `type` defaults to scatter in Plotly, which is allowed, so we keep it).
    - Scrubs XSS vectors from every string in every trace and in the layout,
      recursively, preserving benign formatting.
    - Never raises: a malformed figure yields an empty/So-cleaned figure rather
      than taking down the whole response (chart generation is best-effort; a
      lost chart must not lose the analyst's textual answer).
    """
    clean: list[dict] = []
    for fig in figures or []:
        if not isinstance(fig, dict):
            continue
        data = fig.get("data") or []
        kept_traces = []
        for tr in data:
            if not isinstance(tr, dict):
                continue
            ttype = tr.get("type")
            # `type` absent -> Plotly treats as scatter (allowed). Present ->
            # must be recognized. Compare case-insensitively and as a string.
            if ttype is not None and str(ttype).lower() not in ALLOWED_TRACE_TYPES:
                continue
            kept_traces.append(_scrub(tr))
        candidate = {
            **_scrub({k: v for k, v in fig.items() if k not in ("data",)}),
            "data": kept_traces,
        }
        # Drop a figure with too little to plot. Logged rather than silent:
        # a chart vanishing with no explanation looks like a failure, and
        # whoever tunes the prompt needs to see it happening.
        if _is_too_thin(candidate):
            title = ((candidate.get("layout") or {}).get("title") or {})
            name = title.get("text") if isinstance(title, dict) else title
            log.info("[sanitize] figure dropped, too few points to plot: %r", name)
            continue
        clean.append(candidate)
    return clean
