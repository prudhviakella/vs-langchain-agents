"""chart_gen's output contract — enforced by the schema, not asked for in the prompt.

    supervisor ──table (columns + rows) + question──► model ──► ChartDecision
                                                                 chart_type · insight · figures[0..3]
                                                                 each figure = Plotly {data, layout}

FROM ACT
    - NO table field: the supervisor already holds the real rows and attaches
      them itself. Re-typing rows through the model doubled output tokens and
      risked numbers changing silently.
    - Zero figures is a real answer (chart_type="none"): an unneeded chart is not
      neutral — readers take a chart as a claim there is something to SEE.
    - chart_type and figures must agree (validator), so a consumer can branch on either.

WHAT THIS DOES NOT DO
    It does not make the figures safe to render — sanitize.py does, in code,
    after the model returns.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .config import CFG

ChartType = Literal["none", "bar", "line", "pie", "scatter", "histogram", "heatmap", "treemap", "sankey",
                    "waterfall", "choropleth", "scattergeo", "box", "funnel", "sunburst"]


class PlotlyFigure(BaseModel):
    """One Plotly figure: traces (`data`) + `layout`, rendered by plotly.js as is."""
    data: list[dict[str, Any]] = Field(description=(
        "Plotly trace dicts, each with `type` and its data arrays (x, y, values, …).\n"
        "NEVER negate values to sort bars: it shows real amounts as negative. To sort a "
        "horizontal bar chart highest-first use POSITIVE values with "
        "layout.yaxis.categoryorder='total ascending'.\n"
        "BUBBLE — a third dimension on a scatter: marker {size: [...], sizemode: 'area'}. "
        "e.g. x=return rate, y=average rating, size=units sold: a high return rate on "
        "trivial volume is noise; the same rate on large volume is the finding."))
    layout: dict[str, Any] = Field(description=(
        "Plotly layout: title, axis titles, legend and margins.\n"
        "DUAL AXIS when two series share an x-axis but not a unit (order VOLUME against "
        "RETURN RATE): give the second trace yaxis='y2' and add yaxis2 {title, "
        "overlaying: 'y', side: 'right'} — on one axis the small-scale series is a flat line."))


class ChartDecision(BaseModel):
    """The complete, enforced response."""
    chart_type: ChartType = Field(description=(
        "From the data's shape AND the question.\n"
        "STANDARD: bar (rankings) · line (trend over time) · pie (share, max 7 slices) · "
        "scatter (two measures) · histogram (distribution) · heatmap (2-D matrix) · "
        "treemap (sizes in a hierarchy) · sankey (flow A->B->C) · waterfall (cumulative change).\n"
        "GEOGRAPHIC: choropleth for a metric by US state — needs 2-letter codes ('CA', 'TX'); "
        "with full names use bar. scattergeo for lat/lon points.\n"
        "PREFER WHEN THEY FIT: box — who stands out from their peers (return rate by "
        "product within a category; boxpoints='outliers'). funnel — drop-off through "
        "stages (placed -> shipped -> delivered). sunburst — a path down the category "
        "tree (labels/parents/values; the root's parent is '').\n"
        "none — when no figure would add anything (with an empty figures list)."))
    insight: str = Field(description=(
        "ONE business observation the chart reveals, max ~200 chars, finding first — "
        "no 'The data shows…'."))
    figures: list[PlotlyFigure] = Field(min_length=0, max_length=CFG.max_figures, description=(
        f"0-{CFG.max_figures} figures, each a SEPARATE plot; the first is the primary view.\n"
        "EMPTY when a chart adds nothing: one row, one number, two or three rows read "
        "faster as text, or no comparison, trend, distribution or share in the result.\n"
        "A 2nd/3rd figure only for a genuinely DIFFERENT view (a map PLUS a top-N bar) — "
        "never the same insight in another chart type.\n"
        "Several SERIES are ONE figure with several traces (five brands over four quarters "
        "= one line chart, five traces) — splitting them destroys the comparison."))

    @model_validator(mode="after")
    def _chart_type_matches_figures(self):
        if not self.figures and self.chart_type != "none":
            raise ValueError(f"chart_type={self.chart_type!r} with no figures: use chart_type='none'")
        if self.figures and self.chart_type == "none":
            raise ValueError("chart_type='none' with figures present: name the chart type")
        return self
