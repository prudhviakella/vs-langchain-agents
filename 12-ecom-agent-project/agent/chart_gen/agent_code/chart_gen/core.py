"""chart_gen core — one table in, Plotly figures out. One structured model call, no tools, no loop.

    {question, columns, rows} ─► table_text (≤ max_rows_to_model rows, the cap stated)
                              ─► gpt-6-sol, structured output = ChartDecision
                              ─► sanitize_figures (in CODE: unknown trace types dropped,
                                 script/handler/URL vectors scrubbed, too-thin figures dropped)
                              ─► {chart_type, insight, figures, usage}  or  {"error", usage}

FROM ACT
    - function_calling structured output: Plotly specs are open dicts, which a
      strict JSON-schema mode rejects.
    - Never raises past the caller: a lost chart must not lose the answer.

WHAT THIS DOES NOT DO
    It never returns the table: the supervisor attaches the real rows itself.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .config import CFG, api_key
from .prompt import BUNDLED_SYSTEM_PROMPT
from .sanitize import sanitize_figures
from .schemas import ChartDecision

log = logging.getLogger("chart_gen.core")


def chart_model():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CFG.model, use_responses_api=True, api_key=api_key(), max_retries=4)


def table_text(question: str, columns: list[str], rows: list[list], note: str = "") -> str:
    """The table the model reads. More rows than the cap are cut, and the cut is SAID."""
    shown = rows[:CFG.max_rows_to_model]
    lines = [f"Question: {question}", "", "| " + " | ".join(map(str, columns)) + " |",
             "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in shown]
    if len(rows) > len(shown):
        lines.append(f"\n(first {len(shown)} of {len(rows)} rows shown — chart these; say the chart is a top slice)")
    if note:
        lines.append(f"\nNote from the data agent: {note}")
    return "\n".join(lines)


def _usage(raw) -> dict:
    meta = getattr(raw, "usage_metadata", None) or {}
    return {k: int(meta.get(k) or 0) for k in ("input_tokens", "output_tokens", "total_tokens")}


async def generate_chart(question: str, columns: list[str], rows: list[list], note: str = "", model=None) -> dict:
    structured = (model or chart_model()).with_structured_output(ChartDecision, method="function_calling",
                                                                 include_raw=True)
    messages = [SystemMessage(content=BUNDLED_SYSTEM_PROMPT), HumanMessage(content=table_text(question, columns, rows, note))]
    try:
        out = await structured.ainvoke(messages)
        decision, usage = out.get("parsed"), _usage(out.get("raw"))
        if decision is None:
            err = str(out.get("parsing_error") or "no parseable ChartDecision")
            log.error("[chart_gen] structured output failed: %s", err)
            return {"error": f"structured output failed: {err}", "usage": usage}
        result = decision.model_dump()
        result["figures"] = sanitize_figures(result.get("figures") or [])
        if not result["figures"]:
            result["chart_type"] = "none"                    # every figure dropped -> say so consistently
        if len(result.get("insight") or "") > CFG.max_insight_chars:
            result["insight"] = result["insight"][:CFG.max_insight_chars].rstrip() + "…"
        log.info("[chart_gen] %s, %d figure(s)", result["chart_type"], len(result["figures"]))
        return {**result, "usage": usage}
    except Exception as exc:
        log.exception("[chart_gen] generation failed")
        return {"error": f"{type(exc).__name__}: {exc}", "usage": {}}


def parse_request(text: str) -> dict:
    body = json.loads(text)
    return {"question": str(body.get("question", "")), "columns": list(body.get("columns") or []),
            "rows": list(body.get("rows") or []), "note": str(body.get("note") or "")}
