"""NLQ tools — each does ONE thing and holds NO policy (middleware enforces policy).

    lookup_schema(term)  RETRIEVAL  find a table/column the grounding did not show
    validate_sql(sql)    CHECK      EXPLAIN without running
    execute_sql(sql)     ACTION     run it; rows -> STATE, a compact summary -> the model

FROM ACT
    execute_sql returns a Command: the rows go into state (`captured`) and the
    model sees only _summarize(): counts, columns, 3 sample rows. Every
    truncation in that summary is MARKED — a hidden column or a cut ID read as
    "missing data" once cost ACT a whole turn of wrong follow-up queries.

WHAT THIS DOES NOT DO
    No guard here: GuardMiddleware already replaced `sql` with the guarded
    statement before these run.
"""
from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

from . import db, grounding
from . import progress as P
from .config import CFG


def _is_id_like(col: str) -> bool:
    c = str(col).lower()
    return c == "id" or c.endswith("_id") or c in ("sku", "promo_code", "tracking_number", "email")


def summarize(captured: dict) -> str:
    """The ONLY view of a result the model ever gets."""
    rows, cols = captured.get("rows") or [], captured.get("columns") or []
    cap = 12
    parts = [f"result: {len(rows)} rows x {len(cols)} columns",
             "columns: " + ", ".join(cols[:cap]) +
             (f" [+{len(cols) - cap} more: {', '.join(cols[cap:])}]" if len(cols) > cap else "")]
    for row in rows[:3]:
        cells = []
        for col, val in list(zip(cols, row))[:cap]:
            s, limit = str(val), (120 if _is_id_like(col) else 40)
            if len(s) > limit:
                s = f"{s[:limit]}…[TRUNCATED, {len(str(val))} chars — do not reuse verbatim; re-query]"
            cells.append(f"{col}={s}")
        if len(cols) > cap:
            cells.append(f"[+{len(cols) - cap} columns not shown — they WERE returned]")
        parts.append("  sample: " + ", ".join(cells))
    if len(rows) > 3:
        parts.append(f"  … {len(rows) - 3} more rows captured, not shown. Do NOT retype rows into your answer.")
    if not rows:
        parts.append("empty: the query ran and matched nothing. That may be the right answer — do not "
                     "invent data, and do not loosen filters unless the question allows it.")
    if captured.get("truncated"):
        parts.append(f"NOTE: stopped at the {CFG.row_cap}-row cap — more rows exist. Aggregate, or say the "
                     "result is a sample.")
    for w in captured.get("warnings") or []:
        parts.append(f"note: {w}")
    return "\n".join(parts)


@tool
async def lookup_schema(term: str) -> str:
    """Search the schema for a table or column by name or meaning, when the SCHEMA
    section does not show what the question needs. Returns matching tables/columns."""
    P.emit(P.LOOKUP, detail=f"looking up '{term}'")
    g = grounding.retrieve(term)
    return g.context_block[:6000] or "nothing found"


@tool
async def validate_sql(sql: str) -> str:
    """Check a query WITHOUT running it (EXPLAIN). Returns VALID with a row estimate,
    or the error. Use before an expensive or uncertain query."""
    P.emit(P.VALIDATING, sql=sql, detail="checking the query plan")
    try:
        result = await db.explain(sql)
    except db.GuardRejected as exc:
        return f"GUARD: {exc}"
    return ("VALID: " if result.get("ok") else "INVALID: ") + str(result.get("detail"))


@tool
async def execute_sql(sql: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Run ONE read-only SELECT against the ecom schema and capture the result.
    You receive a compact summary; the full rows go to the answer automatically —
    never retype them."""
    P.emit(P.EXECUTING, sql=sql, detail="running the query")
    try:
        captured = await db.execute(sql)
    except (db.SqlError, db.GuardRejected) as exc:
        return Command(update={"messages": [ToolMessage(
            content=f"SQL_ERROR: {exc}\nRewrite the query.", tool_call_id=tool_call_id)]})
    P.emit(P.RESULT, detail=f"{captured['row_count']:,} rows x {len(captured['columns'])} columns"
           + (" (capped)" if captured.get("truncated") else ""), elapsed_ms=captured.get("elapsed_ms"))
    return Command(update={
        "executed_sql": captured.get("sql") or sql,
        "captured": captured,
        "messages": [ToolMessage(content=summarize(captured), tool_call_id=tool_call_id)],
    })


TOOLS = [lookup_schema, validate_sql, execute_sql]
