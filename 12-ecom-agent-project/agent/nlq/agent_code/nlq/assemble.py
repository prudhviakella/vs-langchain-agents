"""Build the NLQResponse from the model's decision and what the TOOLS captured.

    ModelDecision (judgements) + state.captured (rows) + state.executed_sql ──► NLQResponse

Pure function, no framework imports — testable offline. The rows and SQL come
ONLY from state: a model that skipped execute_sql yields `not_executed`, never
a table it typed itself.
"""
from __future__ import annotations

from .schemas import ModelDecision, NLQResponse


def assemble(decision: ModelDecision, captured: dict | None, executed_sql: str,
             grounding_ids: list[str], repair_count: int) -> NLQResponse:
    common = dict(entities=decision.entities, unmet_parts=decision.unmet_parts,
                  grounding_record_ids=grounding_ids, repair_count=repair_count, sql=executed_sql or "")
    if not decision.answerable:
        return NLQResponse(result_shape="unanswerable", result_note=decision.note, **common)
    if not captured:
        return NLQResponse(result_shape="not_executed",
                           result_note=decision.note or "no query was executed", **common)
    rows = captured.get("rows") or []
    note = decision.note
    if captured.get("truncated"):
        note = (note + " " if note else "") + f"Result capped at {len(rows)} rows; more rows exist."
    return NLQResponse(result_shape="table" if rows else "empty", columns=captured.get("columns") or [],
                       rows=rows, row_count=len(rows), truncated=bool(captured.get("truncated")),
                       result_note=note.strip(), **common)
