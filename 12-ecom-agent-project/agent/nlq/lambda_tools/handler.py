"""The NLQ SQL Lambda — AgentCore Gateway target, in the VPC beside RDS.

    Gateway (SigV4) ──► lambda_handler(event=tool args, context.client_context.custom
                                       ["bedrockAgentCoreToolName"] = "nlq-sql-tools___execute_sql")
                          ├─ execute_sql(sql, row_cap, timeout_s) ─► sql_exec ─► RDS as ecom_reader
                          └─ explain_sql(sql)

ERRORS ARE DATA, NOT EXCEPTIONS
    A raised exception reaches the agent as an opaque Gateway failure. Returned
    as {"error", "kind"} it can be acted on:
        guard / sql  -> shown to the model, which rewrites the query
        transient    -> retried by the agent; never blamed on the query

WHAT THIS DOES NOT DO
    It trusts nothing from the agent: sql_exec guards the statement again and
    runs it as a read-only user in a READ ONLY transaction.
"""
from __future__ import annotations

import logging

import sql_exec
from guard import GuardError

log = logging.getLogger()
log.setLevel(logging.INFO)


def _tool_name(context) -> str:
    custom = getattr(getattr(context, "client_context", None), "custom", None) or {}
    return str(custom.get("bedrockAgentCoreToolName", "")).split("___")[-1]


def lambda_handler(event, context):
    tool, args = _tool_name(context), dict(event or {})
    try:
        if tool == "execute_sql":
            result = sql_exec.execute_sql(str(args.get("sql", "")), int(args.get("row_cap") or 1000),
                                          int(args.get("timeout_s") or 20))
            log.info("execute_sql rows=%s truncated=%s ms=%s", result["row_count"], result["truncated"],
                     result["elapsed_ms"])
            return result
        if tool == "explain_sql":
            return sql_exec.explain_sql(str(args.get("sql", "")))
        return {"error": f"unknown tool {tool!r}", "kind": "sql"}
    except GuardError as exc:
        return {"error": str(exc), "kind": "guard"}
    except sql_exec.SqlError as exc:
        return {"error": str(exc), "kind": "sql"}
    except sql_exec.TransientError as exc:
        log.warning("transient: %s", exc)
        return {"error": str(exc), "kind": "transient"}
