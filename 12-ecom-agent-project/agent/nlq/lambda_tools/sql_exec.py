"""Run one guarded, read-only query against PostgreSQL — the only code that touches the database.

    {sql} ─► guard (again) ─► connect as the read-only user ─► READ ONLY transaction
                              statement_timeout ─► fetch row_cap + 1 ─► {columns, rows, row_count, truncated}

Used in two places, one implementation:
    the Lambda behind the AgentCore Gateway (handler.py), in AWS, beside RDS
    the agent itself in LOCAL mode (no Gateway), for development

WHY THE GUARD RUNS AGAIN HERE
    The agent guards every query before sending it. This module does not trust
    that: it is the boundary the database sits behind, so it re-checks.

WHAT THIS DOES NOT DO
    - It does not decide whether a query answers the question — the model does.
    - It never returns more than row_cap rows; `truncated` says when it stopped.
    - It holds no connection between calls: one connection per query. At this
      volume that is simpler and safer than a pool that outlives a Lambda freeze.
"""
from __future__ import annotations

import datetime as dt
import decimal
import json
import os
import time
import uuid

import psycopg

from guard import GuardError, guard


class SqlError(Exception):
    """The query ran and failed (bad column, type mismatch, timeout). The text
    goes back to the model so it can rewrite the query."""


class TransientError(Exception):
    """Could not reach the database. Retried; never shown to the model as a bad query."""


def _credentials() -> dict:
    """PG* environment variables; the password from Secrets Manager when
    READER_SECRET_ID is set (the deployed Lambda), else PGPASSWORD (local)."""
    password = os.environ.get("PGPASSWORD", "")
    if os.environ.get("READER_SECRET_ID"):
        import boto3
        secret = json.loads(boto3.client("secretsmanager").get_secret_value(
            SecretId=os.environ["READER_SECRET_ID"])["SecretString"])
        password = secret["password"]
    return dict(host=os.environ["PGHOST"], port=int(os.environ.get("PGPORT", 5432)),
                dbname=os.environ.get("PGDATABASE", "ecom"), user=os.environ.get("PGUSER", "ecom_reader"),
                password=password, sslmode=os.environ.get("PGSSLMODE", "require"), connect_timeout=10)


def _cell(v):
    """A JSON-safe cell: exact Decimals become numbers, times become ISO text."""
    if isinstance(v, decimal.Decimal):
        return int(v) if v == v.to_integral_value() and abs(v) < 2**53 else float(v)
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat(sep=" ") if isinstance(v, dt.datetime) else v.isoformat()
    if isinstance(v, (bytes, memoryview)):
        return "<binary>"
    if isinstance(v, uuid.UUID):
        return str(v)
    return v


def _run(statement: str, row_cap: int, timeout_s: int) -> tuple[list[str], list[list], bool]:
    try:
        conn = psycopg.connect(**_credentials())
    except psycopg.OperationalError as exc:
        raise TransientError(f"database unreachable: {str(exc).splitlines()[0]}") from exc
    try:
        conn.read_only = True                              # the transaction is READ ONLY
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = '{int(timeout_s)}s'")
            cur.execute(statement)
            columns = [d.name for d in cur.description or []]
            fetched = cur.fetchmany(row_cap + 1) if cur.description else []
        conn.rollback()                                    # nothing to keep, ever
        truncated = len(fetched) > row_cap
        return columns, [[_cell(v) for v in r] for r in fetched[:row_cap]], truncated
    except psycopg.errors.QueryCanceled as exc:
        raise SqlError(f"the query ran longer than {timeout_s}s and was stopped — "
                       "add filters or aggregate before returning rows") from exc
    except psycopg.Error as exc:
        raise SqlError(str(exc).strip().splitlines()[0]) from exc
    finally:
        conn.close()


def execute_sql(sql: str, row_cap: int = 1000, timeout_s: int = 20) -> dict:
    """{columns, rows, row_count, truncated, sql, elapsed_ms}. Raises SqlError /
    TransientError; GuardError for a statement the guard refuses."""
    # LIMIT row_cap + 1, not row_cap: the one extra row is how `truncated` is
    # detected. With LIMIT row_cap the database returns exactly row_cap rows and
    # a 50,000-row result would be reported as complete.
    checked = guard(sql, row_cap + 1)
    t0 = time.time()
    columns, rows, truncated = _run(checked.sql, row_cap, timeout_s)
    return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated,
            "sql": checked.sql, "warnings": checked.warnings,
            "elapsed_ms": int((time.time() - t0) * 1000)}


def explain_sql(sql: str, timeout_s: int = 10) -> dict:
    """Plan the query WITHOUT running it: {ok, detail}. A planning error is the
    same error the query would hit, found without spending a real execution."""
    try:
        checked = guard(sql)
    except GuardError as exc:
        return {"ok": False, "detail": f"GUARD: {exc}"}
    try:
        _, rows, _ = _run(f"EXPLAIN (FORMAT JSON) {checked.sql}", 1, timeout_s)
        plan = rows[0][0][0]["Plan"] if rows else {}
        return {"ok": True, "detail": f"valid; planner estimates ~{int(plan.get('Plan Rows', 0)):,} rows"}
    except SqlError as exc:
        return {"ok": False, "detail": str(exc)}
