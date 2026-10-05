"""Where the SQL actually runs.

    tools.execute_sql ──► db.execute(sql)
                            ├─ gateway: SigV4 MCP session ──► AgentCore Gateway ──► Lambda (VPC) ──► RDS
                            └─ local:   sql_exec.py in-process ──► PG* env (your laptop, development)

The Lambda returns errors as DATA — {"error", "kind": guard|sql|transient} —
and they are raised here as typed exceptions, so the tools treat both modes
identically. TRANSIENT failures (warehouse cold, network) are retried here;
SQL errors are not: the model must see them and rewrite.

WHAT THIS DOES NOT DO
    It does not guard the SQL — middleware already did, and the executor does
    again. It does not keep a session open between calls.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from .config import CFG


class SqlError(Exception):
    pass


class GuardRejected(Exception):
    pass


class TransientError(Exception):
    pass


def _local():
    """sql_exec as the Lambda runs it: a flat folder where `guard.py` sits beside
    `sql_exec.py` (deploy copies the agent's guard.py into the Lambda zip).
    Locally both folders go on the path to reproduce that layout."""
    for folder in (Path(__file__).resolve().parents[2] / "lambda_tools", Path(__file__).resolve().parent):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    import sql_exec
    return sql_exec


async def _gateway_call(tool: str, args: dict) -> dict:
    from mcp import ClientSession
    from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
    async with aws_iam_streamablehttp_client(endpoint=CFG.gateway_url, aws_service="bedrock-agentcore",
                                             aws_region=CFG.region) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(f"{CFG.gateway_target}___{tool}", args)
    text = "".join(getattr(c, "text", "") for c in result.content or [])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": text or "empty response from the SQL tool", "kind": "transient"}


def _raise_for(payload: dict) -> dict:
    kind = payload.get("kind")
    if "error" not in payload:
        return payload
    if kind == "guard":
        raise GuardRejected(payload["error"])
    if kind == "sql":
        raise SqlError(payload["error"])
    raise TransientError(payload["error"])


async def _call(tool: str, args: dict, attempts: int = 3) -> dict:
    delay = 1.0
    for attempt in range(attempts):
        try:
            if CFG.db_mode == "local":
                x = _local()
                try:
                    fn = x.execute_sql if tool == "execute_sql" else x.explain_sql
                    return await asyncio.to_thread(fn, **args)
                except x.GuardError as exc:
                    raise GuardRejected(str(exc)) from exc
                except x.SqlError as exc:
                    raise SqlError(str(exc)) from exc
                except x.TransientError as exc:
                    raise TransientError(str(exc)) from exc
            return _raise_for(await _gateway_call(tool, args))
        except TransientError:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


async def execute(sql: str) -> dict:
    return await _call("execute_sql", {"sql": sql, "row_cap": CFG.row_cap, "timeout_s": CFG.query_timeout_s})


async def explain(sql: str) -> dict:
    return await _call("explain_sql", {"sql": sql})
