"""Settings for the NLQ agent — environment variables, set by deploy.py on the runtime.

    env ──► CFG (frozen at import) ──► every module reads CFG.<name>
    secrets (OpenAI, Pinecone) ──► api_key(): Secrets Manager id, or a plain env key locally

WHAT THIS DOES NOT DO
    It does not read a runtime-editable registry (ACT's DynamoDB prompt/config
    registry). Changing a budget means a redeploy — simpler, and one less store.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "y")


@dataclass(frozen=True)
class Config:
    agent_name: str = "nlq"
    region: str = os.environ.get("AWS_REGION", "us-east-1")
    model: str = os.environ.get("NLQ_MODEL", "gpt-6-sol")
    embed_model: str = os.environ.get("NLQ_EMBED_MODEL", "text-embedding-3-small")
    # where the SQL runs: "gateway" (AgentCore Gateway -> Lambda beside RDS) or
    # "local" (sql_exec.py in-process, PG* env vars — development only)
    db_mode: str = os.environ.get("NLQ_DB_MODE", "gateway")
    gateway_url: str = os.environ.get("NLQ_GATEWAY_URL", "")
    gateway_target: str = os.environ.get("NLQ_GATEWAY_TARGET", "nlq-sql-tools")
    pinecone_index: str = os.environ.get("PINECONE_INDEX", "ecom-kb")
    # grounding sizes
    k_tables: int = _int("NLQ_K_TABLES", 6)
    k_columns: int = _int("NLQ_K_COLUMNS", 20)
    k_examples: int = _int("NLQ_K_EXAMPLES", 4)
    k_glossary: int = _int("NLQ_K_GLOSSARY", 5)
    max_columns: int = _int("NLQ_MAX_COLUMNS", 120)
    # ACT retrieves only verified examples. Ours are all `verified = N` until
    # the team reviews them — True here would ground with NO examples at all.
    examples_verified_only: bool = _bool("NLQ_EXAMPLES_VERIFIED_ONLY", False)
    # budgets (per question)
    max_tool_calls: int = _int("NLQ_MAX_TOOL_CALLS", 12)
    max_queries: int = _int("NLQ_MAX_QUERIES", 6)
    max_repairs: int = _int("NLQ_MAX_REPAIRS", 3)
    queries_after_result: int = _int("NLQ_QUERIES_AFTER_RESULT", 1)
    recursion_limit: int = _int("NLQ_RECURSION_LIMIT", 40)
    # results
    row_cap: int = _int("NLQ_ROW_CAP", 1000)
    query_timeout_s: int = _int("NLQ_QUERY_TIMEOUT_S", 20)


CFG = Config()


@lru_cache(maxsize=None)
def api_key(env_var: str, secret_env: str) -> str:
    """A key from Secrets Manager ({"api_key": …}) when <secret_env> names a
    secret, else the plain <env_var>. Cached: read once per container."""
    secret_id = os.environ.get(secret_env)
    if secret_id:
        import boto3
        raw = boto3.client("secretsmanager", region_name=CFG.region).get_secret_value(SecretId=secret_id)
        return json.loads(raw["SecretString"])["api_key"]
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"{env_var} is not set (nor {secret_env})")
    return key
