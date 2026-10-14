"""Backend settings — environment variables (local: webapp/backend/.env; AWS: the ECS task definition).

    SUPERVISOR_ARN      deployed supervisor (AgentCore, SigV4)  — or —
    SUPERVISOR_URL      a local supervisor A2A server (http://localhost:9000/)
    APP_STORE           dynamodb (default) | memory (tests)
    APP_TABLE           DynamoDB table name (created on first start if missing)
    ADMIN_USERS         who sees /admin: unset or "*" = EVERY user (the default); "a,b" = only those
    MODEL_PRICES        JSON, USD per 1M tokens per model — unset: costs show "—"
    PINECONE_API_KEY / PINECONE_SECRET_ID   the KB browser (admin)
    OPENAI_API_KEY / OPENAI_SECRET_ID       the KB browser's "search by meaning" (optional)
    SESSION_SECRET_ID   Secrets Manager id of the cookie-signing key — or SESSION_SECRET locally
    COOKIE_SECURE       true behind HTTPS (CloudFront); false for http://localhost
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    region: str = os.environ.get("AWS_REGION", "us-east-1")
    supervisor_arn: str = os.environ.get("SUPERVISOR_ARN", "")
    supervisor_url: str = os.environ.get("SUPERVISOR_URL", "")
    store: str = os.environ.get("APP_STORE", "dynamodb")
    table: str = os.environ.get("APP_TABLE", "ecom-webapp")
    cookie_name: str = "ecom_session"
    cookie_secure: bool = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
    session_days: int = int(os.environ.get("SESSION_DAYS", 7))
    history_messages: int = int(os.environ.get("HISTORY_MESSAGES", 10))
    keepalive_s: float = float(os.environ.get("KEEPALIVE_S", 5))
    turn_timeout_s: int = int(os.environ.get("TURN_TIMEOUT_S", 600))
    # USD per 1M tokens, per model: {"gpt-6-sol": {"input": 0.0, "output": 0.0}}. Unset -> cost shown as "—".
    model_prices: dict = field(default_factory=lambda: json.loads(os.environ.get("MODEL_PRICES") or "{}"))
    agent_model: str = os.environ.get("AGENT_MODEL", "gpt-6-sol")
    pinecone_index: str = os.environ.get("PINECONE_INDEX", "ecom-kb")
    embed_model: str = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    admin_users: tuple = field(default_factory=lambda: tuple(
        u.strip().lower() for u in (os.environ.get("ADMIN_USERS") or "*").split(",") if u.strip()))


S = Settings()


@lru_cache(maxsize=None)
def api_key(env_var: str, secret_env: str) -> str | None:
    """A key from Secrets Manager ({"api_key": …}) or the environment; None when neither is set."""
    if os.environ.get(secret_env):
        import boto3
        raw = boto3.client("secretsmanager", region_name=S.region).get_secret_value(SecretId=os.environ[secret_env])
        return json.loads(raw["SecretString"])["api_key"]
    return os.environ.get(env_var) or None


@lru_cache(maxsize=1)
def session_secret() -> bytes:
    """One signing key for every backend task — from Secrets Manager, so a cookie
    issued by one task is accepted by another."""
    if os.environ.get("SESSION_SECRET_ID"):
        import boto3
        raw = boto3.client("secretsmanager", region_name=S.region).get_secret_value(
            SecretId=os.environ["SESSION_SECRET_ID"])["SecretString"]
        try:
            return json.loads(raw)["key"].encode()
        except (ValueError, KeyError):
            return raw.encode()
    secret = os.environ.get("SESSION_SECRET")
    if not secret:
        raise RuntimeError("SESSION_SECRET (local) or SESSION_SECRET_ID (AWS) must be set")
    return secret.encode()
