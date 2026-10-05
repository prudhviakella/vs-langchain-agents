"""Settings for chart_gen — environment variables, set by deploy.py on the runtime."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Config:
    region: str = os.environ.get("AWS_REGION", "us-east-1")
    model: str = os.environ.get("CHART_MODEL", "gpt-6-sol")
    max_figures: int = int(os.environ.get("CHART_MAX_FIGURES", 3))
    max_insight_chars: int = int(os.environ.get("CHART_MAX_INSIGHT_CHARS", 240))
    max_rows_to_model: int = int(os.environ.get("CHART_MAX_ROWS_TO_MODEL", 300))


CFG = Config()


@lru_cache(maxsize=None)
def api_key(env_var: str = "OPENAI_API_KEY", secret_env: str = "OPENAI_SECRET_ID") -> str:
    secret_id = os.environ.get(secret_env)
    if secret_id:
        import boto3
        raw = boto3.client("secretsmanager", region_name=CFG.region).get_secret_value(SecretId=secret_id)
        return json.loads(raw["SecretString"])["api_key"]
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"{env_var} is not set (nor {secret_env})")
    return key
