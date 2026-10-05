"""Settings for the supervisor — environment variables, set by deploy.py on the runtime.

    specialists: deployed  -> NLQ_ARN / CHART_ARN    (invoke_agent_runtime, SigV4)
                 local dev -> NLQ_URL / CHART_URL    (http://localhost:9001/ …, plain A2A)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Config:
    region: str = os.environ.get("AWS_REGION", "us-east-1")
    model: str = os.environ.get("SUPERVISOR_MODEL", "gpt-6-sol")
    nlq_arn: str = os.environ.get("NLQ_ARN", "")
    nlq_url: str = os.environ.get("NLQ_URL", "")
    chart_arn: str = os.environ.get("CHART_ARN", "")
    chart_url: str = os.environ.get("CHART_URL", "")
    max_nlq_calls: int = int(os.environ.get("SUPERVISOR_MAX_NLQ_CALLS", 3))
    chart_min_rows: int = int(os.environ.get("SUPERVISOR_CHART_MIN_ROWS", 3))
    rows_to_composer: int = int(os.environ.get("SUPERVISOR_ROWS_TO_COMPOSER", 40))
    history_chars: int = int(os.environ.get("SUPERVISOR_HISTORY_CHARS", 12000))
    specialist_timeout_s: int = int(os.environ.get("SUPERVISOR_SPECIALIST_TIMEOUT_S", 300))


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
