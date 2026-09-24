"""Supervisor configuration — same env-var-driven pattern as trial_graph
and trial_search's own config.py."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    trial_graph_arn: str
    trial_search_arn: str
    aws_region: str
    guardrail_id: str
    guardrail_version: str
    model_id: str
    max_agent_calls_per_turn: int


def _load() -> Config:
    return Config(
        trial_graph_arn=os.environ["TRIAL_GRAPH_RUNTIME_ARN"],
        trial_search_arn=os.environ["TRIAL_SEARCH_RUNTIME_ARN"],
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        guardrail_id=os.environ["GUARDRAIL_ID"],
        guardrail_version=os.environ.get("GUARDRAIL_VERSION", "DRAFT"),
        model_id=os.environ.get(
            "MODEL_ID", "us.anthropic.claude-sonnet-4-6-20251001-v1:0"),
        max_agent_calls_per_turn=int(os.environ.get("MAX_AGENT_CALLS_PER_TURN", "6")),
    )


CFG = _load()
