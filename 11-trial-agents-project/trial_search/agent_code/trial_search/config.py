"""trial_search configuration — read once from the environment the
AgentCore Runtime deployment sets (deploy.py -> environmentVariables).

    call budgets (recursion)        per turn, per tool
        MAX_SEARCHES_PER_TURN       5
        MAX_NEIGHBOR_CALLS          3     Case A
        MAX_TABLE_CALLS             3     Case C

    size budgets
        MAX_WINDOW                  10    hops per expand_neighbors call
        EXPANSION_TOKEN_BUDGET      6000  tokens of expanded text per turn,
                                          shared by Case A and Case C

WHAT THIS DOES NOT DO

    It does not enforce anything. RetrievalMiddleware (core.py) reads
    these numbers and enforces them; this file only holds them.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    gateway_url: str
    aws_region: str
    guardrail_id: str
    guardrail_version: str
    model_id: str
    max_searches_per_turn: int
    max_neighbor_calls: int
    max_table_calls: int
    max_window: int
    expansion_token_budget: int


def _load() -> Config:
    env = os.environ.get
    return Config(
        gateway_url=os.environ["GATEWAY_URL"],
        aws_region=env("AWS_REGION", "us-east-1"),
        guardrail_id=os.environ["GUARDRAIL_ID"],
        guardrail_version=env("GUARDRAIL_VERSION", "DRAFT"),
        model_id=env("MODEL_ID", "us.anthropic.claude-sonnet-4-6-20251001-v1:0"),
        max_searches_per_turn=int(env("MAX_SEARCHES_PER_TURN", "5")),
        max_neighbor_calls=int(env("MAX_NEIGHBOR_CALLS", "3")),
        max_table_calls=int(env("MAX_TABLE_CALLS", "3")),
        max_window=int(env("MAX_WINDOW", "10")),
        expansion_token_budget=int(env("EXPANSION_TOKEN_BUDGET", "6000")),
    )


CFG = _load()
