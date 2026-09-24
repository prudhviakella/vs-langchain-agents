"""trial_graph configuration — read once from environment variables set
by the AgentCore Runtime deployment (config.yaml / tfvars), not
hardcoded, so dev/tst/val/prd can each point at their own Gateway and
Guardrail without a code change.
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
    row_cap: int
    graph_node_cap: int
    k_entity: int


def _load() -> Config:
    return Config(
        gateway_url=os.environ["GATEWAY_URL"],
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        guardrail_id=os.environ["GUARDRAIL_ID"],
        guardrail_version=os.environ.get("GUARDRAIL_VERSION", "DRAFT"),
        model_id=os.environ.get(
            "MODEL_ID", "us.anthropic.claude-sonnet-4-6-20251001-v1:0"),
        row_cap=int(os.environ.get("ROW_CAP", "500")),
        graph_node_cap=int(os.environ.get("GRAPH_NODE_CAP", "500")),
        k_entity=int(os.environ.get("K_ENTITY", "10")),
    )


CFG = _load()
