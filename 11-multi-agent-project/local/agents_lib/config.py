"""Settings for the fleet.

One place, because four agents have to agree on ports, model provider and the
shape of what they send each other. Disagreement between two of them is the
hardest class of bug in a distributed system: nothing errors, and one agent
quietly ignores what the other sent.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Model provider
#
# Both paths are written. Switch with one variable, because the point of the
# module is the architecture, and the architecture does not care which model
# routes the questions.
#
#   PROVIDER=openai      gpt-4o-mini
#   PROVIDER=anthropic   claude-haiku
#
# Bedrock is deliberately not here. It is the deployment target for module 12,
# and paying Bedrock prices to learn the protocol is the wrong trade.
# ─────────────────────────────────────────────────────────────────────────────
PROVIDER = os.getenv("PROVIDER", "openai").lower()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ─────────────────────────────────────────────────────────────────────────────
# Where each agent listens
#
# Fixed ports rather than discovery, because there is no registry on a laptop.
# In module 12 these become AgentCore runtime ARNs and this table disappears —
# which is the point: the endpoint is configuration, never code.
# ─────────────────────────────────────────────────────────────────────────────
PORTS = {"supervisor": 9200, "vdb": 9201, "cypher": 9202, "chart": 9203}

def endpoint(agent: str) -> str:
    return f"http://127.0.0.1:{PORTS[agent]}"

# ─────────────────────────────────────────────────────────────────────────────
# Budgets
#
# Enforced in code, not asked for in a prompt. A prompt that says "be efficient"
# is a suggestion the model may ignore on any given turn; a counter that refuses
# the fourth call is a wall. The difference matters when each call costs money
# and the failure mode is an unbounded loop.
# ─────────────────────────────────────────────────────────────────────────────
MAX_AGENT_CALLS = int(os.getenv("MAX_AGENT_CALLS", "4"))

# Below this many rows a chart is noise to dismiss, plus a wasted call and its
# latency on every such turn. A two-row identifier lookup is not chartable data.
# The TABLE is attached either way; only the chart is gated.
CHART_MIN_ROWS = int(os.getenv("CHART_MIN_ROWS", "3"))

# How many rows the COMPOSER sees, verbatim. Bounded, but not zero: given only
# counts it writes "the query returned 2 rows", which is a status report rather
# than an answer.
COMPOSER_ROWS = int(os.getenv("COMPOSER_ROWS", "10"))
AGENT_TIMEOUT_S = int(os.getenv("AGENT_TIMEOUT_S", "120"))

# How many rows of a result the supervisor's model is allowed to see.
#
# The model routes and composes; it does not process data. Handing it 400 rows
# costs tokens, adds latency, and invites it to retype values — and a model
# retyping an NCT number will eventually mangle one. The full set travels
# between agents; the model sees a summary.
MODEL_ROW_PREVIEW = int(os.getenv("MODEL_ROW_PREVIEW", "3"))

# Data stores, shared with modules 05 and 08.
INDEX_NAME = os.getenv("INDEX_NAME", "rag-docs")
NAMESPACE = os.getenv("NAMESPACE", "")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
