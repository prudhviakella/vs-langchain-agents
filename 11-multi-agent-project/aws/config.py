"""Names for every AWS resource this module creates.

One file, because five scripts have to agree on what things are called. A name
that drifts between the deploy script and the teardown script leaves an orphaned
resource billing quietly.

Change PROJECT to run a second isolated copy — useful when a class deploys into
one account.
"""

import os

PROJECT = os.getenv("PROJECT", "trials-agents")
REGION = os.getenv("AWS_REGION", "us-east-1")

# ─────────────────────────────────────────────────────────────────────────────
# The four agents
#
# Same four as the local module. `runtime_name` is what AgentCore calls them;
# the ARN comes back at deploy time and is written to deployed.json, because it
# contains the account id and cannot be derived here.
# ─────────────────────────────────────────────────────────────────────────────
AGENTS = {
    "vdb":        {"entry": "vdb_agent.py",    "desc": "Searches trial documents"},
    "cypher":     {"entry": "cypher_agent.py", "desc": "Queries the trial graph"},
    "chart":      {"entry": "chart_agent.py",  "desc": "Turns tables into Plotly specs"},
    "supervisor": {"entry": "supervisor.py",   "desc": "Routes and composes"},
}

def runtime_name(agent: str) -> str:
    # AgentCore runtime names allow letters, digits and underscores.
    return f"{PROJECT}_{agent}".replace("-", "_")

# ─────────────────────────────────────────────────────────────────────────────
# Tools exposed through the Gateway
#
# These are plain Lambda functions. The Gateway turns each into an MCP tool, so
# an agent discovers and calls them over MCP without knowing Lambda exists.
#
# Note what is NOT here: the vector search and the Cypher query. Those are
# agents, not tools — they hold a model and make judgements. A tool does one
# deterministic thing and returns. Getting that line wrong is the most common
# design mistake in this space: wrapping an agent as a tool loses its streaming
# and its task lifecycle, and wrapping a tool as an agent pays for a model that
# decides nothing.
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = {
    "trial_lookup": {
        "handler": "trial_tools.lookup",
        "description": "Fetch the registry record for one trial by NCT number",
        "input_schema": {
            "type": "object",
            "properties": {"nct_id": {"type": "string",
                                      "description": "e.g. NCT04368728"}},
            "required": ["nct_id"],
        },
    },
    "enrollment_stats": {
        "handler": "trial_tools.enrollment_stats",
        "description": "Summary statistics for enrolment across a list of trials",
        "input_schema": {
            "type": "object",
            "properties": {"nct_ids": {"type": "array",
                                       "items": {"type": "string"}}},
            "required": ["nct_ids"],
        },
    },
}

LAMBDA_NAME = f"{PROJECT}-tools"
GATEWAY_NAME = f"{PROJECT}-gateway"

# Roles. Three, because they are assumed by three different services.
LAMBDA_ROLE = f"{PROJECT}-lambda-role"
GATEWAY_ROLE = f"{PROJECT}-gateway-role"
AGENT_ROLE = f"{PROJECT}-agent-role"

SECRET_NAME = f"{PROJECT}/api-keys"
ECR_REPO = PROJECT

# Where deploy scripts record what they made. Read by the invoke and teardown
# scripts, so nothing has to be pasted between steps.
STATE_FILE = "deployed.json"
