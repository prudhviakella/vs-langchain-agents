"""The Lambda handlers that become MCP tools.

Plain functions. Nothing here knows about MCP, agents or Bedrock — the Gateway
handles that translation, which is the point of using it.

WHY THESE ARE TOOLS AND NOT AGENTS

Each does one deterministic thing and returns. No model, no judgement, no
conversation. That is what separates a tool from an agent, and the line is worth
holding:

    a tool    does exactly what it is told, every time
    an agent  decides what to do

Wrap an agent as a tool and you lose its streaming and its task lifecycle. Wrap
a tool as an agent and you pay for a model that decides nothing.
"""

import json
import os
import urllib.request

CT_API = "https://clinicaltrials.gov/api/v2/studies"


def _fetch(nct_id: str) -> dict:
    """One study from the registry. Public API, no key."""
    with urllib.request.urlopen(f"{CT_API}/{nct_id}?format=json", timeout=20) as r:
        return json.loads(r.read())


def lookup(event, context):
    """Registry facts for one trial.

    The Gateway passes the tool's arguments as the event. Whatever this returns
    becomes the tool result the agent sees, so it should be small and already
    shaped — an agent asked to parse a 40 KB registry response will spend tokens
    doing it and sometimes get it wrong.
    """
    nct_id = (event.get("nct_id") or "").strip().upper()
    if not nct_id:
        return {"error": "nct_id is required"}

    try:
        raw = _fetch(nct_id)
    except Exception as exc:
        # Say what failed. An agent told only "error" will retry, and retrying a
        # 404 just burns its call budget to be refused again.
        return {"error": f"registry lookup failed for {nct_id}: {str(exc)[:120]}"}

    protocol = raw.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    design = protocol.get("designModule", {})
    status = protocol.get("statusModule", {})
    sponsors = protocol.get("sponsorCollaboratorsModule", {})
    conditions = protocol.get("conditionsModule", {})

    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle"),
        "phase": ", ".join(design.get("phases", [])),
        "status": status.get("overallStatus"),
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "sponsor": sponsors.get("leadSponsor", {}).get("name"),
        "conditions": conditions.get("conditions", []),
    }


def enrollment_stats(event, context):
    """Enrolment across several trials, summarised.

    Exists because it is the kind of question an agent answers badly on its own:
    given five records it will add the numbers in its head, and it will
    occasionally get it wrong in a way that reads perfectly plausibly.

    Arithmetic belongs in code. This is the general principle behind giving an
    agent tools at all.
    """
    nct_ids = event.get("nct_ids") or []
    if not nct_ids:
        return {"error": "nct_ids is required"}

    counts, missing = [], []
    for nct_id in nct_ids[:25]:            # cap the fan-out; Lambda has a timeout
        try:
            raw = _fetch(nct_id.strip().upper())
            count = (raw.get("protocolSection", {}).get("designModule", {})
                        .get("enrollmentInfo", {}).get("count"))
            counts.append(count) if count else missing.append(nct_id)
        except Exception:
            missing.append(nct_id)

    if not counts:
        return {"error": "no enrolment figures found", "missing": missing}

    return {
        "trials": len(counts),
        "total": sum(counts),
        "mean": round(sum(counts) / len(counts), 1),
        "min": min(counts),
        "max": max(counts),
        # Reported rather than hidden. An average over 3 of 5 trials is a
        # different number from an average over 5, and the agent should be able
        # to say which it has.
        "missing": missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lambda entrypoint
#
# The Gateway invokes one Lambda and passes the tool name in the context, so a
# single function can back several tools. Fewer functions means fewer cold
# starts and one place to deploy.
# ─────────────────────────────────────────────────────────────────────────────
HANDLERS = {"trial_lookup": lookup, "enrollment_stats": enrollment_stats}


def handler(event, context):
    """Route to the right tool.

    AgentCore puts the tool name in the client context. The prefix it arrives
    with depends on the target name, so this takes the last segment rather than
    matching the whole string — a target rename would otherwise break every tool
    silently.
    """
    name = ""
    client_context = getattr(context, "client_context", None)
    if client_context and getattr(client_context, "custom", None):
        name = client_context.custom.get("bedrockAgentCoreToolName", "")
    name = name.split("___")[-1] if name else event.get("__tool_name__", "")

    tool = HANDLERS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}. Known: {list(HANDLERS)}"}
    return tool(event, context)
