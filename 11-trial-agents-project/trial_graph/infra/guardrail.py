"""Create the Bedrock Guardrail applied to trial_graph's model calls.

    deploy.py
        |
        v
    infra/guardrail.py
        |
        `-- create_guardrail()   content filters + PII handling, wired
                                 into ChatBedrockConverse's own
                                 guardrail_config at the agent's model
                                 call (core.py) — this script only
                                 creates the guardrail resource itself

WHAT THIS GUARDS AGAINST, AND WHY THESE SPECIFIC CHOICES

This agent answers questions about clinical trial documents — sponsors,
drugs, sites, outcomes. It has no legitimate reason to discuss anything
sexual, violent, or hateful, so those filters are set to their strongest
setting with no real cost to normal use.

PII is handled differently on input vs output, deliberately. On input,
an analyst's own question might reasonably mention a name in passing
("did the Pfizer trial include children") — that's not sensitive data
about a document, it's ordinary conversation, so input PII detection is
left off. On output, the graph's own Site nodes carry real facility
addresses and PatientPopulation nodes carry population descriptions —
none of it is individual-patient PII (the graph has no patient-level
data at all), so no PII policy is needed there either. The one real
sensitive-data channel — the Neo4j credential itself — is handled by
Secrets Manager (see infra/lambda_deploy.py), not by this guardrail;
a guardrail filters model text, not infrastructure secrets.

WHAT THIS DOES NOT DO

    - It does not configure a denied-topics list. This agent's scope is
      already narrowed structurally — ScopeDisciplineMiddleware in
      core.py (not yet built) is where "don't answer questions outside
      the trial graph" belongs, since that is a judgment about the
      QUESTION, which a keyword-based denied-topic filter cannot make
      reliably. A guardrail is the wrong tool for that job.
"""
import boto3

client = boto3.client("bedrock")

GUARDRAIL_NAME = "trial-graph-guardrail"


def create_guardrail() -> dict:
    """Create the guardrail if it doesn't exist yet. Returns
    {"guardrailId", "guardrailArn", "version"}.
    """
    existing = client.list_guardrails(maxResults=100).get("guardrails", [])
    match = next((g for g in existing if g["name"] == GUARDRAIL_NAME), None)
    if match:
        print(f"  guardrail {GUARDRAIL_NAME!r} already exists")
        return {"guardrailId": match["id"], "guardrailArn": match["arn"],
                "version": match.get("version", "DRAFT")}

    print(f"  creating guardrail {GUARDRAIL_NAME!r}")
    response = client.create_guardrail(
        name=GUARDRAIL_NAME,
        description="Content guardrail for the trial_graph agent",
        contentPolicyConfig={"filtersConfig": [
            {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
            {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        ]},
        blockedInputMessaging=(
            "This request falls outside what this agent can help with."),
        blockedOutputsMessaging=(
            "I can't provide that response — let me know if you'd like to "
            "rephrase your question about the trial data."),
    )
    return {"guardrailId": response["guardrailId"],
            "guardrailArn": response["guardrailArn"],
            "version": response.get("version", "DRAFT")}
