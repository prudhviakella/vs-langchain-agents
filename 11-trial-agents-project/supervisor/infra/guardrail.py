"""Create the Bedrock Guardrail applied to the Supervisor's model calls.
Same reasoning as trial_graph/infra/guardrail.py.
"""
import boto3

client = boto3.client("bedrock")

GUARDRAIL_NAME = "supervisor-guardrail"


def create_guardrail() -> dict:
    existing = client.list_guardrails(maxResults=100).get("guardrails", [])
    match = next((g for g in existing if g["name"] == GUARDRAIL_NAME), None)
    if match:
        print(f"  guardrail {GUARDRAIL_NAME!r} already exists")
        return {"guardrailId": match["id"], "guardrailArn": match["arn"],
                "version": match.get("version", "DRAFT")}

    print(f"  creating guardrail {GUARDRAIL_NAME!r}")
    response = client.create_guardrail(
        name=GUARDRAIL_NAME,
        description="Content guardrail for the Supervisor agent",
        contentPolicyConfig={"filtersConfig": [
            {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
            {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        ]},
        blockedInputMessaging=(
            "This request falls outside what this assistant can help with."),
        blockedOutputsMessaging=(
            "I can't provide that response — let me know if you'd like to "
            "rephrase your question."),
    )
    return {"guardrailId": response["guardrailId"],
            "guardrailArn": response["guardrailArn"],
            "version": response.get("version", "DRAFT")}
