"""The Bedrock guardrail all three agents apply, through ApplyGuardrail.

    ensure_guardrail()
        │
        ├─ create it, or bring its DRAFT to the policy below
        └─ a numbered version, created only when the policy changed
                │
                v
        (id, arn, version) -> each agent's Parameter Store settings

ONE GUARDRAIL, NOT ONE PER AGENT

The policy is the same for every agent. Three copies drift the first time
one is edited.

FILTER STRENGTHS ARE SET FOR CLINICAL TEXT

A higher strength blocks on LOWER-confidence detections. Answers about
trials legitimately mention deaths, overdoses, self-harm screening and
adverse events; at HIGH, the VIOLENCE and MISCONDUCT filters would block
answers restating a protocol's safety section. So those two are MEDIUM on
input and LOW on output: an analyst's question is held to a stricter bar
than the model's restatement of clinical evidence. SEXUAL and HATE have no
legitimate use here and stay HIGH. This calibration is reasoned from the
domain; verify it against real answers before relying on it.

PROMPT_ATTACK

Detects instructions smuggled into the analyst's input ("ignore your
instructions and ..."). It applies to input only — outputStrength must be
NONE — which is why the agents check the question with source=INPUT.

WHAT THIS DOES NOT DO

    No PII policy. The corpus holds no patient-level data. No denied topics:
    whether a question is in scope is the prompt's job, not a keyword list's.
"""
import time

import boto3

bedrock = boto3.client("bedrock")

NAME = "trial-agents-guardrail"
BLOCKED_INPUT = "This request falls outside what this assistant can help with."
BLOCKED_OUTPUT = ("The answer to this question could not be shown because it did not "
                  "pass a content policy. Try rephrasing the question.")
FILTERS = [
    {"type": "SEXUAL",        "inputStrength": "HIGH",   "outputStrength": "HIGH"},
    {"type": "HATE",          "inputStrength": "HIGH",   "outputStrength": "HIGH"},
    {"type": "INSULTS",       "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
    {"type": "VIOLENCE",      "inputStrength": "MEDIUM", "outputStrength": "LOW"},
    {"type": "MISCONDUCT",    "inputStrength": "MEDIUM", "outputStrength": "LOW"},
    {"type": "PROMPT_ATTACK", "inputStrength": "HIGH",   "outputStrength": "NONE"},
]


def _policy(g: dict) -> tuple:
    filters = {(f["type"], f["inputStrength"], f["outputStrength"])
               for f in g.get("contentPolicy", {}).get("filters", [])}
    return (frozenset(filters), g.get("blockedInputMessaging"), g.get("blockedOutputsMessaging"))


_DESIRED = (frozenset((f["type"], f["inputStrength"], f["outputStrength"]) for f in FILTERS),
            BLOCKED_INPUT, BLOCKED_OUTPUT)


def _wait(guardrail_id: str) -> None:
    deadline = time.time() + 120
    while (status := bedrock.get_guardrail(guardrailIdentifier=guardrail_id,
                                           guardrailVersion="DRAFT")["status"]) != "READY":
        if status == "FAILED" or time.time() > deadline:
            raise RuntimeError(f"guardrail {NAME!r} is {status}")
        time.sleep(3)


def _find() -> dict | None:
    token = None
    while True:
        page = bedrock.list_guardrails(maxResults=100, **({"nextToken": token} if token else {}))
        match = next((g for g in page["guardrails"] if g["name"] == NAME), None)
        if match or not page.get("nextToken"):
            return match
        token = page["nextToken"]


def ensure_guardrail() -> dict:
    config = dict(contentPolicyConfig={"filtersConfig": FILTERS},
                  blockedInputMessaging=BLOCKED_INPUT, blockedOutputsMessaging=BLOCKED_OUTPUT)

    # STEP 1 — the DRAFT matches the policy above
    found = _find()
    if found is None:
        print(f"  creating guardrail {NAME!r}")
        created = bedrock.create_guardrail(name=NAME, description="trial agents", **config)
        guardrail_id, arn = created["guardrailId"], created["guardrailArn"]
    else:
        guardrail_id, arn = found["id"], found["arn"]
        draft = bedrock.get_guardrail(guardrailIdentifier=guardrail_id, guardrailVersion="DRAFT")
        if _policy(draft) != _DESIRED:
            print(f"  updating guardrail {NAME!r} policy")
            bedrock.update_guardrail(guardrailIdentifier=guardrail_id, name=NAME,
                                     description="trial agents", **config)
    _wait(guardrail_id)

    # STEP 2 — reuse the latest numbered version if it already has this policy
    versions = [g["version"] for g in bedrock.list_guardrails(
        guardrailIdentifier=guardrail_id, maxResults=100)["guardrails"]
        if g["version"].isdigit()]
    if versions:
        latest = max(versions, key=int)
        if _policy(bedrock.get_guardrail(guardrailIdentifier=guardrail_id,
                                         guardrailVersion=latest)) == _DESIRED:
            print(f"  guardrail {NAME!r} unchanged — version {latest}")
            return {"id": guardrail_id, "arn": arn, "version": latest}
    version = bedrock.create_guardrail_version(guardrailIdentifier=guardrail_id)["version"]
    print(f"  guardrail {NAME!r} -> version {version}")
    return {"id": guardrail_id, "arn": arn, "version": version}
