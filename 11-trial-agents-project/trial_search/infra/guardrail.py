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

DENIED TOPICS — SPECIFIC RISKS, NOT "EVERYTHING OFF-TOPIC"

Four themes this platform must never engage with, each defined POSITIVELY,
as AWS requires ("avoid negative definitions" — "all contents except
medical information" is AWS's own example of what not to write):

    Personal medical advice      a decision for a specific person
    Deceiving trial staff        hiding or faking information to be accepted
    Participant identification   finding or contacting trial participants
    Investment advice            trading on trial information

Off-topic questions in general (recipes, code) are the supervisor prompt's
job: it sets out_of_scope and the platform replies with a fixed message. A
negatively defined topic would also catch "remember that I focus on phase
3" and "do a deeper analysis", which never mention a trial.

PERSONAL DATA IS MASKED, NOT BLOCKED

Emails and phone numbers become {EMAIL} and {PHONE}, in questions and
answers. Protocols print medical monitors' contact details; blocking would
refuse every answer that quotes one. Names are left alone: investigators
and sponsors are named in the registry.

WHAT THIS DOES NOT DO

    It does not decide whether a question is in scope — the supervisor does.
    It does not use the STANDARD topic tier: questions are in English, which
    the CLASSIC tier covers.
"""
import time

import boto3

bedrock = boto3.client("bedrock")

NAME = "trial-agents-guardrail"
BLOCKED_INPUT = ("This assistant answers questions about the 20 clinical trials in its registry "
                 "and protocols. It cannot give personal medical, investment or enrolment advice, "
                 "or help identify trial participants.")
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


def _topic(name, definition, examples):
    return {"name": name, "definition": definition, "examples": examples, "type": "DENY",
            "inputEnabled": True, "outputEnabled": True, "inputAction": "BLOCK", "outputAction": "BLOCK"}


TOPICS = [
    _topic("Personal medical advice",
           "Recommendations about diagnosis, treatment, medication, dosing, or whether to start or "
           "stop a therapy for a specific person, including the person asking.",
           ["Should I stop taking my semaglutide?",
            "What dose of atezolizumab should my father take?",
            "Do my symptoms mean I have hepatitis B?",
            "Is this drug safe for me while I am pregnant?",
            "Should my mother join a cancer trial instead of having chemotherapy?"]),
    # Defined by the DECEPTION, not by eligibility. The first version named
    # IMbrave150 and used "enrol", "qualify" and "screening" in its examples;
    # the classifier learned "IMbrave150 + eligibility" and blocked "What are
    # the exclusion criteria of the IMbrave150 trial?" — the platform's core
    # question. Examples here name no trial and use no eligibility vocabulary.
    _topic("Deceiving trial staff",
           "Help with lying to, misleading or hiding information from doctors or study staff, "
           "such as concealing an illness or faking a test result, to be accepted into a study.",
           ["How can I hide my liver disease from the study doctor?",
            "What should I lie about so they accept me?",
            "How do I fake a lower blood pressure reading at my visit?",
            "Can I conceal my past chemotherapy from the investigators?"]),
    _topic("Participant identification",
           "Attempts to identify, locate, contact or learn personal details about individual people "
           "who took part in a clinical trial.",
           ["Who were the patients enrolled at the Seoul site?",
            "Give me the names of participants in STEP 1.",
            "How can I contact people who were in the PIONEER 4 trial?",
            "Which patient had the serious adverse event at Georgetown?"]),
    _topic("Investment advice",
           "Recommendations to buy, sell or hold shares or other securities, or predictions of share "
           "prices, based on clinical trial information.",
           ["Should I buy Novo Nordisk stock after STEP 1?",
            "Will Roche shares rise because of IMbrave150?",
            "Is Moderna a good investment based on its trials?"]),
]
PII = [{"type": t, "action": "ANONYMIZE", "inputAction": "ANONYMIZE", "outputAction": "ANONYMIZE",
        "inputEnabled": True, "outputEnabled": True} for t in ("EMAIL", "PHONE")]


def _policy(g: dict) -> tuple:
    """Everything the deploy controls, in a comparable form: a change to any
    filter, topic, example phrase, personal-data rule or message counts."""
    filters = {(f["type"], f["inputStrength"], f["outputStrength"])
               for f in g.get("contentPolicy", {}).get("filters", [])}
    topics = {(t["name"], t["definition"], tuple(t.get("examples", [])),
               t.get("inputAction", "BLOCK"), t.get("outputAction", "BLOCK"))
              for t in g.get("topicPolicy", {}).get("topics", [])}
    pii = {(e["type"], e.get("inputAction", e.get("action")), e.get("outputAction", e.get("action")))
           for e in g.get("sensitiveInformationPolicy", {}).get("piiEntities", [])}
    return (frozenset(filters), frozenset(topics), frozenset(pii),
            g.get("blockedInputMessaging"), g.get("blockedOutputsMessaging"))


_DESIRED = _policy({"contentPolicy": {"filters": FILTERS},
                    "topicPolicy": {"topics": TOPICS},
                    "sensitiveInformationPolicy": {"piiEntities": PII},
                    "blockedInputMessaging": BLOCKED_INPUT, "blockedOutputsMessaging": BLOCKED_OUTPUT})


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
                  topicPolicyConfig={"topicsConfig": TOPICS},
                  sensitiveInformationPolicyConfig={"piiEntitiesConfig": PII},
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
