"""Publish a prompt file to Bedrock Prompt Management.

    prompts/<file>.md  ──► find prompt by name ──► create, or update its DRAFT
                                                        │
                        latest version's text == file?  │
                           yes -> reuse that version    │
                           no  -> create_prompt_version ┘
                                                        v
                                         (id, arn, version) -> Parameter Store

WHY A VERSION ONLY ON CHANGE

The agent runs the version pinned in Parameter Store. Creating a version on
every deploy would move the pin even when nothing changed — the version
number would stop meaning "this text". Comparing first keeps one version per
distinct prompt, so a version number identifies exactly one text, and
rolling back is setting the parameter to an earlier number.

WHY NO modelId

A variant's modelId is optional. The prompt is run by an OpenAI model the
agent calls itself, not by Bedrock, so it is stored model-agnostic.

WHAT THIS DOES NOT DO

    It does not render variables. {{name}} placeholders are declared as
    inputVariables and filled by the agent (config.render) at start or,
    for the composer, per question.
"""
import re
from pathlib import Path

import boto3

agent = boto3.client("bedrock-agent")
VARIANT = "default"


def _find(name: str) -> dict | None:
    token = None
    while True:
        page = agent.list_prompts(maxResults=100, **({"nextToken": token} if token else {}))
        match = next((p for p in page.get("promptSummaries", []) if p["name"] == name), None)
        if match or not page.get("nextToken"):
            return match
        token = page["nextToken"]


def _text(prompt_id: str, version: str) -> str:
    variants = agent.get_prompt(promptIdentifier=prompt_id, promptVersion=version)["variants"]
    return variants[0]["templateConfiguration"]["text"]["text"]


def _latest_version(prompt_id: str) -> str | None:
    versions, token = [], None
    while True:
        page = agent.list_prompts(promptIdentifier=prompt_id, maxResults=100,
                                  **({"nextToken": token} if token else {}))
        versions += [p["version"] for p in page.get("promptSummaries", [])
                     if p["version"].isdigit()]
        token = page.get("nextToken")
        if not token:
            return max(versions, key=int) if versions else None


def publish(name: str, path: Path, description: str) -> dict:
    text = path.read_text()
    variables = sorted(set(re.findall(r"{{\s*(\w+)\s*}}", text)))
    variants = [{"name": VARIANT, "templateType": "TEXT",
                 "templateConfiguration": {"text": {
                     "text": text, "inputVariables": [{"name": v} for v in variables]}}}]

    # STEP 1 — the draft holds the file's text
    found = _find(name)
    if found is None:
        print(f"  creating prompt {name!r}")
        created = agent.create_prompt(name=name, description=description,
                                      variants=variants, defaultVariant=VARIANT)
        prompt_id, arn = created["id"], created["arn"]
    else:
        prompt_id, arn = found["id"], found["arn"]
        if _text(prompt_id, "DRAFT") != text:
            agent.update_prompt(promptIdentifier=prompt_id, name=name,
                                description=description, variants=variants,
                                defaultVariant=VARIANT)

    # STEP 2 — a new version only if the text differs from the latest one
    latest = _latest_version(prompt_id)
    if latest is not None and _text(prompt_id, latest) == text:
        print(f"  prompt {name!r} unchanged — version {latest}")
        return {"id": prompt_id, "arn": arn, "version": latest, "variables": variables}
    version = agent.create_prompt_version(promptIdentifier=prompt_id)["version"]
    print(f"  prompt {name!r} -> version {version}")
    return {"id": prompt_id, "arn": arn, "version": version, "variables": variables}
