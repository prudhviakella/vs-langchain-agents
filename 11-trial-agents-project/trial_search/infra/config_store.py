"""Where the agents' configuration lives in AWS, and the helpers that write it.

    Secrets Manager   trial-agents/openai              {"api_key", "model"}
                      shared by all three agents AND trial_search's embedding
                      Lambda: one key, so rotating it is one operation
    Parameter Store   /trial-agents/<agent>/<name>     plain settings
                      /trial-agents/registry/<agent>   {"arn", "description"}
                                                       written by each specialist
                                                       when it deploys; read by
                                                       the supervisor

WHY A REGISTRY RATHER THAN A LIST IN THE SUPERVISOR

A specialist that deploys writes its own ARN and description. The
supervisor reads the registry at start and renders its prompt's AVAILABLE
AGENTS block from it. Adding a specialist is deploying it, then redeploying
the supervisor (which re-scopes its IAM to the new ARN) — no code change.

WHAT THIS DOES NOT DO

    It never writes a real credential. A new secret is created with
    "replace-me" values, and the agent refuses to start until an operator
    sets real ones.
"""
import json

import boto3

OPENAI_SECRET = "trial-agents/openai"
ROOT = "/trial-agents"
REGISTRY_PATH = f"{ROOT}/registry"


def prefix(agent: str) -> str:
    return f"{ROOT}/{agent}"


def ensure_secret(name: str, placeholder: dict) -> str:
    """Create `name` with placeholder values if it does not exist. Its ARN."""
    sm = boto3.client("secretsmanager")
    try:
        return sm.describe_secret(SecretId=name)["ARN"]
    except sm.exceptions.ResourceNotFoundException:
        print(f"  creating placeholder secret {name!r}")
        return sm.create_secret(Name=name, SecretString=json.dumps(placeholder))["ARN"]


def ensure_openai_secret() -> str:
    return ensure_secret(OPENAI_SECRET, {"api_key": "replace-me", "model": "replace-me"})


def placeholders(name: str) -> list[str]:
    """Keys of secret `name` still holding "replace-me"."""
    value = json.loads(boto3.client("secretsmanager").get_secret_value(
        SecretId=name)["SecretString"])
    return sorted(k for k, v in value.items() if v == "replace-me")


def put_parameters(path: str, values: dict) -> None:
    """Write each value as <path>/<key>. Overwrite: the deploy is the source."""
    ssm = boto3.client("ssm")
    for key, value in values.items():
        ssm.put_parameter(Name=f"{path}/{key}", Value=str(value),
                          Type="String", Overwrite=True)
    print(f"  wrote {len(values)} parameter(s) under {path}")


def register(agent: str, runtime_arn: str, description: str) -> None:
    put_parameters(REGISTRY_PATH, {agent: json.dumps(
        {"arn": runtime_arn, "description": description})})


def read_registry() -> dict:
    ssm = boto3.client("ssm")
    found, token = {}, None
    while True:
        page = ssm.get_parameters_by_path(Path=REGISTRY_PATH,
                                          **({"NextToken": token} if token else {}))
        for p in page["Parameters"]:
            found[p["Name"].rsplit("/", 1)[1]] = json.loads(p["Value"])
        token = page.get("NextToken")
        if not token:
            return found
