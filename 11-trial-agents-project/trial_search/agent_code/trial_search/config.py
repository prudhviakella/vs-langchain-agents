"""trial_search configuration — loaded once, at container start, from AWS.

    env PARAM_PREFIX = /trial-agents/trial_search          the only env var read here
        │
        ├─ Parameter Store    GetParametersByPath(prefix)     every plain setting
        ├─ Secrets Manager    <openai_secret_id>             {"api_key", "model"}
        └─ Prompt Management  <prompt_id> @ <prompt_version>  the system prompt

WHY THREE SOURCES

    Parameter Store     settings an operator changes without a rebuild
    Secrets Manager     the one credential: rotated and audited, never an
                        environment variable (visible to anyone who can
                        describe the runtime)
    Prompt Management   the prompt is versioned. The version is PINNED in
                        Parameter Store, so editing the prompt changes nothing
                        in production until prompt_version is bumped — and
                        rolling back is changing one number.

FAIL AT START, NOT AT THE FIRST QUESTION

A missing parameter, an unset secret, or an unrendered {{variable}} raises
here, while the container starts. AgentCore then reports the start failure
in /aws/bedrock-agentcore/runtimes/. Failing on the first analyst question
instead would look like a model error.

WHAT THIS DOES NOT DO

    It does not reload. Settings are read once per container; a change takes
    effect on the next cold start or redeploy.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

REQUIRED = ('openai_secret_id', 'prompt_id', 'prompt_version', 'guardrail_id', 'guardrail_version', 'gateway_url', 'max_resolve_calls', 'max_searches_per_turn', 'max_neighbor_calls', 'max_table_calls', 'max_window', 'expansion_token_budget',)


@dataclass(frozen=True)
class Settings:
    region: str
    openai_api_key: str = field(repr=False)
    openai_model: str
    system_prompt: str
    prompt_version: str
    guardrail_id: str
    guardrail_version: str
    gateway_url: str
    max_resolve_calls: int
    max_searches_per_turn: int
    max_neighbor_calls: int
    max_table_calls: int
    max_window: int
    expansion_token_budget: int

    def chat_model(self):
        """The OpenAI chat model, over the RESPONSES API.

        ChatOpenAI calls Chat Completions by default. OpenAI's models page
        lists the latest models (GPT-6) as available via the Responses API
        and does not mention Chat Completions, so a GPT-6 model could fail on
        every call through the default endpoint. The Responses API serves
        every current model, so it is used unconditionally.

        Temperature is left at the provider default: reasoning models reject
        a temperature argument outright.
        """
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.openai_model, api_key=self.openai_api_key,
                          use_responses_api=True, timeout=120, max_retries=2)


def _parameters(ssm, path: str) -> dict[str, str]:
    """Every parameter under `path`, keyed by the name after the path."""
    found, token = {}, None
    while True:
        page = ssm.get_parameters_by_path(Path=path, Recursive=True,
                                          **({"NextToken": token} if token else {}))
        for p in page["Parameters"]:
            found[p["Name"][len(path):].lstrip("/")] = p["Value"]
        token = page.get("NextToken")
        if not token:
            return found


def _prompt(agent_client, prompt_id: str, version: str) -> str:
    response = agent_client.get_prompt(promptIdentifier=prompt_id, promptVersion=version)
    variants = response["variants"]
    variant = next((v for v in variants if v["name"] == response.get("defaultVariant")),
                   variants[0])
    return variant["templateConfiguration"]["text"]["text"]


def render(template: str, variables: dict[str, str]) -> str:
    """Fill {{name}} variables. A variable left unfilled is an error, not text
    the model reads literally."""
    text = template
    for name, value in variables.items():
        text = text.replace("{{" + name + "}}", value)
    left = sorted(set(re.findall(r"{{\s*(\w+)\s*}}", text)))
    if left:
        raise RuntimeError(f"system prompt has unrendered variables: {left}")
    return text


def load(env=os.environ, session=None) -> Settings:
    import boto3
    session = session or boto3.Session()
    region = env.get("AWS_REGION") or session.region_name or "us-east-1"
    prefix = env["PARAM_PREFIX"].rstrip("/")

    # STEP 1 — plain settings
    params = _parameters(session.client("ssm", region_name=region), prefix)
    missing = [k for k in REQUIRED if k not in params]
    if missing:
        raise RuntimeError(f"missing parameters under {prefix}: {missing}")

    # STEP 2 — the credential
    secret = json.loads(session.client("secretsmanager", region_name=region)
                        .get_secret_value(SecretId=params["openai_secret_id"])["SecretString"])
    unset = [k for k in ("api_key", "model") if secret.get(k, "replace-me") == "replace-me"]
    if unset:
        raise RuntimeError(f"{params['openai_secret_id']} still holds placeholder {unset}")

    # STEP 3 — the prompt, at its pinned version
    template = _prompt(session.client("bedrock-agent", region_name=region),
                       params["prompt_id"], params["prompt_version"])

    return Settings(
        region=region, openai_api_key=secret["api_key"], openai_model=secret["model"],
        system_prompt=render(template, {
            "max_resolve_calls": params["max_resolve_calls"],
            "max_searches_per_turn": params["max_searches_per_turn"],
            "max_neighbor_calls": params["max_neighbor_calls"],
            "max_table_calls": params["max_table_calls"],
            "expansion_token_budget": params["expansion_token_budget"]}),
        prompt_version=params["prompt_version"],
        guardrail_id=params["guardrail_id"], guardrail_version=params["guardrail_version"],
        gateway_url=params["gateway_url"],
        max_resolve_calls=int(params["max_resolve_calls"]),
        max_searches_per_turn=int(params["max_searches_per_turn"]),
        max_neighbor_calls=int(params["max_neighbor_calls"]),
        max_table_calls=int(params["max_table_calls"]),
        max_window=int(params["max_window"]),
        expansion_token_budget=int(params["expansion_token_budget"]),)


_current: Settings | None = None


def settings() -> Settings:
    """The loaded settings; loaded on first use."""
    global _current
    if _current is None:
        _current = load()
    return _current


def use(value: Settings) -> None:
    """Install settings directly — tests, and nothing else."""
    global _current
    _current = value
