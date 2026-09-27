"""The AWS-backed config loaders: every failure must happen at load, loudly."""
import json

import pytest

import fakes

assert fakes.ROOT.exists()   # importing fakes puts each agent_code on sys.path
from supervisor import config as sup_config
from trial_search import config as ts_config


class Session:
    """boto3.Session stand-in. Parameter pages of 2 exercise NextToken."""

    def __init__(self, params: dict, secret: dict, prompts: dict):
        self.params, self.secret, self.prompts = params, secret, prompts
        self.region_name = "us-east-1"

    def client(self, name, region_name=None):
        return {"ssm": SSM(self.params), "secretsmanager": Secrets(self.secret),
                "bedrock-agent": Agent(self.prompts)}[name]


class SSM:
    def __init__(self, params): self.params = params

    def get_parameters_by_path(self, Path, Recursive=True, NextToken=None):
        items = sorted((k, v) for k, v in self.params.items() if k.startswith(Path + "/"))
        start = int(NextToken or 0)
        page = items[start:start + 2]
        out = {"Parameters": [{"Name": k, "Value": v} for k, v in page]}
        if start + 2 < len(items):
            out["NextToken"] = str(start + 2)
        return out


class Secrets:
    def __init__(self, secret): self.secret = secret

    def get_secret_value(self, SecretId):
        return {"SecretString": json.dumps(self.secret)}


class Agent:
    def __init__(self, prompts): self.prompts = prompts

    def get_prompt(self, promptIdentifier, promptVersion):
        text = self.prompts[(promptIdentifier, promptVersion)]
        return {"defaultVariant": "default", "variants": [
            {"name": "default", "templateConfiguration": {"text": {"text": text}}}]}


TS = "/trial-agents/trial_search"
TS_PARAMS = {f"{TS}/{k}": v for k, v in {
    "gateway_url": "https://gw", "openai_secret_id": "trial-agents/openai",
    "prompt_id": "P1", "prompt_version": "3", "guardrail_id": "G", "guardrail_version": "2",
    "max_searches_per_turn": "5", "max_neighbor_calls": "3", "max_table_calls": "3",
    "max_window": "10", "expansion_token_budget": "6000"}.items()}
TS_PROMPT = ("Budgets: {{max_searches_per_turn}} searches, {{max_neighbor_calls}} neighbours, "
             "{{max_table_calls}} tables, {{expansion_token_budget}} tokens.")
SECRET = {"api_key": "sk-real", "model": "gpt-real"}


def load_ts(params=TS_PARAMS, secret=SECRET, prompt=TS_PROMPT):
    return ts_config.load(env={"PARAM_PREFIX": TS},
                          session=Session(params, secret, {("P1", "3"): prompt}))


def test_trial_search_loads_and_renders_budgets():
    s = load_ts()
    assert (s.max_window, s.expansion_token_budget, s.openai_model) == (10, 6000, "gpt-real")
    assert s.system_prompt == "Budgets: 5 searches, 3 neighbours, 3 tables, 6000 tokens."
    assert s.prompt_version == "3" and s.guardrail_version == "2"
    assert "sk-real" not in repr(s), "the API key must not appear in repr()"


def test_missing_parameter_is_named():
    params = {k: v for k, v in TS_PARAMS.items() if not k.endswith("/max_window")}
    with pytest.raises(RuntimeError, match="max_window"):
        load_ts(params=params)


@pytest.mark.parametrize("secret", [{"api_key": "replace-me", "model": "gpt"},
                                    {"api_key": "sk", "model": "replace-me"}])
def test_placeholder_secret_refused(secret):
    with pytest.raises(RuntimeError, match="placeholder"):
        load_ts(secret=secret)


def test_unrendered_variable_refused():
    with pytest.raises(RuntimeError, match="unrendered variables.*undeclared"):
        load_ts(prompt=TS_PROMPT + " {{undeclared}}")


SUP = "/trial-agents/supervisor"
REG = "/trial-agents/registry"
SUP_PARAMS = {
    **{f"{SUP}/{k}": v for k, v in {
        "openai_secret_id": "trial-agents/openai", "prompt_id": "S", "prompt_version": "1",
        "compose_prompt_id": "C", "compose_prompt_version": "4",
        "guardrail_id": "G", "guardrail_version": "2", "registry_path": REG,
        "max_agent_calls_per_turn": "6"}.items()},
    f"{REG}/trial_graph": json.dumps({"arn": "arn:g", "description": "graph things"}),
    f"{REG}/trial_search": json.dumps({"arn": "arn:s", "description": "text things"}),
}


def load_sup(params=SUP_PARAMS, compose="{{question}} / {{evidence}}"):
    prompts = {("S", "1"): "Agents:\n{{available_agents}}\nLimit {{max_agent_calls_per_turn}}",
               ("C", "4"): compose}
    return sup_config.load(env={"PARAM_PREFIX": SUP}, session=Session(params, SECRET, prompts))


def test_supervisor_reads_registry_and_renders_agents():
    s = load_sup()
    assert set(s.specialists) == {"trial_graph", "trial_search"}
    assert s.specialists["trial_graph"]["arn"] == "arn:g"
    assert "- trial_graph: graph things" in s.system_prompt
    assert "- trial_search: text things" in s.system_prompt
    assert "Limit 6" in s.system_prompt
    assert s.compose_template == "{{question}} / {{evidence}}"   # rendered per question


def test_supervisor_refuses_empty_registry():
    params = {k: v for k, v in SUP_PARAMS.items() if not k.startswith(REG)}
    with pytest.raises(RuntimeError, match="no specialists registered"):
        load_sup(params=params)


def test_compose_prompt_must_take_the_evidence():
    with pytest.raises(RuntimeError, match="evidence"):
        load_sup(compose="{{question}} only")
