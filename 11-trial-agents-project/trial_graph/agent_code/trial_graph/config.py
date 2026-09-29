"""trial_graph configuration — loaded once, at container start, from AWS.

CONFIGURATION FLOW
------------------

The agent reads only one environment variable:

    PARAM_PREFIX = /trial-agents/trial_graph

That prefix tells the application where its configuration lives in
AWS Systems Manager Parameter Store.

From there, configuration is assembled from three AWS services:

    PARAM_PREFIX
         │
         ▼
    Parameter Store
         │
         ├── openai_secret_id
         ├── prompt_id
         ├── prompt_version
         ├── guardrail_id
         ├── guardrail_version
         ├── gateway_url
         ├── row_cap
         ├── graph_node_cap
         └── max_repairs
              │
              ├──────────────────────────┐
              ▼                          ▼
    Secrets Manager              Bedrock Prompt Management
    <openai_secret_id>            <prompt_id>@<prompt_version>
              │                          │
              ├── api_key                └── system prompt
              └── model
              │
              ▼
        Settings object
              │
              ▼
       trial_graph agent


WHY THREE AWS SOURCES
---------------------

1. PARAMETER STORE

Used for normal operational configuration.

Examples:

    row_cap
    graph_node_cap
    max_repairs
    gateway_url
    prompt_version

These values can be changed without rebuilding the container image.

The changes take effect on the next container start/redeploy because
this module loads configuration only once.


2. SECRETS MANAGER

Used for the OpenAI credential and model configuration.

The API key is intentionally NOT stored in:

    - environment variables
    - source code
    - Parameter Store

Secrets Manager provides controlled access, rotation, and auditing.

The Settings dataclass also hides the API key from its repr() output.


3. BEDROCK PROMPT MANAGEMENT

The system prompt is stored and versioned separately from the code.

Parameter Store contains the pinned prompt version:

    prompt_version = "5"

The application then asks Prompt Management for exactly:

    prompt_id + prompt_version

This means changing the prompt in Prompt Management does not
automatically change the production behavior.

To move production to a new prompt, the operator changes the pinned
version and restarts/redeploys the runtime.

Rolling back therefore means changing the version back.


FAIL FAST AT STARTUP
--------------------

Configuration errors are detected while the container is starting.

Examples:

    - missing Parameter Store parameter
    - missing OpenAI secret
    - placeholder secret still present
    - invalid/missing prompt version
    - unrendered {{variable}} in the system prompt

This is preferable to discovering the problem when the first analyst
question arrives.

AgentCore can then report the container startup failure through the
runtime logs.


WHAT THIS MODULE DOES NOT DO
----------------------------

Configuration is NOT dynamically reloaded.

The lifecycle is:

    container starts
          │
          ▼
    settings()
          │
          ▼
        load()
          │
          ▼
    _current = Settings(...)
          │
          ▼
    reused for all requests

Therefore, changing AWS configuration does not change an already-running
container.

The new configuration takes effect on:

    - a cold start
    - a new container
    - a redeployment
"""


from __future__ import annotations


# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

# Used to parse the JSON object stored in Secrets Manager.
import json

# Used to read PARAM_PREFIX and AWS_REGION from the environment.
import os

# Used to detect any remaining {{variable}} placeholders in the system prompt.
import re

# Dataclass provides the immutable Settings configuration object.
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Required Parameter Store settings
# ---------------------------------------------------------------------------

# Every one of these values must exist underneath PARAM_PREFIX.
#
# Example:
#
#     PARAM_PREFIX=/trial-agents/trial_graph
#
# Parameter Store might contain:
#
#     /trial-agents/trial_graph/openai_secret_id
#     /trial-agents/trial_graph/prompt_id
#     /trial-agents/trial_graph/prompt_version
#     ...
#
# If even one required parameter is missing, load() fails immediately.
REQUIRED = (
    "openai_secret_id",
    "prompt_id",
    "prompt_version",
    "guardrail_id",
    "guardrail_version",
    "gateway_url",
    "row_cap",
    "graph_node_cap",
    "max_repairs",
)


# ---------------------------------------------------------------------------
# Immutable runtime configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """All configuration required by trial_graph.

    `frozen=True` makes the object immutable after creation.

    That is intentional because configuration should not unexpectedly
    change while the agent is processing requests.

    The API key uses:

        field(repr=False)

    so printing/logging the Settings object will not expose the secret.
    """

    # AWS region used by AWS SDK clients.
    region: str

    # OpenAI API key.
    #
    # repr=False prevents this value from appearing in:
    #
    #     repr(settings)
    #
    # or similar debug output.
    openai_api_key: str = field(
        repr=False
    )

    # OpenAI model name loaded from Secrets Manager.
    openai_model: str

    # Fully rendered system prompt retrieved from Bedrock Prompt Management.
    system_prompt: str

    # Prompt Management version that was pinned through Parameter Store.
    prompt_version: str

    # Bedrock Guardrail identifier.
    guardrail_id: str

    # Bedrock Guardrail version.
    guardrail_version: str

    # AgentCore Gateway endpoint used by the MCP client.
    gateway_url: str

    # Maximum number of table rows allowed to be returned/captured.
    row_cap: int

    # Maximum number of graph nodes allowed to be returned/captured.
    graph_node_cap: int

    # Maximum number of Cypher repair attempts allowed.
    max_repairs: int

    def chat_model(self):
        """Create the OpenAI LangChain chat model.

        The model configuration is created from the values loaded into
        Settings.

        IMPORTANT:
        ----------
        `use_responses_api=True` explicitly tells ChatOpenAI to use the
        OpenAI Responses API.

        This avoids relying on the provider's default endpoint.

        Temperature is intentionally NOT supplied.

        Reasoning models may reject an explicit temperature argument, so
        leaving it at the provider default keeps the configuration compatible
        with reasoning models as well.
        """

        # Import lazily.
        #
        # This prevents LangChain/OpenAI imports from happening simply when
        # this configuration module is imported.
        from langchain_openai import ChatOpenAI

        # Return a configured LangChain ChatOpenAI instance.
        return ChatOpenAI(
            model=self.openai_model,
            api_key=self.openai_api_key,

            # Explicitly use OpenAI's Responses API.
            use_responses_api=True,

            # Prevent an individual model request from hanging indefinitely.
            timeout=120,

            # Retry transient provider failures up to two times.
            max_retries=2,
        )


# ---------------------------------------------------------------------------
# Parameter Store
# ---------------------------------------------------------------------------

def _parameters(ssm, path: str) -> dict[str, str]:
    """Load every parameter below `path`.

    Parameters are loaded using AWS Systems Manager Parameter Store's
    GetParametersByPath API.

    The API is paginated, so this function continues requesting pages
    until AWS stops returning NextToken.

    Returned dictionary keys contain only the parameter name relative
    to the supplied path.

    Example:

        path:
            /trial-agents/trial_graph

        AWS parameter:
            /trial-agents/trial_graph/row_cap

        returned key:
            row_cap
    """

    # Accumulator for all parameters discovered under the path.
    #
    # `token` is the pagination token returned by AWS.
    found, token = {}, None

    while True:

        # Fetch one page of parameters.
        #
        # Recursive=True means parameters under nested paths are also
        # included.
        #
        # NextToken is included only when AWS supplied one previously.
        page = ssm.get_parameters_by_path(
            Path=path,
            Recursive=True,
            **(
                {"NextToken": token}
                if token
                else {}
            ),
        )

        # Convert the full AWS parameter name into a relative name.
        #
        # Example:
        #
        #     /trial-agents/trial_graph/row_cap
        #
        # becomes:
        #
        #     row_cap
        for p in page["Parameters"]:

            found[
                p["Name"][len(path):].lstrip("/")
            ] = p["Value"]

        # Check whether another page exists.
        token = page.get(
            "NextToken"
        )

        # No token means all parameters have been retrieved.
        if not token:
            return found


# ---------------------------------------------------------------------------
# Bedrock Prompt Management
# ---------------------------------------------------------------------------

def _prompt(
    agent_client,
    prompt_id: str,
    version: str,
) -> str:
    """Retrieve the pinned system prompt from Bedrock Prompt Management.

    `prompt_id` identifies the prompt.

    `version` identifies the exact version that should be used.

    This is intentionally version-specific so production does not
    unexpectedly change when somebody edits a prompt in Prompt Management.
    """

    # Request the specific prompt version from Bedrock Agent APIs.
    response = agent_client.get_prompt(
        promptIdentifier=prompt_id,
        promptVersion=version,
    )

    # A prompt can contain multiple variants.
    variants = response[
        "variants"
    ]

    # Prefer the variant marked as the default variant.
    #
    # If no defaultVariant is returned, fall back to the first variant.
    variant = next(
        (
            v
            for v in variants
            if v["name"] == response.get(
                "defaultVariant"
            )
        ),
        variants[0],
    )

    # Extract the actual text template from the selected variant.
    return variant[
        "templateConfiguration"
    ][
        "text"
    ][
        "text"
    ]


# ---------------------------------------------------------------------------
# Prompt variable rendering
# ---------------------------------------------------------------------------

def render(
    template: str,
    variables: dict[str, str],
) -> str:
    """Render {{name}} variables inside the system prompt.

    Example:

        template:
            "You are {{agent_name}}."

        variables:
            {"agent_name": "Trial Graph"}

        result:
            "You are Trial Graph."

    IMPORTANT:
    ----------
    An unresolved variable is treated as a configuration error.

    We do NOT allow:

        "You are {{agent_name}}."

    to reach the model accidentally.

    Such a placeholder could indicate an incomplete deployment or
    incorrectly configured prompt.
    """

    # Start with the original prompt template.
    text = template

    # Replace every explicitly supplied variable.
    for name, value in variables.items():

        text = text.replace(
            "{{" + name + "}}",
            value,
        )

    # Look for any {{variable}} placeholders that remain.
    #
    # The regex allows optional whitespace:
    #
    #     {{name}}
    #     {{ name }}
    #
    # Both are detected.
    left = sorted(
        set(
            re.findall(
                r"{{\s*(\w+)\s*}}",
                text,
            )
        )
    )

    # Any remaining placeholder means the prompt was not completely
    # rendered.
    if left:
        raise RuntimeError(
            f"system prompt has unrendered variables: {left}"
        )

    # Return the fully rendered prompt.
    return text


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load(
    env=os.environ,
    session=None,
) -> Settings:
    """Load the complete trial_graph configuration from AWS.

    Configuration is assembled in three stages:

        STEP 1
            Parameter Store
            -> normal settings

        STEP 2
            Secrets Manager
            -> OpenAI API key and model

        STEP 3
            Bedrock Prompt Management
            -> pinned system prompt

    If anything required is missing or invalid, this function raises
    immediately.
    """

    # Import boto3 lazily so importing this module alone does not require
    # creating AWS clients.
    import boto3

    # Allow tests to inject a fake boto3 session.
    #
    # In production, create a normal boto3.Session().
    session = session or boto3.Session()

    # Determine the AWS region.
    #
    # Priority:
    #
    #     1. AWS_REGION environment variable
    #     2. boto3 session region
    #     3. us-east-1 fallback
    region = (
        env.get("AWS_REGION")
        or session.region_name
        or "us-east-1"
    )

    # PARAM_PREFIX is the only required environment variable.
    #
    # Remove a trailing slash so path concatenation/extraction remains
    # consistent.
    prefix = env[
        "PARAM_PREFIX"
    ].rstrip("/")


    # =======================================================================
    # STEP 1 — Plain operational settings
    # =======================================================================

    # Create an AWS Systems Manager Parameter Store client.
    ssm = session.client(
        "ssm",
        region_name=region,
    )

    # Load every parameter underneath PARAM_PREFIX.
    params = _parameters(
        ssm,
        prefix,
    )

    # Verify that every required setting exists.
    #
    # This is intentionally done before attempting to read the secret or
    # prompt so configuration errors are reported clearly.
    missing = [
        key
        for key in REQUIRED
        if key not in params
    ]

    if missing:
        raise RuntimeError(
            f"missing parameters under {prefix}: {missing}"
        )


    # =======================================================================
    # STEP 2 — OpenAI credential and model
    # =======================================================================

    # Create the Secrets Manager client.
    secretsmanager = session.client(
        "secretsmanager",
        region_name=region,
    )

    # Retrieve the secret referenced by Parameter Store.
    #
    # Parameter Store therefore contains the SECRET ID, while the actual
    # credential remains in Secrets Manager.
    secret = json.loads(
        secretsmanager.get_secret_value(
            SecretId=params[
                "openai_secret_id"
            ]
        )[
            "SecretString"
        ]
    )

    # Detect placeholder values.
    #
    # A deployment containing:
    #
    #     "api_key": "replace-me"
    #
    # or:
    #
    #     "model": "replace-me"
    #
    # should fail during startup rather than when the first question
    # reaches the OpenAI model.
    unset = [
        key
        for key in (
            "api_key",
            "model",
        )
        if secret.get(
            key,
            "replace-me",
        ) == "replace-me"
    ]

    if unset:
        raise RuntimeError(
            f"{params['openai_secret_id']} still holds placeholder {unset}"
        )


    # =======================================================================
    # STEP 3 — Version-pinned system prompt
    # =======================================================================

    # Create the Bedrock Agent client used for Prompt Management.
    agent_client = session.client(
        "bedrock-agent",
        region_name=region,
    )

    # Retrieve exactly the prompt version specified in Parameter Store.
    #
    # This creates the important deployment boundary:
    #
    #     Parameter Store
    #          │
    #          └── prompt_version = X
    #                         │
    #                         ▼
    #                 Prompt Management
    #                         │
    #                         ▼
    #                   exact version X
    template = _prompt(
        agent_client,
        params["prompt_id"],
        params["prompt_version"],
    )


    # =======================================================================
    # Build immutable Settings object
    # =======================================================================

    # Convert string parameters from Parameter Store into their required
    # Python types.
    #
    # Parameter Store returns values as strings, so numeric limits need
    # explicit conversion.
    return Settings(
        region=region,

        # Credential loaded from Secrets Manager.
        openai_api_key=secret["api_key"],

        # Model loaded from Secrets Manager.
        openai_model=secret["model"],

        # Render the prompt.
        #
        # An empty variable dictionary is currently passed because this
        # deployment does not inject runtime prompt variables here.
        #
        # render() still verifies that no {{variable}} placeholders remain.
        system_prompt=render(
            template,
            {},
        ),

        # Pin the exact prompt version used by this runtime.
        prompt_version=params[
            "prompt_version"
        ],

        # Guardrail configuration.
        guardrail_id=params[
            "guardrail_id"
        ],

        guardrail_version=params[
            "guardrail_version"
        ],

        # AgentCore Gateway endpoint used by the MCP connection.
        gateway_url=params[
            "gateway_url"
        ],

        # Maximum number of table rows.
        row_cap=int(
            params["row_cap"]
        ),

        # Maximum number of graph nodes.
        graph_node_cap=int(
            params["graph_node_cap"]
        ),

        # Maximum number of Cypher repairs.
        max_repairs=int(
            params["max_repairs"]
        ),
    )


# ---------------------------------------------------------------------------
# Process-local settings cache
# ---------------------------------------------------------------------------

# Holds the Settings instance after the first successful load().
#
# None means configuration has not been loaded yet.
_current: Settings | None = None


def settings() -> Settings:
    """Return the loaded settings.

    This function implements lazy, process-local configuration caching.

    First call:

        settings()
            │
            ▼
        _current is None
            │
            ▼
        load()
            │
            ▼
        Settings
            │
            ▼
        _current

    Subsequent calls:

        settings()
            │
            ▼
        _current
            │
            ▼
        return immediately

    Therefore AWS services are not queried for every analyst question.
    """

    global _current

    # Configuration has not been loaded yet.
    if _current is None:

        # Load it once and cache it.
        _current = load()

    # Return the cached immutable Settings object.
    return _current


def use(
    value: Settings,
) -> None:
    """Install a Settings object directly.

    This exists primarily for tests.

    Tests can construct a Settings object with known values and inject it
    without making real calls to:

        - Parameter Store
        - Secrets Manager
        - Bedrock Prompt Management

    This keeps unit tests deterministic and fast.

    It should not be used as the normal production configuration path.
    """

    global _current

    # Replace the process-local cached configuration.
    _current = value