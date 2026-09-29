"""trial_search configuration — loaded once, at container start, from AWS.

CONFIGURATION FLOW
------------------

The runtime exposes only one environment variable:

    PARAM_PREFIX=/trial-agents/trial_search

Everything else is loaded from AWS.

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
         ├── max_searches_per_turn
         ├── max_neighbor_calls
         ├── max_table_calls
         ├── max_window
         └── expansion_token_budget
              │
              ├──────────────────────────────┐
              ▼                              ▼
    Secrets Manager                  Prompt Management
    <openai_secret_id>               <prompt_id>@<prompt_version>
              │                              │
              ├── api_key                    └── system prompt
              └── model
              │
              ▼
         Settings object
              │
              ▼
       trial_search agent


WHY THREE CONFIGURATION SOURCES
-------------------------------

1. PARAMETER STORE

Stores normal operational settings.

Examples:

    max_searches_per_turn
    max_neighbor_calls
    max_table_calls
    max_window
    expansion_token_budget
    gateway_url
    prompt_version

These values can be changed without rebuilding the Docker image.

The new values take effect when a new runtime container starts.


2. SECRETS MANAGER

Stores the OpenAI credential and model configuration.

The API key is deliberately not stored in:

    - source code
    - environment variables
    - Parameter Store

The runtime receives only the secret identifier from Parameter Store
and retrieves the actual secret from Secrets Manager.

The Settings dataclass also hides the API key from repr() output.


3. PROMPT MANAGEMENT

The system prompt is stored and versioned in Bedrock Prompt Management.

Parameter Store contains the exact prompt version to use.

For example:

    prompt_version = 7

The runtime therefore loads:

    prompt_id + version 7

Editing a prompt does not automatically change the running production
configuration.

The version must be explicitly changed and the runtime restarted/redeployed.


FAIL FAST AT STARTUP
--------------------

Configuration errors are detected during container startup.

Examples:

    - missing Parameter Store setting
    - missing OpenAI secret
    - placeholder API key
    - placeholder model
    - missing prompt
    - unrendered {{variable}} in the prompt

This is preferable to discovering configuration problems when the first
analyst question arrives.

AgentCore can then report the runtime startup failure.


WHAT THIS MODULE DOES NOT DO
----------------------------

Configuration is not dynamically reloaded.

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
    Settings(...)
          │
          ▼
       _current
          │
          ▼
    reused for requests

Therefore a Parameter Store, Secrets Manager, or Prompt Management
change affects new containers/cold starts rather than an already-loaded
Settings object.
"""


# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# Used to parse the JSON object stored in Secrets Manager.
import json

# Used to read AWS_REGION and PARAM_PREFIX from the environment.
import os

# Used to detect remaining {{variable}} placeholders in the prompt.
import re

# Used to create the immutable Settings configuration object.
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Required Parameter Store settings
# ---------------------------------------------------------------------------

# Every value listed here must exist underneath PARAM_PREFIX.
#
# Example:
#
#     /trial-agents/trial_search/openai_secret_id
#     /trial-agents/trial_search/prompt_id
#     /trial-agents/trial_search/max_window
#
# If any setting is missing, load() fails immediately.
REQUIRED = (
    "openai_secret_id",
    "prompt_id",
    "prompt_version",
    "guardrail_id",
    "guardrail_version",
    "gateway_url",
    "max_searches_per_turn",
    "max_neighbor_calls",
    "max_table_calls",
    "max_window",
    "expansion_token_budget",
)


# ===========================================================================
# Immutable runtime settings
# ===========================================================================

@dataclass(frozen=True)
class Settings:
    """Complete runtime configuration for trial_search.

    `frozen=True` makes the configuration immutable after it has been
    loaded.

    This is useful for agent configuration because the values should not
    unexpectedly change in the middle of an agent execution.

    The OpenAI API key uses:

        field(repr=False)

    so printing the Settings object does not expose the credential.
    """

    # AWS region used to create AWS SDK clients.
    region: str

    # OpenAI API key.
    #
    # repr=False prevents the secret from appearing in:
    #
    #     repr(settings)
    #
    # or similar debug output.
    openai_api_key: str = field(
        repr=False
    )

    # OpenAI model name loaded from Secrets Manager.
    openai_model: str

    # Fully rendered system prompt retrieved from Prompt Management.
    system_prompt: str

    # Exact Prompt Management version used by this runtime.
    prompt_version: str

    # Bedrock Guardrail identifier.
    guardrail_id: str

    # Bedrock Guardrail version.
    guardrail_version: str

    # AgentCore Gateway endpoint used by trial_search tools.
    gateway_url: str

    # Maximum number of independent searches allowed in one turn.
    #
    # This limits how aggressively the search agent can expand its search
    # activity.
    max_searches_per_turn: int

    # Maximum number of graph-neighbor expansion calls.
    max_neighbor_calls: int

    # Maximum number of table-related calls.
    max_table_calls: int

    # Maximum search window/range used by the search tools.
    max_window: int

    # Maximum token budget allocated to search expansion.
    expansion_token_budget: int


    def chat_model(self):
        """Create the configured OpenAI LangChain chat model.

        The model is accessed through OpenAI's Responses API.

        The configuration intentionally does not specify `temperature`.

        Reasoning models may reject an explicit temperature parameter,
        so the provider default is used.

        The model also gets:

            timeout=120
            max_retries=2

        to provide bounded execution and limited retry behavior.
        """

        # Import lazily so importing this configuration module does not
        # immediately import LangChain/OpenAI dependencies.
        from langchain_openai import ChatOpenAI

        # Build the ChatOpenAI instance from the immutable Settings object.
        return ChatOpenAI(
            model=self.openai_model,
            api_key=self.openai_api_key,

            # Explicitly use the OpenAI Responses API.
            use_responses_api=True,

            # Maximum time allowed for an individual model request.
            timeout=120,

            # Retry transient provider failures twice.
            max_retries=2,
        )


# ===========================================================================
# Parameter Store
# ===========================================================================

def _parameters(
    ssm,
    path: str,
) -> dict[str, str]:
    """Load every Parameter Store value underneath `path`.

    AWS Systems Manager's GetParametersByPath API is paginated.

    This function therefore continues requesting pages until AWS no
    longer returns a NextToken.

    The returned dictionary uses the parameter name relative to `path`.

    Example:

        path:
            /trial-agents/trial_search

        AWS parameter:
            /trial-agents/trial_search/max_window

        returned dictionary:
            {
                "max_window": "50"
            }
    """

    # Dictionary containing all discovered parameters.
    #
    # token stores AWS's pagination token.
    found, token = {}, None

    while True:

        # Request one page of parameters.
        #
        # Recursive=True allows nested parameters below the prefix.
        #
        # Only send NextToken when AWS actually returned one.
        page = ssm.get_parameters_by_path(
            Path=path,
            Recursive=True,
            **(
                {
                    "NextToken": token
                }
                if token
                else {}
            ),
        )

        # Convert each full AWS parameter path into a relative key.
        #
        # Example:
        #
        #     /trial-agents/trial_search/max_window
        #
        # becomes:
        #
        #     max_window
        for p in page["Parameters"]:

            found[
                p["Name"][len(path):].lstrip("/")
            ] = p["Value"]

        # Get the token for the next page.
        token = page.get(
            "NextToken"
        )

        # No token means all parameters have been loaded.
        if not token:
            return found


# ===========================================================================
# Prompt Management
# ===========================================================================

def _prompt(
    agent_client,
    prompt_id: str,
    version: str,
) -> str:
    """Retrieve the version-pinned system prompt.

    `prompt_id` identifies the prompt.

    `version` identifies the exact version to retrieve.

    This prevents production from silently switching prompts when a
    newer version is created in Prompt Management.
    """

    # Retrieve the requested prompt version.
    response = agent_client.get_prompt(
        promptIdentifier=prompt_id,
        promptVersion=version,
    )

    # Prompt Management can contain multiple variants.
    variants = response[
        "variants"
    ]

    # Prefer the variant marked as the default variant.
    #
    # If no default variant is specified, use the first variant.
    variant = next(
        (
            v
            for v in variants
            if v["name"]
            == response.get(
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


# ===========================================================================
# Prompt variable rendering
# ===========================================================================

def render(
    template: str,
    variables: dict[str, str],
) -> str:
    """Replace {{name}} placeholders in the system prompt.

    Example:

        Template:

            "Use at most {{max_window}} results."

        Variables:

            {
                "max_window": "50"
            }

        Result:

            "Use at most 50 results."


    IMPORTANT
    ---------

    An unresolved variable is treated as a configuration error.

    We do not allow a literal placeholder such as:

        {{max_window}}

    to accidentally reach the model.
    """

    # Start with the prompt template returned by Prompt Management.
    text = template

    # Replace each configured variable.
    for name, value in variables.items():

        text = text.replace(
            "{{" + name + "}}",
            value,
        )

    # Find any {{variable}} placeholders that remain.
    #
    # Both of these are detected:
    #
    #     {{name}}
    #     {{ name }}
    left = sorted(
        set(
            re.findall(
                r"{{\s*(\w+)\s*}}",
                text,
            )
        )
    )

    # Any remaining variable means the prompt was not fully configured.
    if left:

        raise RuntimeError(
            f"system prompt has unrendered variables: {left}"
        )

    # Return the fully rendered system prompt.
    return text


# ===========================================================================
# Load configuration
# ===========================================================================

def load(
    env=os.environ,
    session=None,
) -> Settings:
    """Load trial_search configuration from AWS.

    Configuration is assembled from three sources:

        Parameter Store
            -> operational configuration

        Secrets Manager
            -> OpenAI API key and model

        Prompt Management
            -> version-pinned system prompt

    The function intentionally fails immediately if required configuration
    is missing.
    """

    # Import boto3 lazily.
    #
    # This makes the module easier to import in tests and avoids creating
    # AWS dependencies until configuration is actually loaded.
    import boto3

    # Allow callers/tests to provide their own boto3 session.
    #
    # In production, create a normal AWS session.
    session = session or boto3.Session()


    # -----------------------------------------------------------------------
    # Determine AWS region
    # -----------------------------------------------------------------------

    # Region priority:
    #
    #     1. AWS_REGION environment variable
    #     2. boto3 session region
    #     3. us-east-1 fallback
    region = (
        env.get("AWS_REGION")
        or session.region_name
        or "us-east-1"
    )


    # -----------------------------------------------------------------------
    # Determine Parameter Store prefix
    # -----------------------------------------------------------------------

    # PARAM_PREFIX is the only required runtime environment variable.
    #
    # Example:
    #
    #     /trial-agents/trial_search
    #
    # Remove a trailing slash to keep path handling consistent.
    prefix = env[
        "PARAM_PREFIX"
    ].rstrip("/")


    # =======================================================================
    # STEP 1 — Plain operational settings
    # =======================================================================

    # Create the Parameter Store client.
    ssm = session.client(
        "ssm",
        region_name=region,
    )

    # Load every parameter underneath the trial_search prefix.
    params = _parameters(
        ssm,
        prefix,
    )

    # Verify that every required parameter exists.
    missing = [
        key
        for key in REQUIRED
        if key not in params
    ]

    # Fail immediately if configuration is incomplete.
    if missing:

        raise RuntimeError(
            f"missing parameters under {prefix}: {missing}"
        )


    # =======================================================================
    # STEP 2 — OpenAI credentials
    # =======================================================================

    # Create the Secrets Manager client.
    secretsmanager = session.client(
        "secretsmanager",
        region_name=region,
    )

    # Parameter Store contains the secret ID.
    #
    # Secrets Manager contains the actual:
    #
    #     api_key
    #     model
    #
    # This keeps the credential itself out of Parameter Store.
    secret = json.loads(
        secretsmanager.get_secret_value(
            SecretId=params[
                "openai_secret_id"
            ]
        )[
            "SecretString"
        ]
    )

    # Detect placeholder credentials.
    #
    # A new deployment may intentionally create a secret containing
    # "replace-me".
    #
    # The runtime must not start with those values.
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

    # Fail during startup instead of allowing the first search request
    # to fail later.
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
    template = _prompt(
        agent_client,
        params["prompt_id"],
        params["prompt_version"],
    )


    # =======================================================================
    # Build final immutable Settings object
    # =======================================================================

    return Settings(

        # AWS region used by the application.
        region=region,

        # OpenAI credential loaded from Secrets Manager.
        openai_api_key=secret[
            "api_key"
        ],

        # OpenAI model loaded from Secrets Manager.
        openai_model=secret[
            "model"
        ],

        # Render the prompt using search-specific limits.
        #
        # These values come from Parameter Store.
        #
        # This means the prompt can explicitly communicate the configured
        # operational limits to the model.
        system_prompt=render(
            template,
            {
                "max_searches_per_turn": params[
                    "max_searches_per_turn"
                ],

                "max_neighbor_calls": params[
                    "max_neighbor_calls"
                ],

                "max_table_calls": params[
                    "max_table_calls"
                ],

                "expansion_token_budget": params[
                    "expansion_token_budget"
                ],
            },
        ),

        # Exact Prompt Management version used.
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

        # AgentCore Gateway endpoint.
        gateway_url=params[
            "gateway_url"
        ],

        # Convert Parameter Store strings into integers.
        max_searches_per_turn=int(
            params[
                "max_searches_per_turn"
            ]
        ),

        max_neighbor_calls=int(
            params[
                "max_neighbor_calls"
            ]
        ),

        max_table_calls=int(
            params[
                "max_table_calls"
            ]
        ),

        max_window=int(
            params[
                "max_window"
            ]
        ),

        expansion_token_budget=int(
            params[
                "expansion_token_budget"
            ]
        ),
    )


# ===========================================================================
# Process-local configuration cache
# ===========================================================================

# Stores the Settings object after the first successful load().
#
# None means the configuration has not yet been loaded.
_current: Settings | None = None


def settings() -> Settings:
    """Return the cached trial_search settings.

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
        Settings(...)
            │
            ▼
        _current

    Later calls:

        settings()
            │
            ▼
        _current
            │
            ▼
        return immediately

    This prevents Parameter Store, Secrets Manager, and Prompt
    Management from being queried for every user request.
    """

    # We modify the module-level cache when the settings have not yet
    # been loaded.
    global _current

    # Lazy initialization.
    if _current is None:

        # Load configuration from AWS once.
        _current = load()

    # Return the cached immutable Settings object.
    return _current


def use(
    value: Settings,
) -> None:
    """Install a Settings object directly.

    This function exists primarily for tests.

    Tests can construct a Settings instance directly and inject it without
    making real AWS calls.

    Example use cases:

        - unit testing search limits
        - testing prompt rendering
        - testing model configuration
        - testing guardrail configuration

    It is not the normal production configuration path.
    """

    # Replace the process-local cached settings.
    global _current
    _current = value