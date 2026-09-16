"""Secrets and tuning configuration.

Two different stores for two different things, and the difference is not
cosmetic:

    Secrets Manager     OPENAI_API_KEY, PINECONE_API_KEY, NEO4J_PASSWORD
                         Rotatable, audited, never appears in a task
                         definition, the console, or CloudTrail.

    Parameter Store     USE_OUTLINE_HEADINGS, MIN_CHUNK_TOKENS, and the
                         rest — the chunking and heading settings this
                         project spent real effort getting right. Changing
                         one is editing a value, not rotating a credential,
                         and it should not need the same process.

Both are injected into the container the same way, as `secrets` in the ECS
task definition rather than `environment` — the ECS agent resolves both at
launch, so the running code never calls either API directly.
"""

import json

import boto3
from botocore.exceptions import ClientError

from . import config

sm = boto3.client("secretsmanager", region_name=config.REGION)
ssm = boto3.client("ssm", region_name=config.REGION)


def exists(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except ClientError:
        return False


def create_secret() -> None:
    """Create the API-key secret with placeholder values.

    Placeholders rather than prompting, so deploy stays non-interactive.
    Real values are set afterward with a separate command — printed here so
    the deploy output tells you the exact next step rather than leaving you
    to remember it.

    Every key gets the literal string "replace-me", not a generic "..." or
    an empty value. Non-empty for the same reason NEO4J_URI's Parameter
    Store default is "replace-me" rather than "" — see infra/config.py's
    comment on that for the concrete failure an empty value produces. The
    printed example command below uses the SAME literal string, on purpose:
    someone comparing what was just created against the instructions for
    updating it should see the same value in both places, not a stored
    "replace-me" next to a printed "...".
    """
    print(f"-- Secrets Manager {config.SECRET_NAME} --")
    placeholder = {k: "replace-me" for k in config.SECRET_KEYS}
    if exists(sm.describe_secret, SecretId=config.SECRET_NAME):
        print("   exists (update with the CLI command below if keys changed)")
    else:
        sm.create_secret(
            Name=config.SECRET_NAME,
            SecretString=json.dumps(placeholder),
        )
        print("   created with placeholder values")

    print("   set the real values with:")
    print(f"   aws secretsmanager put-secret-value --secret-id {config.SECRET_NAME} \\")
    print(f"     --secret-string '{json.dumps(placeholder)}'")
    print("   (each 'replace-me' above is a real value — this is what the "
          "secret currently holds; substitute your actual keys before "
          "running the command)")


def create_parameters() -> None:
    """Write every default tuning parameter that is not already set.

    Checked individually rather than overwritten wholesale: a re-deploy must
    never silently reset a value someone tuned by hand after the first
    deploy. Overwrite=False on put_parameter would raise on an existing
    parameter, which is the wrong failure mode here — this should skip
    quietly, not abort the rest of deploy over a parameter that is
    intentionally different from its default.
    """
    print(f"-- Parameter Store: {len(config.DEFAULT_PARAMS)} tuning parameters --")
    for key, default in config.DEFAULT_PARAMS.items():
        name = config.param_name(key)
        if exists(ssm.get_parameter, Name=name):
            print(f"   {key:<24} exists, left as-is")
            continue
        ssm.put_parameter(Name=name, Value=default, Type="String")
        print(f"   {key:<24} = {default!r} (default)")
