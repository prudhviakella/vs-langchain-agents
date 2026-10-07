"""Every setting, read once from the environment.

    where the backend runs       APP_TABLE          table                  who creates it
    ───────────────────────────  ─────────────────  ─────────────────────  ─────────────────────────
    your machine (uvicorn)       not set            trial-webapp-local     this backend, on start
    AWS (ECS, deployed stack)    set by the stack   trial-webapp           CloudFormation

    APP_TABLE          the DynamoDB table. Unset = a local run, which uses (and
                       creates if missing) LOCAL_TABLE
    APP_DDB_ENDPOINT   optional DynamoDB Local URL — still DynamoDB, on your machine
    SUPERVISOR_ARN     the supervisor runtime to invoke
    SESSION_SECRET_ID  Secrets Manager id of the cookie-signing key (set by the stack)
    SESSION_SECRET     the key itself, if you want local sign-ins to survive a restart
    ADMIN_USERS        comma-separated usernames that see everyone's AgentOps
    PRICE_*_PER_MTOK   model prices for the AgentOps cost column

EVERYTHING IS STORED IN DYNAMODB, LOCAL RUN INCLUDED

Users, conversations, messages and feedback always go to a DynamoDB table.
A local run and the deployed app behave the same; only the table differs.

WHY A LOCAL RUN HAS ITS OWN TABLE

The deployed table is created by the CloudFormation stack. If a local run
created a table with the stack's name first, the stack would later fail with
"already exists". So a local run owns a differently named table and creates
it itself; the backend never creates a table it was pointed at with
APP_TABLE — that table belongs to whoever named it. To work locally against
the deployed app's data, set APP_TABLE=trial-webapp.

WHY THE SIGNING KEY MUST BE SHARED

Up to two backend tasks run behind the load balancer. A key generated per
process would differ between them: a cookie minted by one task would fail on
the other, and users would be signed out at random. In AWS the key therefore
comes from Secrets Manager. A local run is one process, so without
SESSION_SECRET it may use a random key — a restart signs you out, but your
user and conversations are in DynamoDB and are all still there.
"""
import os

REGION = os.getenv("AWS_REGION", "us-east-1")
LOCAL_TABLE = "trial-webapp-local"
TABLE = os.getenv("APP_TABLE") or LOCAL_TABLE
OWNS_TABLE = not os.getenv("APP_TABLE")                     # a local run: create if missing
DDB_ENDPOINT = os.getenv("APP_DDB_ENDPOINT", "")          # DynamoDB Local, optional

SUPERVISOR_ARN = os.getenv("SUPERVISOR_ARN", "")
SUPERVISOR_TIMEOUT_S = int(os.getenv("SUPERVISOR_TIMEOUT_S", "300"))
HISTORY_MESSAGES = 20                                       # 10 interactions

SESSION_SECRET_ID = os.getenv("SESSION_SECRET_ID", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "30"))
COOKIE = "trial_session"
# HTTP-only deployment for now (no domain/certificate). Set true behind HTTPS.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

ADMIN_USERS = {u.strip().lower() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()}
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "").split(",") if o]   # local dev only

# gpt-6-sol list prices (OpenAI models page). Cached input is billed at the
# full input price unless configured — an upper bound, never an undercount.
PRICE_INPUT_PER_MTOK = float(os.getenv("PRICE_INPUT_PER_MTOK", "2.0"))
PRICE_OUTPUT_PER_MTOK = float(os.getenv("PRICE_OUTPUT_PER_MTOK", "10.0"))
PRICE_CACHED_INPUT_PER_MTOK = float(os.getenv("PRICE_CACHED_INPUT_PER_MTOK",
                                              str(PRICE_INPUT_PER_MTOK)))
