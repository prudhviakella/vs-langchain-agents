"""Every setting, read once from the environment.

    APP_STORE          dynamodb (default) | memory — memory is for tests and a
                       zero-setup local run; nothing survives a restart
    APP_TABLE          the single DynamoDB table (created by the CloudFormation stack)
    SUPERVISOR_ARN     the supervisor runtime to invoke
    SESSION_SECRET_ID  Secrets Manager id of the cookie-signing key
    SESSION_SECRET     the key itself — tests and local runs only
    ADMIN_USERS        comma-separated usernames that see everyone's AgentOps
    PRICE_*_PER_MTOK   model prices for the AgentOps cost column

WHY THE SIGNING KEY MUST BE SHARED

Up to two backend tasks run behind the load balancer. A key generated per
process would differ between them: a cookie minted by one task would fail on
the other, and users would be signed out at random. The key therefore comes
from Secrets Manager, the same for every task. Only the in-memory store — a
single local process — may fall back to a random key.
"""
import os

REGION = os.getenv("AWS_REGION", "us-east-1")
STORE = os.getenv("APP_STORE", "dynamodb")
TABLE = os.getenv("APP_TABLE", "trial-app")
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
