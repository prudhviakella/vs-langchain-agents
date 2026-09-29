"""The signed session: minted at login, verified locally on every request.

    POST /api/auth/login ──► mint(username) ──► Set-Cookie: trial_session=<JWT>
                                                  HttpOnly, SameSite=Lax, 30 days
    any /api request      ──► current_user() ──► verify signature, issuer,
                                                  audience, expiry ──► username
    POST /api/auth/logout ──► cookie cleared

Carried from the reference (session.py): HS256, our own issuer and audience,
the key from Secrets Manager so every backend task agrees on it.

WHY A COOKIE, NOT localStorage

HttpOnly means page scripts cannot read the token, so an injected script
cannot steal a session. The frontend and backend share one origin behind the
load balancer (/ and /api/*), so the browser sends the cookie on its own —
including on the streaming chat request.

WHAT THIS DOES NOT DO

    It does not look the user up per request: the username in a validly
    signed token is trusted until it expires. Deleting a user therefore
    takes effect when their cookie expires or they sign out.
"""
import json
import secrets
import time
from typing import Annotated

import jwt
from fastapi import Cookie, Depends, HTTPException

from . import settings

_ISS, _AUD = "trial-agents-app", "trial-agents-session"
_key: str | None = None


def _signing_key() -> str:
    global _key
    if _key:
        return _key
    if settings.SESSION_SECRET:
        _key = settings.SESSION_SECRET
    elif settings.SESSION_SECRET_ID:
        import boto3
        raw = boto3.client("secretsmanager", region_name=settings.REGION).get_secret_value(
            SecretId=settings.SESSION_SECRET_ID)["SecretString"]
        try:
            _key = json.loads(raw)["secret"]
        except (ValueError, KeyError, TypeError):
            _key = raw
    elif settings.STORE == "memory":
        _key = secrets.token_urlsafe(48)          # one local process only
    else:
        raise RuntimeError("SESSION_SECRET_ID is not set: every backend task must share "
                           "one signing key, or sessions fail at random between tasks")
    return _key


def mint(username: str) -> tuple[str, int]:
    """(token, max_age_seconds)."""
    ttl = settings.SESSION_DAYS * 86400
    now = int(time.time())
    token = jwt.encode({"sub": username, "iss": _ISS, "aud": _AUD, "iat": now,
                        "exp": now + ttl}, _signing_key(), algorithm="HS256")
    return token, ttl


def verify(token: str) -> str | None:
    """The username in a valid token, else None."""
    try:
        claims = jwt.decode(token, _signing_key(), algorithms=["HS256"],
                            audience=_AUD, issuer=_ISS)
        return claims.get("sub") or None
    except jwt.PyJWTError:
        return None


def current_user(trial_session: Annotated[str | None, Cookie()] = None) -> str:
    username = verify(trial_session) if trial_session else None
    if not username:
        raise HTTPException(status_code=401, detail="not signed in")
    return username


CurrentUser = Annotated[str, Depends(current_user)]


def is_admin(username: str) -> bool:
    return username in settings.ADMIN_USERS
