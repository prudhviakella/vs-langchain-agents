"""Passwords and the session cookie — no third-party auth.

    password ─► PBKDF2-SHA256, 600,000 iterations, 16-byte salt ─► "pbkdf2$600000$<salt>$<hash>"
    session  ─► "<base64 json {sub, exp}>.<base64 HMAC-SHA256>" in an HttpOnly cookie
                (Secure behind HTTPS, SameSite=Lax)

WHAT THIS DOES NOT DO
    No password reset or lockout: users are created on first sign-in with a new
    username (the trial app's flow). Tokens are not revocable before they expire.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from .settings import S, session_secret

ITERATIONS = 600_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(check.hex(), digest)
    except (ValueError, TypeError):
        return False


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue(user_id: str) -> str:
    body = _b64(json.dumps({"sub": user_id, "exp": int(time.time()) + S.session_days * 86400}).encode())
    sig = _b64(hmac.new(session_secret(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify(token: str | None) -> str | None:
    """The user id, or None for a missing, forged or expired token."""
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    good = _b64(hmac.new(session_secret(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, good):
        return None
    try:
        claims = json.loads(_unb64(body))
    except ValueError:
        return None
    return claims.get("sub") if claims.get("exp", 0) > time.time() else None
