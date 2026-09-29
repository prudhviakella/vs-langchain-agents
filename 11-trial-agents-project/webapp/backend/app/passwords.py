"""Password hashing with the standard library: PBKDF2-HMAC-SHA256.

    stored   "pbkdf2_sha256$600000$<salt b64>$<hash b64>"
    verify   recompute with the stored salt and iterations; constant-time compare

600,000 iterations follows OWASP's current guidance for PBKDF2-SHA256. The
iteration count is stored with each hash, so raising it later leaves every
existing password verifiable.

WHAT THIS DOES NOT DO

    No rate limiting of login attempts. The load balancer is restricted to
    known IP ranges for now; add throttling before opening it wider.
"""
import base64
import hashlib
import hmac
import secrets

ITERATIONS = 600_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return "$".join(["pbkdf2_sha256", str(ITERATIONS),
                     base64.b64encode(salt).decode(), base64.b64encode(digest).decode()])


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, expected = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                     base64.b64decode(salt), int(iterations))
        return hmac.compare_digest(digest, base64.b64decode(expected))
    except (ValueError, TypeError):
        return False
