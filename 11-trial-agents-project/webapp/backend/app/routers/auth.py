"""Sign in, sign out, who am I.

    POST /api/auth/login {username, password}
        known username, right password   -> 200 + session cookie
        known username, wrong password   -> 401
        unknown username, no names       -> 404 {"code": "new_user"}  the UI then
                                            asks for first and last name
        unknown username, with names     -> user created -> 200 + cookie
    POST /api/auth/logout                  -> cookie cleared
    GET  /api/auth/me                      -> the signed-in user

WHAT THIS DOES NOT DO

    It does not say whether a username exists to someone with the wrong
    password beyond what sign-up needs: an existing name with a wrong
    password is a plain 401, never "user exists".
"""
import re

from fastapi import APIRouter, HTTPException, Response

from .. import settings
from ..models import USERNAME, LoginRequest, User
from ..passwords import hash_password, verify_password
from ..session import CurrentUser, is_admin, mint
from ..store import StoreDep
from ..store.base import now_iso

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user(record: dict) -> User:
    return User(username=record["username"], first_name=record.get("first_name", ""),
                last_name=record.get("last_name", ""), is_admin=is_admin(record["username"]))


def _signed_in(response: Response, username: str) -> None:
    token, max_age = mint(username)
    response.set_cookie(settings.COOKIE, token, max_age=max_age, httponly=True,
                        samesite="lax", secure=settings.COOKIE_SECURE, path="/")


@router.post("/login", response_model=User, summary="Sign in, or sign up a new username")
def login(body: LoginRequest, response: Response, store: StoreDep) -> User:
    username = body.username.strip().lower()
    if not re.match(USERNAME, username):
        raise HTTPException(422, "username: 3-40 characters, letters, digits, . _ -")

    # STEP 1 an existing user: the password decides
    record = store.get_user(username)
    if record:
        if not verify_password(body.password, record["password_hash"]):
            raise HTTPException(401, "wrong username or password")
        _signed_in(response, username)
        return _user(record)

    # STEP 2 a new username: names first, then create
    if not (body.first_name and body.first_name.strip() and body.last_name
            and body.last_name.strip()):
        raise HTTPException(404, {"code": "new_user",
                                  "message": "new username: first and last name required"})
    record = {"username": username, "first_name": body.first_name.strip(),
              "last_name": body.last_name.strip(),
              "password_hash": hash_password(body.password), "created_at": now_iso()}
    if not store.create_user(record):                 # created by a concurrent request
        raise HTTPException(409, "username was just taken; sign in instead")
    _signed_in(response, username)
    return _user(record)


@router.post("/logout", status_code=204, summary="Sign out")
def logout(response: Response) -> None:
    response.delete_cookie(settings.COOKIE, path="/")


@router.get("/me", response_model=User, summary="The signed-in user")
def me(username: CurrentUser, store: StoreDep) -> User:
    record = store.get_user(username)
    if not record:
        raise HTTPException(401, "user no longer exists")
    return _user(record)
