"""Sign in, sign out, who am I.

    POST /auth/login {username, password, first_name?, last_name?}
        known user + right password ─► 200 Analyst + session cookie
        known user + wrong password ─► 401
        new username, no names      ─► 404 {code: new_user}  (the screen asks for a name)
        new username + names        ─► account created ─► 200 Analyst + cookie
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from .. import security
from ..deps import analyst_from, current_user
from ..models import Analyst
from ..settings import S
from ..store import get_store

router = APIRouter()
_USERNAME = re.compile(r"^[a-z0-9][a-z0-9._-]{2,39}$")


class Login(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=8, max_length=200)
    first_name: str | None = Field(default=None, max_length=60)
    last_name: str | None = Field(default=None, max_length=60)


def _set_cookie(resp: Response, user_id: str) -> None:
    resp.set_cookie(S.cookie_name, security.issue(user_id), httponly=True, secure=S.cookie_secure,
                    samesite="lax", max_age=S.session_days * 86400, path="/")


@router.post("/auth/login", response_model=Analyst)
def login(body: Login, resp: Response):
    user_id = body.username.strip().lower()
    if not _USERNAME.match(user_id):
        raise HTTPException(422, "username: 3-40 characters, letters, digits, . _ -")
    st = get_store()
    user = st.get_user(user_id)
    if user:
        if not security.verify_password(body.password, user["password_hash"]):
            raise HTTPException(401, "wrong username or password")
    else:
        if not (body.first_name and body.last_name):
            raise HTTPException(404, {"code": "new_user", "message": "new username — add your first and last name"})
        user = {"user_id": user_id, "first_name": body.first_name.strip(), "last_name": body.last_name.strip(),
                "password_hash": security.hash_password(body.password)}
        if not st.create_user(user):
            raise HTTPException(409, "that username was just taken — sign in instead")
    _set_cookie(resp, user_id)
    return analyst_from(user)


@router.post("/auth/logout", status_code=204)
def logout(resp: Response):
    resp.delete_cookie(S.cookie_name, path="/")


@router.get("/me", response_model=Analyst)
def me(user: Analyst = Depends(current_user)):
    return user
