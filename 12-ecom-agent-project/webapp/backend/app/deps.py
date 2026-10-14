"""Request dependencies: the signed-in user, and a conversation they own.

    cookie ─► security.verify ─► user_id ─► store user ─► Analyst      (401 if any step fails)
    conversation_id ─► store ─► owner == user ? conversation : 404     (never 403: ids cannot be probed)
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from . import security
from .models import Analyst
from .settings import S
from .store import get_store


def analyst_from(user: dict) -> Analyst:
    first, last = user.get("first_name") or "", user.get("last_name") or ""
    name = f"{first} {last}".strip() or user["user_id"]
    return Analyst(user_id=user["user_id"], name=name, first_name=first or None, last_name=last or None,
                   short_name=(first[:1] + last[:1]).upper() or user["user_id"][:2].upper(),
                   role="Admin" if "*" in S.admin_users or user["user_id"] in S.admin_users else "Analyst")


def current_user(request: Request) -> Analyst:
    user_id = security.verify(request.cookies.get(S.cookie_name))
    user = get_store().get_user(user_id) if user_id else None
    if not user:
        raise HTTPException(401, "not signed in")
    return analyst_from(user)


def owned_conversation(conversation_id: str, me: Analyst = Depends(current_user)) -> dict:
    conv = get_store().get_conversation(conversation_id)
    if not conv or conv["owner_id"] != me.user_id:
        raise HTTPException(404, "conversation not found")
    return conv
