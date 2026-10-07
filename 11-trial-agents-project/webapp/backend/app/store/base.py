"""The Store interface. Records are plain dicts shaped like models.py.

    users          get_user, create_user (fails if the username exists)
    conversations  create, get, list (newest activity first), rename, delete
                   (delete removes its messages and feedback too)
    messages       add (bumps the conversation's activity), list, get,
                   recent (the newest few, role + text + status only — history)
    feedback       put (one per user per answer — replaced on resubmit),
                   by user, by conversation, since a time
    interactions   answered turns since a time, optionally one user's

Times are ISO-8601 UTC strings with milliseconds; they sort as text.
"""
import time
import uuid
from abc import ABC, abstractmethod


def now_iso() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t * 1000) % 1000:03d}Z"


def new_id() -> str:
    return uuid.uuid4().hex


class Store(ABC):
    def ensure_ready(self, create: bool) -> str:
        """Prepare durable storage before the first request. A store with
        nothing to prepare (the tests' MemoryStore) is always ready."""
        return "ready"

    @abstractmethod
    def get_user(self, username: str) -> dict | None: ...
    @abstractmethod
    def create_user(self, user: dict) -> bool: ...

    @abstractmethod
    def create_conversation(self, username: str, title: str) -> dict: ...
    @abstractmethod
    def get_conversation(self, conversation_id: str) -> dict | None: ...
    @abstractmethod
    def list_conversations(self, username: str) -> list[dict]: ...
    @abstractmethod
    def rename_conversation(self, conversation_id: str, title: str) -> None: ...
    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> None: ...

    @abstractmethod
    def add_message(self, message: dict) -> None: ...
    @abstractmethod
    def list_messages(self, conversation_id: str) -> list[dict]: ...
    @abstractmethod
    def get_message(self, conversation_id: str, message_id: str) -> dict | None: ...
    @abstractmethod
    def recent_messages(self, conversation_id: str, limit: int) -> list[dict]:
        """The newest `limit` messages, OLDEST first, each only
        {role, text, status}. What a turn's history needs — not the answer
        details, which are most of an answer's size."""

    @abstractmethod
    def put_feedback(self, feedback: dict) -> None: ...
    @abstractmethod
    def feedback_by_user(self, username: str) -> list[dict]: ...
    @abstractmethod
    def feedback_for_conversation(self, conversation_id: str) -> list[dict]: ...
    @abstractmethod
    def feedback_since(self, since: str) -> list[dict]: ...

    @abstractmethod
    def interactions_since(self, since: str, username: str | None = None) -> list[dict]: ...
