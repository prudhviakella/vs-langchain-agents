"""Persistence behind one interface (the reference's Store pattern).

    routers ──► Store (base.py) ──► DynamoStore    every run: local and AWS
                                                     (table chosen in settings.py)

    tests only:  use_store(MemoryStore())           installed directly by the test
                                                     setup; no setting selects it

Routers depend on the interface only, so tests can swap the store without
touching a router.

WHAT THIS DOES NOT DO

    There is no environment variable that switches the app to an in-memory
    store. A local run stores users, conversations, messages and feedback in
    DynamoDB, exactly like the deployed app — so what you test locally is
    what runs in AWS.
"""
from typing import Annotated

from fastapi import Depends

from .base import Store

_store: Store | None = None


def get_store() -> Store:
    """The process's store, built on first use: always DynamoDB."""
    global _store
    if _store is None:
        from .dynamodb import DynamoStore
        _store = DynamoStore()
    return _store


def use_store(store: Store | None) -> None:
    """Install a store directly — tests, and nothing else. None resets it."""
    global _store
    _store = store


StoreDep = Annotated[Store, Depends(get_store)]
