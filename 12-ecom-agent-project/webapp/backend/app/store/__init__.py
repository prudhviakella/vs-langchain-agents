"""The store: one interface, two implementations.

    get_store() ─► DynamoStore   (APP_STORE=dynamodb — local runs and AWS)
                └► MemoryStore   (APP_STORE=memory — tests)
"""
from __future__ import annotations

from functools import lru_cache

from ..settings import S


@lru_cache(maxsize=1)
def get_store():
    if S.store == "memory":
        from .memory import MemoryStore
        return MemoryStore()
    from .dynamo import DynamoStore
    return DynamoStore(S.table, S.region)
