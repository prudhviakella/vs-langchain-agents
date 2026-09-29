"""Persistence behind one interface (the reference's Store pattern).

    routers ──► Store (base.py) ──┬── MemoryStore    APP_STORE=memory   tests, local
                                  └── DynamoStore    APP_STORE=dynamodb AWS (default)

Routers depend on the interface only, so switching backends is one
environment variable.
"""
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from .. import settings
from .base import Store


@lru_cache(maxsize=1)
def get_store() -> Store:
    if settings.STORE == "memory":
        from .memory import MemoryStore
        return MemoryStore()
    from .dynamodb import DynamoStore
    return DynamoStore()


StoreDep = Annotated[Store, Depends(get_store)]
