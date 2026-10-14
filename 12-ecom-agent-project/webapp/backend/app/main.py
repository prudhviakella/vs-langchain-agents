"""The backend: FastAPI under /api/v1 (and /api/health for the load balancer).

    local:  uvicorn app.main:app --port 8000      (Vite proxies /api from the frontend)
    AWS:    the same app in the ECS container, behind CloudFront /api/* -> internal ALB
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routers import admin, auth, chat, conversations, feedback
from .store import get_store

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Open the store BEFORE serving: on a first start it creates the DynamoDB table and
    waits until it is active (20-60 s). Done here, the server is ready when it says so —
    not done here, the first sign-in silently waits for the table instead."""
    log.info("opening the store…")
    get_store()
    log.info("store ready — serving")
    yield


app = FastAPI(title="ecom NLQ web backend", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
for r in (auth.router, conversations.router, chat.router, feedback.router, admin.router):
    app.include_router(r, prefix="/api/v1")


@app.get("/api/health")
def health():
    return {"status": "ok"}
