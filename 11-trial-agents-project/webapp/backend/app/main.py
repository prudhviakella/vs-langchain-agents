"""The FastAPI app: every route under /api, which is what the load balancer
forwards to this service (everything else goes to the frontend).

    /api/health                 load balancer health check (public)
    /api/auth/*                 sign in, sign out, me
    /api/conversations/*        list, search, create, rename, delete, messages
    /api/chat                   one turn, streamed (SSE)
    /api/feedback               rate an answer; my feedback
    /api/agentops/*             usage, interactions, feedback
    /api/docs                   Swagger UI

    on start    the DynamoDB table is checked (and, for a local run, created)
                before the first request — see store/dynamodb.py ensure_ready
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import settings
from .routers import agentops, auth, chat, conversations, feedback, health
from .store import get_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """STEP 1  the table: ready, or created for a local run, or a clear stop.
    A missing table fails the START, with the reason in the log — not the
    first sign-in, with a 500."""
    state = get_store().ensure_ready(create=settings.OWNS_TABLE)
    log.info("DynamoDB table %r %s (region %s)", settings.TABLE, state, settings.REGION)
    yield


app = FastAPI(title="Trial agents API", docs_url="/api/docs", openapi_url="/api/openapi.json",
              lifespan=lifespan)
if settings.CORS_ORIGINS:                          # local dev only; same origin behind the ALB
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
for module in (health, auth, conversations, chat, feedback, agentops):
    app.include_router(module.router)
