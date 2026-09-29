"""The FastAPI app: every route under /api, which is what the load balancer
forwards to this service (everything else goes to the frontend).

    /api/health                 load balancer health check (public)
    /api/auth/*                 sign in, sign out, me
    /api/conversations/*        list, search, create, rename, delete, messages
    /api/chat                   one turn, streamed (SSE)
    /api/feedback               rate an answer; my feedback
    /api/agentops/*             usage, interactions, feedback
    /api/docs                   Swagger UI
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import settings
from .routers import agentops, auth, chat, conversations, feedback, health

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="Trial agents API", docs_url="/api/docs", openapi_url="/api/openapi.json")
if settings.CORS_ORIGINS:                          # local dev only; same origin behind the ALB
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
for module in (health, auth, conversations, chat, feedback, agentops):
    app.include_router(module.router)
