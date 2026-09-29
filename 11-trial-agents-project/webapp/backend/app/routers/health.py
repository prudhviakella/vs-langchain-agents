"""GET /api/health — the load balancer's target health check. No auth, no AWS calls."""
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", summary="Liveness")
def health() -> dict:
    return {"status": "ok"}
