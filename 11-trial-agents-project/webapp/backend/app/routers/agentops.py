"""AgentOps: token usage, cost, latency, agents, tools, feedback, traces.

    GET /api/agentops/summary?days=7         totals, p50/p95 latency, agent and
                                             tool counts, feedback up/down
    GET /api/agentops/interactions?days=7    one row per answered turn, each
                                             with its CloudWatch trace link
    GET /api/agentops/feedback?days=30       feedback in the window

SCOPE

Everyone sees their own interactions. Usernames in ADMIN_USERS see
everyone's — the reference's admin split, without Okta groups.

WHAT THIS DOES NOT DO

    It holds no spans. The full trace — every model call, tool call and
    specialist step — stays in CloudWatch; trace_url opens it there.
"""
import time

from fastapi import APIRouter, Query

from ..models import FeedbackOut, Interaction, UsageSummary
from ..session import CurrentUser, is_admin
from ..store import StoreDep

router = APIRouter(prefix="/api/agentops", tags=["agentops"])


def _since(days: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() - days * 86400))


def _scope(username: str) -> str | None:
    return None if is_admin(username) else username


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))]


@router.get("/summary", response_model=UsageSummary, summary="Usage over a window")
def summary(username: CurrentUser, store: StoreDep,
            days: int = Query(7, ge=1, le=365)) -> UsageSummary:
    rows = store.interactions_since(_since(days), _scope(username))
    feedback = [f for f in store.feedback_since(_since(days))
                if _scope(username) is None or f["username"] == username]
    agents, tools = {}, {}
    for r in rows:
        for a in r.get("agents", []):
            agents[a] = agents.get(a, 0) + 1
        for t in r.get("tools", []):
            tools[t] = tools.get(t, 0) + 1
    latencies = [r["latency_ms"] for r in rows if r.get("status") == "complete"]
    total = lambda key: sum(int(r.get(key, 0) or 0) for r in rows)
    return UsageSummary(
        days=days, scope="everyone" if is_admin(username) else "mine",
        interactions=len(rows), errors=sum(r.get("status") != "complete" for r in rows),
        input_tokens=total("input_tokens"), cached_input_tokens=total("cached_input_tokens"),
        output_tokens=total("output_tokens"), total_tokens=total("total_tokens"),
        cost_usd=round(sum(float(r.get("cost_usd", 0) or 0) for r in rows), 4),
        latency_p50_ms=_percentile(latencies, 0.5), latency_p95_ms=_percentile(latencies, 0.95),
        agents=agents, tools=tools,
        feedback_up=sum(f["rating"] == "up" for f in feedback),
        feedback_down=sum(f["rating"] == "down" for f in feedback))


@router.get("/interactions", response_model=list[Interaction], summary="Answered turns")
def interactions(username: CurrentUser, store: StoreDep, days: int = Query(7, ge=1, le=365),
                 limit: int = Query(200, ge=1, le=1000)) -> list[Interaction]:
    return [Interaction(**r) for r in store.interactions_since(_since(days),
                                                               _scope(username))[:limit]]


@router.get("/feedback", response_model=list[FeedbackOut], summary="Feedback in the window")
def feedback(username: CurrentUser, store: StoreDep,
             days: int = Query(30, ge=1, le=365)) -> list[FeedbackOut]:
    rows = [f for f in store.feedback_since(_since(days))
            if _scope(username) is None or f["username"] == username]
    return [FeedbackOut(**f) for f in sorted(rows, key=lambda f: f["created_at"], reverse=True)]
