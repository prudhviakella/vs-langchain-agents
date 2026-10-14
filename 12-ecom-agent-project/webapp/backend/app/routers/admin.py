"""The admin console (AgentOps) — usage, feedback, the knowledge base. Admins only (ADMIN_USERS).

    GET /admin/usage?days=7|30|90          totals · by_day · by_agent · by_user      (USAGE#<day> index)
    GET /admin/feedback?month=&vote=       the month's 👍/👎 with question + answer   (FEEDBACK#<month> index)
    GET /admin/kb?q=&namespace=            published KB records, keyword-filtered      (Pinecone ecom-kb)
    GET /admin/kb/search?q=                by meaning: the same embedding model as grounding (needs OpenAI key)

COST
    tokens × MODEL_PRICES (USD per 1M tokens). Without a price for the model, cost is null —
    shown as "—", never estimated.

WHAT THIS DOES NOT DO
    It changes nothing: every route is read-only. The KB is published by the ingestion pipeline.
"""
from __future__ import annotations

import datetime as dt
import json
import time

from fastapi import APIRouter, Depends, HTTPException

from ..deps import current_user
from ..models import Analyst
from ..settings import S, api_key
from ..store import get_store

router = APIRouter(prefix="/admin")


def require_admin(me: Analyst = Depends(current_user)) -> Analyst:
    if me.role != "Admin":
        raise HTTPException(403, "admins only")
    return me


def _cost(model: str, inp: int, out: int) -> float | None:
    p = S.model_prices.get(model)
    if not p:
        return None
    return round(inp / 1e6 * float(p.get("input", 0)) + out / 1e6 * float(p.get("output", 0)), 6)


def _add_cost(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(a + b, 6)


# ── usage ────────────────────────────────────────────────────────────────
@router.get("/usage")
def usage(days: int = 7, _: Analyst = Depends(require_admin)):
    days = max(1, min(days, 366))
    today = dt.datetime.now(dt.timezone.utc).date()
    start = dt.datetime.combine(today - dt.timedelta(days=days - 1), dt.time.min, dt.timezone.utc)
    end_ns = time.time_ns() + 1
    answers = get_store().answers_between(int(start.timestamp() * 1e9), end_ns)

    by_day = {(start.date() + dt.timedelta(days=i)).isoformat(): {"interactions": 0, "tokens": 0, "cost_usd": 0.0}
              for i in range(days)}
    by_agent: dict[str, dict] = {}
    by_user: dict[str, dict] = {}
    latencies, total_cost, failed = [], 0.0, 0
    tot = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for m in answers:
        day = dt.datetime.fromtimestamp(m.created_at_ns / 1e9, dt.timezone.utc).date().isoformat()
        failed += m.status != "complete"
        if m.latency_ms is not None:            # 0 ms is a latency, not a missing one
            latencies.append(m.latency_ms)
        m_tokens, m_cost = 0, 0.0
        for u in m.usage_by_agent or []:
            agent, use = u.get("agent", "?"), u.get("usage") or {}
            inp, out = int(use.get("input_tokens", 0)), int(use.get("output_tokens", 0))
            tokens = int(use.get("total_tokens", inp + out))
            c = _cost(S.agent_model, inp, out)
            a = by_agent.setdefault(agent, {"agent": agent, "calls": 0, "input_tokens": 0, "output_tokens": 0,
                                            "total_tokens": 0, "cost_usd": 0.0})
            a["calls"] += 1
            a["input_tokens"] += inp
            a["output_tokens"] += out
            a["total_tokens"] += tokens
            a["cost_usd"] = _add_cost(a["cost_usd"], c)
            tot["input_tokens"] += inp
            tot["output_tokens"] += out
            tot["total_tokens"] += tokens
            m_tokens += tokens
            m_cost = _add_cost(m_cost, c)
        total_cost = _add_cost(total_cost, m_cost)
        d = by_day.setdefault(day, {"interactions": 0, "tokens": 0, "cost_usd": 0.0})
        d["interactions"] += 1
        d["tokens"] += m_tokens
        d["cost_usd"] = _add_cost(d["cost_usd"], m_cost)
        user = m.user_id or "unknown"
        b = by_user.setdefault(user, {"user_id": user, "interactions": 0, "tokens": 0, "cost_usd": 0.0})
        b["interactions"] += 1
        b["tokens"] += m_tokens
        b["cost_usd"] = _add_cost(b["cost_usd"], m_cost)

    latencies.sort()
    priced = S.agent_model in S.model_prices
    return {
        "days": days, "priced": priced, "model": S.agent_model,
        "totals": {"interactions": len(answers), "users": len(by_user), "failed": failed, **tot,
                   "cost_usd": total_cost if priced else None,
                   "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else None,
                   "p95_latency_ms": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None},
        "by_day": [{"day": k, **v, "cost_usd": v["cost_usd"] if priced else None} for k, v in sorted(by_day.items())],
        "by_agent": sorted(({**a, "cost_usd": a["cost_usd"] if priced else None} for a in by_agent.values()),
                           key=lambda a: -a["total_tokens"]),
        "by_user": sorted(({**u, "cost_usd": u["cost_usd"] if priced else None} for u in by_user.values()),
                          key=lambda u: -u["interactions"]),
    }


# ── feedback ─────────────────────────────────────────────────────────────
@router.get("/feedback")
def feedback(month: str = "", vote: str = "", _: Analyst = Depends(require_admin)):
    try:
        first = dt.date.fromisoformat(f"{month}-01") if month else dt.datetime.now(dt.timezone.utc).date().replace(day=1)
    except ValueError:
        raise HTTPException(422, "month must be YYYY-MM")
    nxt = (first + dt.timedelta(days=32)).replace(day=1)
    to_ns = lambda d: int(dt.datetime.combine(d, dt.time.min, dt.timezone.utc).timestamp() * 1e9)
    items = get_store().feedback_between(to_ns(first), to_ns(nxt))
    if vote in ("up", "down"):
        items = [i for i in items if i.get("vote") == vote]
    items.sort(key=lambda i: -i["at_ns"])
    return {"month": first.isoformat()[:7], "items": items,
            "likes": sum(i.get("vote") == "up" for i in items), "dislikes": sum(i.get("vote") == "down" for i in items),
            "comments": sum(bool(i.get("comment")) for i in items)}


# ── knowledge base ───────────────────────────────────────────────────────
_KB: dict = {"at": 0.0, "records": []}
_NAMESPACES = ("nlq-schema", "nlq-examples", "common")


def _pinecone():
    key = api_key("PINECONE_API_KEY", "PINECONE_SECRET_ID")
    if not key:
        raise HTTPException(503, "the KB browser needs PINECONE_API_KEY (or PINECONE_SECRET_ID)")
    from pinecone import Pinecone
    return Pinecone(api_key=key).Index(S.pinecone_index)


def _record(vid: str, ns: str, meta: dict) -> dict:
    title = (meta.get("table") and meta.get("column") and f"{meta['table']}.{meta['column']}") or meta.get("table") \
        or meta.get("term") or meta.get("question") or (meta.get("left_table") and
                                                         f"{meta['left_table']} ⋈ {meta.get('right_table')}") or vid
    return {"id": vid, "namespace": ns, "type": meta.get("type", ""), "title": title,
            "text": meta.get("text") or meta.get("description") or "", "metadata": meta}


def kb_records() -> list[dict]:
    """Every published record (≈340), cached 5 minutes."""
    if _KB["records"] and time.time() - _KB["at"] < 300:
        return _KB["records"]
    index, out = _pinecone(), []
    for ns in _NAMESPACES:
        ids = [v.id for page in index.list(namespace=ns) for v in page.vectors]
        for i in range(0, len(ids), 100):
            fetched = index.fetch(ids=ids[i:i + 100], namespace=ns).vectors
            out += [_record(vid, ns, dict(v.metadata or {})) for vid, v in fetched.items()]
    out.sort(key=lambda r: (r["namespace"], r["type"], r["title"]))
    _KB.update(at=time.time(), records=out)
    return out


@router.get("/kb")
def kb(q: str = "", namespace: str = "", _: Analyst = Depends(require_admin)):
    records = kb_records()
    if namespace:
        records = [r for r in records if r["namespace"] == namespace]
    if q.strip():
        needle = q.strip().lower()
        records = [r for r in records if needle in (r["title"] + " " + r["text"] + " " +
                                                    json.dumps(r["metadata"], default=str)).lower()]
    counts = {ns: sum(r["namespace"] == ns for r in kb_records()) for ns in _NAMESPACES}
    return {"records": records[:500], "total": len(records), "counts": counts}


@router.get("/kb/search")
def kb_search(q: str, _: Analyst = Depends(require_admin)):
    key = api_key("OPENAI_API_KEY", "OPENAI_SECRET_ID")
    if not key:
        raise HTTPException(503, "search by meaning needs OPENAI_API_KEY (or OPENAI_SECRET_ID)")
    from openai import OpenAI
    vec = OpenAI(api_key=key).embeddings.create(model=S.embed_model, input=[q]).data[0].embedding
    index, hits = _pinecone(), []
    for ns in _NAMESPACES:
        for m in index.query(vector=vec, top_k=8, namespace=ns, include_metadata=True).matches:
            hits.append({**_record(m.id, ns, dict(m.metadata or {})), "score": round(float(m.score), 3)})
    return {"records": sorted(hits, key=lambda r: -r["score"])}
