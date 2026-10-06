"""Cohere re-ranking for semantic_search — recall first, then precision.

    semantic_search(query, top_k=8)
        │
        ├─ Pinecone query, top_k = pool (40)        RECALL: the right passage is
        │                                            somewhere in these 40
        v
    rerank(query, candidates, top_n=8, credentials)  PRECISION
        │  POST https://api.cohere.com/v2/rerank
        │  {"model": "rerank-v3.5", "query": "...",
        │   "documents": ["<40 passage texts>"], "top_n": 8}
        │
        ├─ HTTP 200   the 8 best, best first,         reranked=True
        │             each with rerank_score
        └─ anything   the first 8 in vector order,    reranked=False
           else       no rerank_score                 rerank_note says why

WHY A CROSS-ENCODER AFTER THE VECTOR SEARCH

The vector search compares two embeddings made separately: one for the
query, one for each passage. It is fast and finds the right passage
somewhere in a wide pool, but its order inside that pool is rough. Cohere
Rerank reads the query and each passage together and scores how well that
passage answers that query. So the search casts wide (40) and the re-ranker
chooses the 8 that are kept.

WHY A FAILURE FALLS BACK AND NEVER RAISES

Re-ranking only reorders. Without it, the vector order is still a usable
answer, so no re-ranking failure may fail a search. Failures are expected,
not rare: a Cohere trial key allows 10 rerank calls a minute and 1,000 API
calls a month (Cohere's rate-limit page). One question can run several
searches, and several people can ask at once, so HTTP 429 will happen.

The fallback is visible, not silent: reranked=False and rerank_note travel
back to the agent, which can see that the order is embedding similarity
only.

WHY THERE IS NO RETRY

A retry on 429 spends another of the month's 1,000 calls and makes the
analyst wait for the rate-limit window. Falling back to vector order at once
is cheaper and faster.

WHY THE KEY IS RE-READ UNTIL IT IS REAL

deploy.py creates the secret with a "replace-me" placeholder, and the
operator sets the real key afterwards, while Lambda containers may already
be warm. A placeholder is therefore never cached: each call reads the secret
again until a real key appears, which is then cached for the container's
lifetime.

WHAT THE RE-RANKER READS

The passage text, preceded by its section headings: "Eligibility >
Exclusion Criteria\\n<text>". A heading often carries the topic the text
itself only implies.

WHAT THIS DOES NOT DO

    - It sets no relevance floor. Every candidate keeps its place in the
      top_n even with a low score. A floor set too high turns "weak
      evidence" into "no passage found" — a confident negative. Choose one
      only after looking at real scores from this corpus.
    - It does not re-rank expand_neighbors or expand_table. Those fetch
      fixed things — the chunks around a hit, the rows of one table — not
      a ranked-relevance result.
    - It uses no Cohere SDK. One HTTPS POST with the standard library keeps
      the Lambda zip free of another dependency.
"""
import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger()

ENDPOINT = "https://api.cohere.com/v2/rerank"
TIMEOUT_S = 5
PLACEHOLDER = "replace-me"
DEFAULT_MODEL = "rerank-v3.5"

_cached: dict | None = None        # {"api_key", "model"} once a real key is seen


def document_text(passage: dict) -> str:
    """What Cohere reads for one passage: its headings, then its text."""
    headings = " > ".join(h for h in passage.get("headings") or [] if h)
    text = passage.get("text") or ""
    return f"{headings}\n{text}" if headings else text


def _credentials(read_secret) -> dict | None:
    """The Cohere key and model, or None while the secret is a placeholder."""
    global _cached
    if _cached is not None:
        return _cached
    secret = read_secret()
    if not secret.get("api_key") or secret["api_key"] == PLACEHOLDER:
        return None                                    # re-read on the next call
    model = secret.get("model")
    _cached = {"api_key": secret["api_key"],
               "model": model if model and model != PLACEHOLDER else DEFAULT_MODEL}
    return _cached


def _post(body: dict, api_key: str) -> dict:
    """One POST to Cohere's v2 rerank endpoint. Raises on any non-200."""
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())


def _fallback(passages: list[dict], top_n: int, note: str) -> dict:
    log.warning("rerank skipped: %s", note)
    return {"passages": passages[:top_n], "reranked": False, "rerank_note": note}


def rerank(query: str, passages: list[dict], top_n: int, read_secret) -> dict:
    """The top_n passages by Cohere relevance, best first.

    read_secret() returns the trial-search/cohere secret as a dict. Never
    raises: on any failure, the first top_n passages in their given (vector)
    order, with reranked=False and the reason in rerank_note.
    """
    if not passages:
        return {"passages": [], "reranked": False, "rerank_note": "no candidates to re-rank"}

    # STEP 1  credentials — a placeholder means "not configured yet", not an error
    try:
        creds = _credentials(read_secret)
    except Exception as exc:
        return _fallback(passages, top_n, f"Cohere secret unreadable ({type(exc).__name__})")
    if creds is None:
        return _fallback(passages, top_n, "Cohere key not set: the secret still holds "
                                          "replace-me")

    # STEP 2  one call; no retry (see the module docstring)
    body = {"model": creds["model"], "query": query,
            "documents": [document_text(p) for p in passages],
            "top_n": min(top_n, len(passages))}
    try:
        response = _post(body, creds["api_key"])
    except urllib.error.HTTPError as exc:
        reason = ("rate limit — a trial key allows 10 rerank calls a minute"
                  if exc.code == 429 else "key rejected" if exc.code == 401 else "")
        return _fallback(passages, top_n, f"Cohere HTTP {exc.code}"
                         + (f": {reason}" if reason else ""))
    except Exception as exc:                            # timeout, DNS, bad JSON
        return _fallback(passages, top_n, f"Cohere call failed ({type(exc).__name__})")

    # STEP 3  reorder by Cohere's indexes; an index outside the list is ignored
    ranked = []
    for result in response.get("results") or []:
        index = result.get("index")
        if isinstance(index, int) and 0 <= index < len(passages):
            ranked.append({**passages[index], "rerank_score": result.get("relevance_score")})
    if not ranked:
        return _fallback(passages, top_n, "Cohere returned no usable results")
    return {"passages": ranked[:top_n], "reranked": True, "rerank_note": ""}
