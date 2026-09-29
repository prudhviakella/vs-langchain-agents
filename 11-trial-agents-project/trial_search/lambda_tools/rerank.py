"""Cohere re-ranking for semantic_search — recall first, then precision.

The semantic-search pipeline uses a two-stage retrieval strategy:

    semantic_search(query, top_k=8)
        │
        │  Stage 1 — RECALL
        │
        ▼
    Pinecone vector search
        │
        │  Retrieve a wider candidate pool
        │  e.g. 40 passages
        ▼
    40 candidate passages
        │
        │  Stage 2 — PRECISION
        │
        ▼
    Cohere Rerank
        │
        │  Query + all candidate texts
        ▼
    Top 8 passages
        │
        ▼
    trial_search agent


WHY VECTOR SEARCH + CROSS-ENCODER RERANKING
--------------------------------------------

Vector search compares:

    embedding(query)
        vs.
    embedding(passage)

This is very efficient and works well for finding a broad set of
potentially relevant passages.

However, the ordering inside that candidate pool is not necessarily the
best answer ordering.

Cohere Rerank instead evaluates:

    query + passage

together.

Therefore the architecture deliberately separates:

    Pinecone
        = broad recall

    Cohere
        = precise ordering


FAILURE / FALLBACK BEHAVIOR
---------------------------

Reranking is an optimization, not a hard dependency.

If Cohere is unavailable:

    Cohere failure
         │
         ▼
    use Pinecone vector order
         │
         ▼
    return top_n passages
         │
         └── reranked=False


This means a Cohere outage does not turn a valid search into:

    "no results"


The fallback is explicitly reported using:

    reranked=False
    rerank_note="..."


WHY THERE IS NO RETRY
---------------------

A retry would:

    - consume another API call
    - potentially worsen rate-limit pressure
    - increase user latency

The design therefore fails fast and uses the already-valid Pinecone
ranking.


CREDENTIAL CACHING
------------------

The deployment can initially create a Secrets Manager secret containing:

    replace-me

The operator can populate the real Cohere key afterwards.

Lambda containers can already be warm when this happens.

Therefore:

    placeholder
        │
        └── DO NOT CACHE

    real key
        │
        └── CACHE for warm-container lifetime


WHAT COHERE SEES
----------------

Each document sent to Cohere is constructed as:

    <heading 1> > <heading 2>
    <passage text>

The headings are included because they often provide context that the
passage text alone does not explicitly state.


WHAT THIS MODULE DOES NOT DO
----------------------------

1. It does not apply a relevance threshold.

   Every usable Cohere result is retained.

   A hard threshold could turn weak evidence into:

       "no evidence"

   which would be a potentially incorrect negative.

2. It does not rerank neighbor expansions.

   Neighbor expansion is deterministic context around a known chunk.

3. It does not rerank table expansions.

   Table expansion retrieves the rows belonging to one known table.

4. It does not use the Cohere SDK.

   A direct HTTPS request keeps the Lambda dependency footprint small.
"""


# ===========================================================================
# Standard library imports
# ===========================================================================

# Used to serialize the Cohere request body and parse its JSON response.
import json

# Used to record fallback/reranking events.
import logging

# Used to distinguish HTTP errors from other network/runtime errors.
import urllib.error

# Standard-library HTTPS client.
#
# No Cohere SDK is required.
import urllib.request


# ===========================================================================
# Logger
# ===========================================================================

log = logging.getLogger()


# ===========================================================================
# Cohere configuration
# ===========================================================================

# Cohere v2 Rerank API endpoint.
ENDPOINT = (
    "https://api.cohere.com/v2/rerank"
)


# Maximum amount of time to wait for the HTTP request.
#
# Reranking is optional, so we keep this deliberately bounded.
TIMEOUT_S = 5


# Placeholder used by deployment infrastructure before the real key is
# populated.
#
# A placeholder must never be cached.
PLACEHOLDER = "replace-me"


# Default Cohere reranking model.
DEFAULT_MODEL = "rerank-v3.5"


# ===========================================================================
# Warm-container credential cache
# ===========================================================================

# Once a real API key is discovered, cache it for the lifetime of the
# Lambda's warm container.
#
# Shape:
#
#     {
#         "api_key": "...",
#         "model": "rerank-v3.5"
#     }
#
# None means:
#
#     credentials have not yet been cached
#
# IMPORTANT:
#
# A placeholder is deliberately NOT cached.
_cached: dict | None = None


# ===========================================================================
# Convert a Passage into Cohere document text
# ===========================================================================

def document_text(
    passage: dict,
) -> str:
    """Build the exact text that Cohere will rerank.

    Format:

        Heading1 > Heading2
        Passage text

    Including headings gives the reranker additional structural context.

    For example:

        Eligibility > Exclusion Criteria
        Patients who received prior therapy...

    is more informative than the passage text by itself.
    """

    # Combine non-empty section headings into a readable hierarchy.
    headings = " > ".join(
        heading
        for heading in (
            passage.get(
                "headings"
            )
            or []
        )
        if heading
    )


    # Extract the actual passage text.
    text = passage.get(
        "text"
    ) or ""


    # If headings exist, place them before the passage.
    #
    # Otherwise send only the passage text.
    return (
        f"{headings}\n{text}"
        if headings
        else text
    )


# ===========================================================================
# Load Cohere credentials
# ===========================================================================

def _credentials(
    read_secret,
) -> dict | None:
    """Return Cohere credentials or None while the secret is a placeholder.

    `read_secret` is injected by the caller rather than hard-coding the
    Secrets Manager implementation here.

    This also makes the function easier to test.

    Credential lifecycle:

        first call
            │
            ▼
        read Secrets Manager
            │
            ├── placeholder
            │      │
            │      └── return None
            │
            └── real key
                   │
                   ▼
                cache it


    Once a real key has been cached, Secrets Manager is no longer queried
    for every reranking request in the warm container.
    """

    global _cached


    # If a real credential was already discovered, reuse it.
    if _cached is not None:
        return _cached


    # Read the current secret.
    #
    # This is intentionally done before caching because deployment may have
    # initially populated the secret with "replace-me".
    secret = read_secret()


    # If the key is missing or still contains the deployment placeholder,
    # treat it as "not configured yet".
    #
    # IMPORTANT:
    #
    # Do not cache this state.
    #
    # The next Lambda invocation should read Secrets Manager again in case
    # the operator has populated the real key.
    if (
        not secret.get(
            "api_key"
        )
        or secret[
            "api_key"
        ] == PLACEHOLDER
    ):
        return None


    # A custom model can be supplied through the secret.
    model = secret.get(
        "model"
    )


    # Cache the real credential.

    # If the configured model is missing or is itself a placeholder,
    # fall back to the known default model.
    _cached = {
        "api_key": secret[
            "api_key"
        ],
        "model": (
            model
            if model
            and model != PLACEHOLDER
            else DEFAULT_MODEL
        ),
    }


    return _cached


# ===========================================================================
# HTTP request to Cohere
# ===========================================================================

def _post(
    body: dict,
    api_key: str,
) -> dict:
    """Send exactly one request to Cohere's v2 rerank endpoint.

    This function deliberately does NOT retry.

    The caller handles HTTP/network failures and converts them into a
    fallback result.
    """

    # Construct the HTTP POST request.
    request = urllib.request.Request(

        # Cohere rerank endpoint.
        ENDPOINT,

        # Serialize the request body to JSON bytes.
        data=json.dumps(
            body
        ).encode(),

        # Explicit HTTP method.
        method="POST",

        # Required HTTP headers.
        headers={
            "Authorization": (
                f"Bearer {api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
            "Accept": (
                "application/json"
            ),
        },
    )


    # Execute the request.

    # If the HTTP status is not successful, urllib raises HTTPError.
    #
    # If the request times out/DNS fails/etc., another exception is raised.
    with urllib.request.urlopen(
        request,
        timeout=TIMEOUT_S,
    ) as response:

        # Parse Cohere's JSON response.
        return json.loads(
            response.read()
        )


# ===========================================================================
# Fallback
# ===========================================================================

def _fallback(
    passages: list[dict],
    top_n: int,
    note: str,
) -> dict:
    """Return vector-search order when Cohere cannot rerank.

    The fallback preserves the original candidate ordering.

    That ordering comes from Pinecone's vector similarity.

    The result explicitly says:

        reranked=False

    and includes the reason in:

        rerank_note
    """

    # Record the reason operationally.
    log.warning(
        "rerank skipped: %s",
        note,
    )


    # Keep the first top_n passages in their original order.
    return {
        "passages": passages[
            :top_n
        ],
        "reranked": False,
        "rerank_note": note,
    }


# ===========================================================================
# Main reranking function
# ===========================================================================

def rerank(
    query: str,
    passages: list[dict],
    top_n: int,
    read_secret,
) -> dict:
    """Rerank candidate passages using Cohere relevance.

    Parameters
    ----------
    query:
        Original semantic-search query.

    passages:
        Candidate passages returned by Pinecone.

        These should normally be the wider recall pool, e.g. 40 passages.

    top_n:
        Number of passages to keep after reranking.

    read_secret:
        Callable that returns the Cohere secret as a dictionary.


    Returns
    -------
    dict

    Successful reranking:

        {
            "passages": [...],
            "reranked": True,
            "rerank_note": ""
        }

    Fallback:

        {
            "passages": [...],
            "reranked": False,
            "rerank_note": "reason"
        }


    IMPORTANT:

    This function NEVER raises because of a reranking problem.

    Reranking improves ordering but is not required for a valid search.
    """

    # -----------------------------------------------------------------------
    # No candidates
    # -----------------------------------------------------------------------

    # There is nothing to rerank.

    # Return an explicit empty result rather than making a pointless
    # external API request.
    if not passages:
        return {
            "passages": [],
            "reranked": False,
            "rerank_note": (
                "no candidates to re-rank"
            ),
        }


    # -----------------------------------------------------------------------
    # STEP 1 — credentials
    # -----------------------------------------------------------------------

    # Read the Cohere credential.

    # This may return None when the Secrets Manager value is still the
    # deployment placeholder.
    try:
        creds = _credentials(
            read_secret
        )

    except Exception as exc:

        # Secrets Manager failure is not allowed to break semantic search.
        #
        # Pinecone already provided usable candidate passages.
        return _fallback(
            passages,
            top_n,
            (
                "Cohere secret unreadable "
                f"({type(exc).__name__})"
            ),
        )


    # No real key yet.

    # Use the vector-search order and report why reranking was skipped.
    if creds is None:
        return _fallback(
            passages,
            top_n,
            (
                "Cohere key not set: "
                "the secret still holds replace-me"
            ),
        )


    # -----------------------------------------------------------------------
    # STEP 2 — one Cohere call
    # -----------------------------------------------------------------------

    # Build the Cohere request.

    # `documents` contains the full candidate passage text plus headings.
    #
    # `top_n` is capped to the number of available candidates so Cohere
    # never receives an impossible request.
    body = {
        "model": creds[
            "model"
        ],
        "query": query,
        "documents": [
            document_text(
                passage
            )
            for passage in passages
        ],
        "top_n": min(
            top_n,
            len(passages),
        ),
    }


    try:

        # Exactly ONE request is made.

        # There is intentionally no retry loop.
        response = _post(
            body,
            creds[
                "api_key"
            ],
        )


    except urllib.error.HTTPError as exc:

        # HTTP 429 is especially important for the trial key.

        # Convert it into an explicit human-readable fallback reason.
        reason = (
            "rate limit — a trial key allows "
            "10 rerank calls a minute"
            if exc.code == 429

            # Authentication failure.
            else (
                "key rejected"
                if exc.code == 401

                # Other HTTP status codes don't need a specialized
                # explanation here.
                else ""
            )
        )


        # Fall back immediately rather than retrying.
        return _fallback(
            passages,
            top_n,
            (
                f"Cohere HTTP {exc.code}"
                + (
                    f": {reason}"
                    if reason
                    else ""
                )
            ),
        )


    except Exception as exc:

        # Handles failures such as:
        #
        #     timeout
        #     DNS failure
        #     connection failure
        #     malformed response
        #
        # The vector-search result is still usable, so return it.
        return _fallback(
            passages,
            top_n,
            (
                "Cohere call failed "
                f"({type(exc).__name__})"
            ),
        )


    # -----------------------------------------------------------------------
    # STEP 3 — map Cohere ranking back to original passages
    # -----------------------------------------------------------------------

    # Cohere returns indexes pointing into the original `documents` list.

    # Start with an empty final ranking.
    ranked = []


    # Iterate through Cohere's ranking results.
    for result in (
        response.get(
            "results"
        )
        or []
    ):

        # Index identifies which original passage Cohere ranked.
        index = result.get(
            "index"
        )


        # Validate the index before using it.

        # This is defensive programming against malformed/unexpected API
        # responses.
        if (
            isinstance(
                index,
                int,
            )
            and 0 <= index < len(passages)
        ):

            # Copy the original passage and attach Cohere's relevance score.
            #
            # The original vector score remains intact.
            ranked.append(
                {
                    **passages[index],
                    "rerank_score": result.get(
                        "relevance_score"
                    ),
                }
            )


    # -----------------------------------------------------------------------
    # Validate reranker response
    # -----------------------------------------------------------------------

    # If Cohere returned no usable ranking entries, do not return an empty
    # result.

    # The original vector candidates are still valid evidence.
    if not ranked:
        return _fallback(
            passages,
            top_n,
            "Cohere returned no usable results",
        )


    # Return the reranked passages.

    # Cohere's ordering is preserved, so the most relevant passage is first.
    return {
        "passages": ranked[
            :top_n
        ],
        "reranked": True,
        "rerank_note": "",
    }