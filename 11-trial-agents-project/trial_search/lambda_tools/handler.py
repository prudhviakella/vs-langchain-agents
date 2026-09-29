"""trial_search's tool Lambda — three tools, one per retrieval case.

This Lambda is the execution layer behind the trial_search specialist.

The high-level architecture is:

    trial_search agent
    (LangGraph / AgentCore Runtime)
              │
              │ MCP tool call
              │ SigV4 signed
              ▼
    AgentCore Gateway
              │
              ▼
    THIS LAMBDA
              │
              ├── semantic_search
              │       │
              │       ├── OpenAI embedding
              │       ├── Pinecone vector search
              │       ├── wide recall pool
              │       └── Cohere reranking
              │
              ├── expand_neighbors
              │       │
              │       ├── Neo4j finds neighbouring Chunk IDs
              │       └── Pinecone fetches their text
              │
              └── expand_table
                      │
                      ├── Pinecone fetches table summary
                      └── Pinecone query retrieves table fragments


THREE RETRIEVAL CASES
---------------------

1. semantic_search

    The normal entry point.

    User question
        │
        ▼
    OpenAI embedding
        │
        ▼
    Pinecone wide vector search
        │
        ▼
    Recall pool
        │
        ▼
    Cohere reranking
        │
        ▼
    top_k passages


2. expand_neighbors — Case A

    Used when a correctly retrieved passage is cut off at a chunk
    boundary.

    Existing chunk
        │
        ▼
    Neo4j NEXT traversal
        │
        ▼
    Neighbour chunk IDs
        │
        ▼
    Pinecone fetch(ids)
        │
        ▼
    Actual neighbour text


3. expand_table — Case C

    Used when semantic search returns a table_summary but the question
    requires exact values.

    table_summary chunk
        │
        ▼
    Pinecone fetch(summary)
        │
        ├── doc_id
        └── table_id
              │
              ▼
    Pinecone filtered query
              │
              ▼
    Actual table fragments


THE CONTRACT BETWEEN NEO4J AND PINECONE
---------------------------------------

Neo4j and Pinecone have deliberately different responsibilities.

Neo4j:

    "Which chunks are related?"

Pinecone:

    "What does this chunk contain?"

The join key is:

    chunk_id


Neo4j Chunk nodes intentionally do NOT contain the full text.

Pinecone contains the actual chunk text.

Therefore:

    Neo4j
       │
       │ chunk_id
       ▼
    Pinecone
       │
       ▼
    text


WHY NEIGHBOURS COME FROM NEO4J
------------------------------

Pinecone metadata contains next_id/prev_id, but following those links
would require repeated fetches:

    chunk 5
       │
       ▼
    read chunk 4
       │
       ▼
    read chunk 3
       │
       ▼
    read chunk 2

Neo4j can traverse the NEXT relationship in one query:

    chunk 5
       │
       ├── chunk 4
       ├── chunk 3
       ├── chunk 2
       └── ...

The traversal window can therefore be widened by changing one bounded
integer rather than creating additional network round trips.


WHY expand_table USES doc_id + table_id
---------------------------------------

`table_id` is derived from Docling's table self-reference and is
document-local.

Therefore the same table_id can appear in multiple documents.

Filtering only on:

    table_id

could accidentally return table fragments belonging to other clinical
trials.

The safe lookup is:

    doc_id + table_id + content_type=table

This ensures that table expansion stays inside the original document.


BUDGET OWNERSHIP
----------------

The Lambda itself does NOT own the per-turn retrieval budget.

The architecture is:

    Agent
      │
      ▼
    RetrievalMiddleware
      │
      │ enforces:
      │
      ├── search call budget
      ├── neighbour call budget
      ├── table call budget
      └── shared expansion token budget
      │
      ▼
    Gateway
      │
      ▼
    Lambda

The Lambda is stateless.

It simply respects values such as:

    max_tokens
    exclude_ids


EXPANSION DECISION
------------------

This Lambda does NOT decide whether an expansion is necessary.

The model decides:

    "This passage is incomplete."

or:

    "I need the exact table values."

The Lambda only executes the requested operation.

"""


# ===========================================================================
# Standard library imports
# ===========================================================================

# Used to deserialize secrets and serialize Lambda responses.
import json

# Used for Lambda/application logging.
import logging

# Used for environment variables such as secret IDs and index names.
import os


# ===========================================================================
# AWS / external dependencies
# ===========================================================================

# Used to access AWS Secrets Manager.
import boto3


# Cohere reranking implementation.
#
# semantic_search retrieves a broad recall pool and this function reduces
# it to the most relevant top_k passages.
from rerank import rerank


# Normalizes special symbols in retrieved text.
#
# Example:
#
#     "BP \uf0b3 150"
#
# becomes:
#
#     "BP ≥ 150"
from symbol_fonts import normalize


# ===========================================================================
# Logging
# ===========================================================================

# Lambda's root logger is used here because the function is a small
# execution handler rather than a larger application package.
log = logging.getLogger()

log.setLevel(
    logging.INFO
)


# ===========================================================================
# Retrieval configuration / safety limits
# ===========================================================================

# IMPORTANT:
#
# This embedding model MUST match the model used to create the vectors
# already stored in Pinecone.
#
# A model mismatch can still produce apparently valid vectors and rankings,
# making the problem particularly difficult to detect.
EMBED_MODEL = "text-embedding-3-small"


# Maximum neighbour expansion window.
#
# RetrievalMiddleware also clamps this value, but the Lambda keeps a
# hard limit as defense in depth.
HARD_MAX_WINDOW = 10


# Maximum number of final semantic-search results.
HARD_MAX_TOP_K = 20


# Size of the vector-search recall pool before Cohere reranking.
#
# The idea is:
#
#     vector search -> RECALL
#     Cohere        -> PRECISION
#
# A wider pool gives the reranker more candidates from which to choose.
#
# RERANK_POOL can be overridden through an environment variable.
RERANK_POOL = int(
    os.environ.get(
        "RERANK_POOL",
        "40",
    )
)


# Absolute upper bound for the recall pool.
#
# Prevents an environment/configuration mistake from creating an
# unexpectedly large Pinecone query.
HARD_MAX_POOL = 100


# Maximum number of table fragments that can be considered.
#
# This is another defense-in-depth bound.
HARD_MAX_FRAGMENTS = 50


# ===========================================================================
# Allowed content types
# ===========================================================================

# All content types currently represented in the Pinecone index.
#
# The Gateway cannot enforce this as a strict enum, so the Lambda validates
# it before spending an embedding/vector-search operation.
CONTENT_TYPES = (
    "text",
    "table",
    "table_summary",
    "figure",
    "formula",
)


# ===========================================================================
# Warm-container client caches
# ===========================================================================

# These clients are initialized lazily.

# Lambda containers can be reused for multiple invocations, so caching these
# clients avoids creating them repeatedly on warm invocations.
_openai = None
_index = None
_driver = None


# ===========================================================================
# Client / secret helpers
# ===========================================================================

def _secret(
    env_name: str,
) -> dict:
    """Load and decode a JSON secret from AWS Secrets Manager.

    `env_name` contains the environment-variable name holding the actual
    Secrets Manager SecretId.

    Example:

        OPENAI_SECRET_ID
        PINECONE_SECRET_ID
        NEO4J_SECRET_ID
    """

    # Create the Secrets Manager client.
    client = boto3.client(
        "secretsmanager"
    )

    # Read the SecretId from the environment and fetch the secret value.
    #
    # The secret is expected to contain JSON.
    return json.loads(
        client.get_secret_value(
            SecretId=os.environ[
                env_name
            ]
        )[
            "SecretString"
        ]
    )


# ===========================================================================
# OpenAI client
# ===========================================================================

def _get_openai():
    """Return the cached OpenAI client.

    The API key is loaded from Secrets Manager only when the client is first
    required.

    Subsequent warm Lambda invocations reuse the same client.
    """

    global _openai

    # Lazy initialization.
    if _openai is None:

        # Import the SDK only when it is actually required.
        from openai import OpenAI

        # Read the API key from Secrets Manager.
        _openai = OpenAI(
            api_key=_secret(
                "OPENAI_SECRET_ID"
            )[
                "api_key"
            ]
        )

    return _openai


# ===========================================================================
# Pinecone client
# ===========================================================================

def _get_index():
    """Return the cached Pinecone index client."""

    global _index

    # Create the Pinecone client only once per warm container.
    if _index is None:

        # Import lazily.
        from pinecone import Pinecone

        # Read the Pinecone API key from Secrets Manager.
        pc = Pinecone(
            api_key=_secret(
                "PINECONE_SECRET_ID"
            )[
                "api_key"
            ]
        )

        # Select the configured index.
        #
        # `rag-docs` is the default if PINECONE_INDEX is not provided.
        _index = pc.Index(
            os.environ.get(
                "PINECONE_INDEX",
                "rag-docs",
            )
        )

    return _index


# ===========================================================================
# Neo4j client
# ===========================================================================

def _get_driver():
    """Return the cached Neo4j driver.

    Neo4j credentials are loaded from Secrets Manager.

    The driver is reused across warm Lambda invocations.
    """

    global _driver

    if _driver is None:

        # Import lazily because only neighbor expansion requires Neo4j.
        from neo4j import GraphDatabase

        # Read the Neo4j connection details from Secrets Manager.
        s = _secret(
            "NEO4J_SECRET_ID"
        )

        # Create the Neo4j driver.
        _driver = GraphDatabase.driver(
            s["uri"],
            auth=(
                s.get(
                    "user",
                    "neo4j",
                ),
                s["password"],
            ),
        )

    return _driver


# ===========================================================================
# Shared passage shaping
# ===========================================================================

def _passage(
    chunk_id: str,
    meta: dict,
    origin: str,
    score=None,
) -> dict:
    """Convert a Pinecone record into the common Passage shape.

    All three tools return the same logical passage structure.

    This is important because the agent should not need different parsing
    logic depending on whether a passage came from:

        semantic_search
        expand_neighbors
        expand_table


    `origin` identifies how the passage was obtained:

        search
        neighbor
        table


    The actual text and headings are normalized before being returned.
    """

    return {

        # Stable chunk identifier used as the join key between Neo4j and
        # Pinecone.
        "chunk_id": chunk_id,

        # Source document/trial identifier.
        "doc_id": meta.get(
            "doc_id",
            "",
        ),

        # Vector similarity score.
        #
        # This is populated for semantic-search hits.
        #
        # Neighbor/table expansions are fetched explicitly by ID and
        # therefore normally have no vector similarity score.
        "score": score,

        # Actual passage text.
        #
        # normalize() converts encoded/special symbols into readable text.
        "text": normalize(
            meta.get(
                "text",
                "",
            )
        ),

        # Content type stored in Pinecone metadata.
        "content_type": meta.get(
            "content_type",
            "",
        ),

        # Normalize document headings as well.
        "headings": [
            normalize(h)
            for h in (
                meta.get(
                    "headings",
                    []
                )
                or []
            )
        ],

        # Source document page.
        "page": meta.get(
            "page"
        ),

        # Position of the chunk in document order.
        "position": meta.get(
            "position"
        ),

        # Table identifier, when applicable.
        "table_id": meta.get(
            "table_id",
            "",
        ),

        # Number of fragments associated with the source.
        "n_fragments": meta.get(
            "n_fragments"
        ),

        # Token count used by expansion-budget enforcement.
        "n_tokens": meta.get(
            "n_tokens"
        ),

        # How this passage entered the result.
        "origin": origin,
    }


# ===========================================================================
# Pinecone batch fetch
# ===========================================================================

def _fetch(
    ids: list[str],
) -> dict:
    """Fetch Pinecone vectors by ID in batches.

    This helper is primarily used for expansion operations.

    Why fetch by ID?

        Neo4j determines WHICH chunks are needed.

        Pinecone determines WHAT those chunks say.

    Therefore the chunk ID is the exact join between the two stores.
    """

    # Map:
    #
    #     chunk_id -> Pinecone vector
    #
    # This makes later lookups O(1) and preserves a simple interface for
    # the callers.
    found = {}


    # Pinecone limits the number of IDs accepted by a single fetch.
    #
    # Fetch in batches of 100.
    for start in range(
        0,
        len(ids),
        100,
    ):

        # Fetch one batch.
        page = _get_index().fetch(
            ids=ids[
                start:start + 100
            ]
        )

        # Merge returned vectors into the final dictionary.
        found.update(
            page.vectors
        )


    return found


# ===========================================================================
# TOOL 1 — semantic_search
# ===========================================================================

def semantic_search(
    args: dict,
) -> dict:
    """Perform semantic retrieval followed by Cohere reranking.

    Retrieval has two phases:

        Phase 1 — RECALL

            Pinecone retrieves a wider candidate pool.

        Phase 2 — PRECISION

            Cohere reranks those candidates and keeps top_k.

    Example:

        question
           │
           ▼
        embedding
           │
           ▼
        Pinecone top 40
           │
           ▼
        Cohere rerank
           │
           ▼
        top 8
    """

    # Clamp top_k to a safe range.
    #
    # This protects the backend even if the caller supplies an excessive
    # value.
    top_k = max(
        1,
        min(
            int(
                args.get(
                    "top_k",
                    8,
                )
            ),
            HARD_MAX_TOP_K,
        ),
    )


    # -----------------------------------------------------------------------
    # STEP 1 — validate request before spending an embedding call
    # -----------------------------------------------------------------------

    # If content_type is provided, it must be one of the known types.
    #
    # Otherwise a typo could silently produce zero matches and the model
    # could incorrectly conclude that the corpus contains no answer.
    if (
        args.get("content_type")
        and args["content_type"]
        not in CONTENT_TYPES
    ):

        return {
            "error": True,
            "detail": (
                f"content_type {args['content_type']!r} "
                "is not valid; use one of "
                f"{', '.join(CONTENT_TYPES)}, "
                "or omit it to search every type."
            ),
        }


    # -----------------------------------------------------------------------
    # Build Pinecone metadata filter
    # -----------------------------------------------------------------------

    # Build individual Pinecone filter clauses.
    clauses = []


    # Restrict by content type when requested.
    if args.get(
        "content_type"
    ):
        clauses.append(
            {
                "content_type": {
                    "$eq": args[
                        "content_type"
                    ]
                }
            }
        )


    # Restrict to a specific document when requested.
    if args.get(
        "doc_id"
    ):
        clauses.append(
            {
                "doc_id": {
                    "$eq": args[
                        "doc_id"
                    ]
                }
            }
        )


    # Build the final Pinecone filter.
    #
    # No clauses:
    #
    #     None
    #
    # One clause:
    #
    #     {"content_type": ...}
    #
    # Multiple clauses:
    #
    #     {"$and": [...]}
    flt = (
        {
            "$and": clauses
        }
        if len(clauses) > 1
        else (
            clauses[0]
            if clauses
            else None
        )
    )


    # -----------------------------------------------------------------------
    # STEP 2 — RECALL
    # -----------------------------------------------------------------------

    # Convert the natural-language query into the same embedding space
    # used when the Pinecone index was created.
    embedding = (
        _get_openai()
        .embeddings.create(
            model=EMBED_MODEL,
            input=args[
                "query"
            ],
        )
        .data[0]
        .embedding
    )


    # Retrieve a wider candidate pool than the final top_k.
    #
    # Example:
    #
    #     top_k = 8
    #     RERANK_POOL = 40
    #
    # Pinecone returns up to 40 candidates.
    pool = min(
        max(
            top_k,
            RERANK_POOL,
        ),
        HARD_MAX_POOL,
    )


    # Query Pinecone using the generated embedding.
    #
    # Metadata is included because the returned text, document information,
    # headings, position, etc. are needed to construct Passage objects.
    result = _get_index().query(
        vector=embedding,
        top_k=pool,
        include_metadata=True,
        filter=flt,
    )


    # Convert each Pinecone match into the common passage shape.
    #
    # At this stage the score is Pinecone's vector similarity score.
    candidates = [
        _passage(
            m.id,
            m.metadata or {},
            "search",
            m.score,
        )
        for m in result.matches
    ]


    # -----------------------------------------------------------------------
    # STEP 3 — PRECISION / RERANKING
    # -----------------------------------------------------------------------

    # Cohere reranks the wider candidate pool.
    #
    # The reranker receives the user's original query and the candidate
    # passages and selects the most relevant top_k.
    #
    # If reranking cannot run, rerank() handles the fallback to vector
    # order.
    ranked = rerank(
        args[
            "query"
        ],
        candidates,
        top_k,

        # The reranking implementation loads its secret lazily.
        read_secret=lambda: _secret(
            "COHERE_SECRET_ID"
        ),
    )


    # Return the reranked result together with the size of the recall pool.
    #
    # `candidates` is useful for retrieval observability:
    #
    #     candidates = recall pool
    #     results    = final passages
    return {
        **ranked,
        "candidates": len(
            candidates
        ),
    }


# ===========================================================================
# TOOL 2 — expand_neighbors
# ===========================================================================
#
# Case A:
#
# A semantically relevant chunk is correct but cut off at its boundary.
#
# Neo4j is used to identify neighbouring chunks because the graph contains
# explicit NEXT relationships between document chunks.


# Cypher path length cannot be passed as a normal Cypher parameter.
#
# Therefore `%d` is interpolated into the query.
#
# This is safe because `window` is first converted to an integer and
# bounded by HARD_MAX_WINDOW before the query is constructed.
_NEIGHBOURS = """
MATCH (seed:Chunk {chunkId: $id})
OPTIONAL MATCH path = (seed)-[:NEXT*1..%d]-(n:Chunk)
WITH seed, n, min(length(path)) AS distance
RETURN seed.position AS seed_position,
       n.chunkId AS chunk_id, n.n_tokens AS n_tokens,
       n.position AS position, distance
ORDER BY distance, position
"""


def expand_neighbors(
    args: dict,
) -> dict:
    """Expand around a chunk that appears cut off.

    The expansion order is:

        nearest neighbour
             ↓
        next nearest
             ↓
        next nearest
             ↓
        stop when token budget is exhausted


    IMPORTANT:

    The function does NOT skip an oversized nearby chunk and continue
    farther away.

    Doing that would create a gap in the text presented to the model.

    Example:

        chunk 5
          │
          ├── chunk 4  ← too large
          │
          └── chunk 3

    We do NOT skip chunk 4 and return chunk 3.

    The correct behavior is to stop at chunk 4.
    """

    # ID of the seed chunk that needs contextual expansion.
    chunk_id = args[
        "chunk_id"
    ]


    # Clamp the requested window to the Lambda's hard safety limit.
    window = max(
        1,
        min(
            int(
                args.get(
                    "window",
                    2,
                )
            ),
            HARD_MAX_WINDOW,
        ),
    )


    # Remaining shared expansion-token budget supplied by middleware.
    max_tokens = max(
        0,
        int(
            args.get(
                "max_tokens",
                0,
            )
        ),
    )


    # IDs already retrieved earlier in the same turn.
    #
    # The middleware uses this to prevent duplicate passages.
    exclude = set(
        args.get(
            "exclude_ids"
        )
        or []
    )


    # -----------------------------------------------------------------------
    # STEP 1 — identify neighbouring chunks using Neo4j
    # -----------------------------------------------------------------------

    # Neo4j is responsible for document/chunk relationships.
    #
    # The query returns:
    #
    #     - seed position
    #     - neighbour chunk ID
    #     - neighbour token count
    #     - neighbour position
    #     - graph distance
    #
    # All neighbouring IDs are obtained in one query.
    with _get_driver().session() as session:

        rows = [
            dict(row)
            for row in session.run(
                _NEIGHBOURS % window,
                parameters={
                    "id": chunk_id
                },
            )
        ]


    # If the seed does not exist in the graph, the expansion request is
    # invalid.
    if not rows:
        return {
            "error": True,
            "detail": (
                f"chunk_id {chunk_id!r} "
                "is not in the graph"
            ),
        }


    # Position of the original seed chunk.
    seed_position = rows[
        0
    ][
        "seed_position"
    ]


    # Remove:
    #
    #     - the seed itself
    #     - chunks already returned earlier
    #
    # The middleware normally handles duplicate prevention, but the Lambda
    # still respects the provided exclusion list.
    candidates = [
        row
        for row in rows
        if (
            row["chunk_id"]
            and row["chunk_id"]
            not in exclude
        )
    ]


    # -----------------------------------------------------------------------
    # STEP 2 — consume the token budget in nearest-first order
    # -----------------------------------------------------------------------

    # Selected neighbour records.
    chosen = []

    # Total expansion tokens consumed so far.
    used = 0

    # Default stopping reason.
    #
    # If the loop consumes the complete available window, this remains
    # "window".
    stopped_by = "window"


    # Process neighbours in the order returned by Neo4j:
    #
    #     distance ASC
    #     position ASC
    for row in candidates:

        # Token cost of this chunk.
        cost = int(
            row["n_tokens"]
            or 0
        )


        # If adding this chunk would exceed the remaining budget:
        #
        #     STOP
        #
        # Do not skip it and continue farther away.
        if (
            used + cost
            > max_tokens
        ):

            stopped_by = (
                "token_budget"
            )

            break


        # Chunk fits within the remaining budget.
        chosen.append(
            row
        )

        # Increase consumed token count.
        used += cost


    # Detect the document-edge case when the graph has no neighbour.
    #
    # This preserves the original behavior.
    if (
        stopped_by == "window"
        and len(rows) == 1
        and rows[0]["chunk_id"]
        is None
    ):
        stopped_by = (
            "document_edge"
        )


    # -----------------------------------------------------------------------
    # STEP 3 — fetch actual text from Pinecone
    # -----------------------------------------------------------------------

    # Neo4j gave us the IDs.

    # Pinecone gives us the actual text.
    vectors = _fetch(
        [
            row["chunk_id"]
            for row in chosen
        ]
    )


    # Convert each fetched Pinecone vector into the common Passage shape.
    passages = [
        _passage(
            row["chunk_id"],
            vectors[
                row["chunk_id"]
            ].metadata
            or {},
            "neighbor",
        )
        for row in chosen
        if row["chunk_id"]
        in vectors
    ]


    # Restore document order.

    # This ensures the model sees the context in natural reading order.
    passages.sort(
        key=lambda passage:
            passage["position"]
            if passage["position"]
            is not None
            else 0
    )


    # Return both the actual passages and expansion metadata.
    return {
        "passages": passages,
        "tokens_used": used,
        "stopped_by": stopped_by,
        "seed_position": seed_position,
        "window": window,
    }


# ===========================================================================
# TOOL 3 — expand_table
# ===========================================================================
#
# Case C:
#
# semantic_search found a table summary, but the model needs exact table
# values.


def expand_table(
    args: dict,
) -> dict:
    """Expand a table_summary into its actual table fragments.

    Flow:

        table_summary chunk
               │
               ▼
        Pinecone fetch(summary)
               │
               ├── doc_id
               ├── table_id
               └── summary vector
               │
               ▼
        filtered Pinecone query
               │
               ▼
        exact table fragments
    """

    # Chunk ID of the table summary.
    chunk_id = args[
        "chunk_id"
    ]


    # Remaining shared expansion-token budget.
    max_tokens = max(
        0,
        int(
            args.get(
                "max_tokens",
                0,
            )
        ),
    )


    # Previously retrieved chunk IDs.
    exclude = set(
        args.get(
            "exclude_ids"
        )
        or []
    )


    # -----------------------------------------------------------------------
    # STEP 1 — fetch the table summary
    # -----------------------------------------------------------------------

    # The summary contains the metadata needed to identify the exact table:
    #
    #     doc_id
    #     table_id
    #     n_fragments
    #     stored vector
    summary = _fetch(
        [chunk_id]
    ).get(
        chunk_id
    )


    # Unknown chunk ID.
    if summary is None:
        return {
            "error": True,
            "detail": (
                f"chunk_id {chunk_id!r} "
                "not found"
            ),
        }


    # Extract summary metadata.
    meta = summary.metadata or {}


    # The expansion entry point MUST be a table_summary.
    #
    # This prevents accidental use of expand_table on a normal text chunk.
    if (
        meta.get(
            "content_type"
        )
        != "table_summary"
        or not meta.get(
            "table_id"
        )
    ):

        return {
            "error": True,
            "detail": (
                f"{chunk_id!r} is "
                f"content_type="
                f"{meta.get('content_type')!r}, "
                "not a table_summary. "
                "expand_table only works from a summary; "
                "use expand_neighbors for other chunks."
            ),
        }


    # Number of table fragments expected for this summary.
    #
    # Clamp the value to the hard Lambda safety limit.
    n_fragments = max(
        1,
        min(
            int(
                meta.get(
                    "n_fragments"
                )
                or 1
            ),
            HARD_MAX_FRAGMENTS,
        ),
    )


    # -----------------------------------------------------------------------
    # STEP 2 — retrieve the exact table fragments
    # -----------------------------------------------------------------------

    # Use the table summary's own stored vector as the query vector.
    #
    # There is NO new embedding call here.
    #
    # The metadata filter provides the exact table scope:
    #
    #     same doc_id
    #     same table_id
    #     content_type == table
    #
    # This prevents fragments from similarly identified tables in other
    # documents from being returned.
    result = _get_index().query(
        vector=list(
            summary.values
        ),
        top_k=n_fragments,
        include_metadata=True,

        filter={
            "$and": [
                {
                    "doc_id": {
                        "$eq": meta[
                            "doc_id"
                        ]
                    }
                },
                {
                    "table_id": {
                        "$eq": meta[
                            "table_id"
                        ]
                    }
                },
                {
                    "content_type": {
                        "$eq": "table"
                    }
                },
            ]
        },
    )


    # Sort fragments by document position so the model sees the table
    # fragments in their original order.
    fragments = sorted(
        result.matches,
        key=lambda match:
            (
                match.metadata
                or {}
            ).get(
                "position",
                0,
            ),
    )


    # -----------------------------------------------------------------------
    # STEP 3 — apply the shared expansion token rule
    # -----------------------------------------------------------------------

    # Final table passages.
    passages = []

    # Tokens consumed so far.
    used = 0

    # Default stopping reason.
    stopped_by = "complete"


    # Process table fragments in document order.
    for match in fragments:

        # Do not return a chunk that the agent already has.
        if match.id in exclude:
            continue


        # Token cost of this table fragment.
        cost = int(
            (
                match.metadata
                or {}
            ).get(
                "n_tokens"
            )
            or 0
        )


        # Stop when adding this fragment would exceed the remaining
        # expansion budget.
        if (
            used + cost
            > max_tokens
        ):

            stopped_by = (
                "token_budget"
            )

            break


        # Convert the Pinecone result into the common Passage shape.
        passages.append(
            _passage(
                match.id,
                match.metadata or {},
                "table",
            )
        )


        # Track consumed expansion tokens.
        used += cost


    # Return the actual table passages plus expansion metadata.
    return {
        "passages": passages,
        "tokens_used": used,
        "stopped_by": stopped_by,
        "n_fragments": n_fragments,
    }


# ===========================================================================
# Tool dispatch
# ===========================================================================

# Map logical tool names to their Python implementations.
#
# AgentCore Gateway supplies the tool name at runtime.
_TOOLS = {
    "semantic_search": semantic_search,
    "expand_neighbors": expand_neighbors,
    "expand_table": expand_table,
}


# ===========================================================================
# AWS Lambda entrypoint
# ===========================================================================

def lambda_handler(
    event,
    context,
):
    """Dispatch an AgentCore Gateway tool call to the correct function.

    Gateway invokes this Lambda for all three retrieval tools.

    The tool name is obtained from:

        context.client_context.custom[
            "bedrockAgentCoreToolName"
        ]

    The Gateway may prefix the logical tool name, for example:

        trial-search-tools___expand_table

    Therefore dispatch uses suffix matching.
    """

    # -----------------------------------------------------------------------
    # Identify requested Gateway tool
    # -----------------------------------------------------------------------

    # AWS Lambda client context may not exist in every invocation/testing
    # environment.
    cc = getattr(
        context,
        "client_context",
        None,
    )


    # Read the Gateway-provided tool name.
    #
    # If client_context/custom is unavailable, use an empty string.
    name = (
        cc.custom.get(
            "bedrockAgentCoreToolName",
            "",
        )
        if cc
        and cc.custom
        else ""
    )


    # Gateway can prefix the logical tool name.
    #
    # Example:
    #
    #     trial-search-tools___expand_table
    #
    # The actual logical tool is:
    #
    #     expand_table
    #
    # `endswith()` allows both forms.
    matched = next(
        (
            tool
            for tool in _TOOLS
            if name.endswith(
                tool
            )
        ),
        None,
    )


    # Reject unknown tools rather than executing arbitrary functions.
    if matched is None:

        log.error(
            "unknown tool requested: %r",
            name,
        )

        return {
            "error": True,
            "detail": (
                f"unknown tool: {name}"
            ),
        }


    # Log the logical tool selected for execution.
    log.info(
        "dispatching to %s",
        matched,
    )


    try:

        # Execute the selected retrieval operation.

        # `event` contains the tool arguments supplied by the Gateway.
        result = _TOOLS[
            matched
        ](
            event
        )


    except KeyError as exc:

        # A missing required argument is converted into a structured tool
        # error instead of causing an opaque Lambda failure.
        return {
            "error": True,
            "detail": (
                f"missing required argument: {exc}"
            ),
        }


    # -----------------------------------------------------------------------
    # Normalize Lambda response to JSON
    # -----------------------------------------------------------------------

    # Neo4j/Pinecone/AWS SDKs can sometimes return values that the standard
    # JSON encoder does not understand directly.
    #
    # `default=str` converts such values to strings.
    #
    # The outer json.loads() converts the serialized string back into a
    # normal Python dictionary for the Lambda runtime.
    return json.loads(
        json.dumps(
            result,
            default=str,
        )
    )