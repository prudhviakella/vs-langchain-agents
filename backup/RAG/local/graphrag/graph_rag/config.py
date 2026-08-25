"""Configuration for the graph layer.

This module is independent of the RAG ingestion pipeline. It does not import it and
does not need its source — the only thing shared between them is a Pinecone index,
and the only identifier shared is `chunk_id`.

That seam is deliberate. The vector store is the source of chunk text; this module
reads from it and writes structure to Neo4j. Neither owns the other, and either can
be rebuilt without touching the other's code.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Neo4j
#
# The URI scheme decides the connection, and getting it wrong is the most common
# first failure:
#
#   neo4j+s://xxxx.databases.neo4j.io     Aura. Encrypted, certificate verified.
#   bolt://localhost:7687                 local Docker. Unencrypted.
#
# Aura will refuse a plain `bolt://` connection, and the error talks about routing
# rather than about encryption — which sends people looking in the wrong place.
#
# Aura gives you a credentials file when the instance is created. It contains the
# URI, the username and a generated password, and the password is shown once and
# never again. Put those in the environment; never in a source file.
# ─────────────────────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ─────────────────────────────────────────────────────────────────────────────
# Pinecone — read only
#
# This module never writes to the index. It reads chunk text to extract entities
# from, and reads it again at query time to turn a traversal into an answer. The
# index name and embedding model must match whatever built it, which read_manifest()
# in the ingestion package enforces on its side; here a dimension mismatch surfaces
# as an obvious error rather than as wrong results.
# ─────────────────────────────────────────────────────────────────────────────
INDEX_NAME = os.getenv("INDEX_NAME", "rag-docs")
NAMESPACE = os.getenv("NAMESPACE", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

# One call per chunk, so this is the dominant cost of building a graph — roughly
# $1.50 across a 3,000-chunk corpus at gpt-4o-mini. Cached by content, so it is paid
# once per chunk per schema.
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gpt-4o-mini")

# Community summaries and final answers.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ─────────────────────────────────────────────────────────────────────────────
# Cache
#
# Keyed on (model, prompt, chunk text). The prompt is part of the key because
# changing the schema changes the output — a cache that ignored it would return
# extractions made under a schema that no longer exists.
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(os.getenv("GRAPH_CACHE_DIR", ".cache/graph"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
