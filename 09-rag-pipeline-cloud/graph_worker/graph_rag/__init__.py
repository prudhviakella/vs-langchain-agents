"""Graph retrieval over a corpus already indexed in a vector store.

Three layers, each with different provenance and cost:

    structure   Document, Section, Chunk        free, exact, from metadata
    registry    Trial and everything it links   free, authoritative, from the API
    extracted   protocol operational detail     one LLM call per chunk

Every node carries a `source` property, so a query can demand facts or accept
claims — and because the registry is ground truth for the fields it covers,
extraction accuracy is measurable rather than assumed.

The module is independent of the ingestion pipeline that built the vector index.
The only thing shared is the index, and the only identifier shared is `chunk_id`.
"""

from . import (accuracy, answer, chunks, config, extract, registry, schema,
               store, structure)

__all__ = ["accuracy", "answer", "chunks", "config", "extract", "registry",
           "schema", "store", "structure"]
