"""The document structure layer.

Documents, sections and chunks — the shape of the corpus rather than its content.

Everything here is free. It comes from metadata the ingestion pipeline already
wrote, so there is no LLM call and nothing to hallucinate. That makes it the layer
worth building first: it is exact by construction, and it gives every extracted
claim in layer 3 a path back to the page it came from.

    (:Document {docId})-[:ABOUT]->(:Trial {nctId})
    (:Document)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(:Chunk)

The ABOUT edge is the join between this layer and the registry. Without it the two
are separate graphs in one database; with it, a traversal can start from a registry
fact and end at the passage of the protocol that discusses it.

READING-ORDER EDGES COME FROM THE CHUNK, NOT FROM RE-SORTING

    The RAG pipeline already computed the reading-order chain once, correctly, in
    `_finalise()` — `prev_id`/`next_id` are on every chunk's metadata, and that
    computation deliberately handles a case a plain sort cannot: a table's fragments
    and its summary can share the same `position` value, ordered relative to each
    other by a second, content-type-aware key. Re-deriving NEXT edges here by
    sorting on `position` alone throws that away — chunks sharing a position fall
    back to whatever order Pinecone's fetch happened to return them in, which can
    silently put a table's summary before its own fragments in the graph.

    So this module does not re-sort. It reads `prev_id`/`next_id` straight off each
    chunk and writes exactly that chain. One computation, trusted once, instead of
    two computations that can disagree.

SECTIONS GET A .key TOO

    A Section's heading — "Eligibility Criteria", "Adverse Events" — is exactly the
    kind of term a question would name, and store.find_chunks()/neighbours() reach
    a node by searching `.key`. Without one, a section is only reachable by walking
    down from its Document, never by naming it directly. Document itself is left
    without a `.key`: a document is reached via its Trial (the ABOUT edge) or by
    fetching its chunks directly, not by a user naming the document by title.

WHY load_structure() WRITES IN BATCHES, NOT ONE CALL PER CHUNK

    load_structure(session, records)
        |
        |-- STEP 1  documents          one UNWIND, all documents in one round trip
        |-- STEP 2  ABOUT edges        one UNWIND, only for documents with a nct_id
        |-- STEP 3  sections           dedup first, then one UNWIND per batch
        |-- STEP 4  chunks             one UNWIND per batch of BATCH_SIZE chunks
        |-- STEP 5  NEXT edges         one UNWIND per batch
        |
        v
    counts dict, same shape as before

    The previous version called session.run() once per chunk — for a corpus
    of 5764 chunks, that is roughly 11,800 individual round trips once
    Chunk-node writes, HAS_CHUNK edges, and NEXT edges are all counted
    separately. Against a remote, cloud-hosted Aura instance rather than a
    local database, each round trip pays real network latency regardless
    of how little work Neo4j actually does server-side for a MERGE that
    mostly matches existing nodes. Measured directly: that produced a
    21-minute run for a re-run of the same 5764 chunks, work that involves
    almost no actual writing the second time. Batching with UNWIND sends
    many rows in one round trip instead of one row per round trip — the
    server-side work per row is unchanged, but the network overhead that
    was actually dominating is paid once per batch instead of once per
    chunk.

WHAT THIS DOES NOT DO

    - It does not change what gets written — same nodes, same
      relationships, same properties, same counts dict shape as the
      unbatched version. This is a performance rewrite, not a behavior
      change.
    - It does not batch across documents in a way that loses the
      per-document ABOUT-link count. STEP 2 still reports linked_to_trial
      accurately, just via one collected result instead of one counted
      result per document.
    - It does not retry a failed batch partially. A batch is one
      transaction; if it fails, none of that batch's rows were written,
      which is the same all-or-nothing behavior a single MERGE call always
      had, just now covering many rows instead of one.
"""

import hashlib
import re
from collections import defaultdict

from .schema import normalise

# An NCT number has fixed form, so no model is needed to find one. Registry
# identifiers are the part of this domain that regex handles better than an LLM —
# faster, free, and incapable of inventing one.
#
# Note the lookarounds rather than \b. An underscore is a word character, so \b
# does not match between "NCT04368728" and "_Remdesivir" — and filenames of the form
# NCT04368728_Remdesivir_COVID.pdf are exactly the common case. Using \b here fails
# to find a single identifier in a corpus named that way, and fails silently: every
# document simply ends up unlinked.
NCT_PATTERN = re.compile(r"(?<![A-Za-z0-9])NCT\d{8}(?!\d)", re.IGNORECASE)

# Property names follow Neo4j convention — lowerCamelCase — rather than the
# snake_case the vector store uses. The two are joined on the *value* of a chunk id,
# not on the property name, so each side keeps its own conventions.
#
# nctId is spelled that way deliberately, matching Trial.nctId exactly, so a
# reader scanning Document's properties sees the same name they would look for on
# the node it joins to — one convention, not almost-one.
CONSTRAINTS = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.docId IS UNIQUE",
    "CREATE CONSTRAINT section_id  IF NOT EXISTS FOR (s:Section)  REQUIRE s.sectionKey IS UNIQUE",
    "CREATE CONSTRAINT chunk_id    IF NOT EXISTS FOR (c:Chunk)    REQUIRE c.chunkId IS UNIQUE",
    "CREATE INDEX chunk_page    IF NOT EXISTS FOR (c:Chunk)   ON (c.page)",
    "CREATE INDEX chunk_type    IF NOT EXISTS FOR (c:Chunk)   ON (c.contentType)",
    "CREATE INDEX section_key   IF NOT EXISTS FOR (s:Section) ON (s.key)",
]

# Rows per UNWIND call. Same value used throughout the student-facing
# dump/load tools, for the same reason: small enough that one query's
# parameter payload is never the bottleneck, large enough that this is
# genuinely a handful of round trips rather than hundreds.
BATCH_SIZE = 500


def create_constraints(session) -> None:
    """Uniqueness constraints for the structure layer."""
    for statement in CONSTRAINTS:
        try:
            session.run(statement)
        except Exception as exc:
            print(f"  DDL skipped: {str(exc)[:80]}", flush=True)


def find_nct_id(doc_id: str, source: str = "", text: str = "") -> str | None:
    """The NCT number for a document, if one can be found.

    Checked in order of reliability: the document id, then the filename, then the
    text. A number in the filename was put there deliberately; one in the body might
    be a reference to a different trial, which is why the body is the last resort
    rather than the first.
    """
    for candidate in (doc_id, source, text[:4000]):
        match = NCT_PATTERN.search(candidate or "")
        if match:
            return match.group(0).upper()
    return None


def section_key(doc_id: str, heading: str) -> str:
    """A stable key for a section within a document.

    Scoped by document: "Eligibility Criteria" is a different section in each
    protocol, and a shared node would merge twenty of them into one and make the
    hierarchy meaningless. This is the MERGE identity, not the `.key` search
    property below — the two answer different questions. A shared, document-scoped
    identity is exactly wrong for search, where the same heading recurring across
    twenty protocols is precisely what should be findable as one concept.
    """
    digest = hashlib.sha256(f"{doc_id}\x00{heading}".encode()).hexdigest()[:12]
    return f"{doc_id}:{digest}"


def _run_batched(session, query: str, rows: list[dict]) -> None:
    """UNWIND $batch through `query` in chunks of BATCH_SIZE, discarding
    any RETURN — used for writes where the write itself is all that
    matters, not what comes back.
    """
    for i in range(0, len(rows), BATCH_SIZE):
        session.run(query, batch=rows[i:i + BATCH_SIZE])


def load_structure(session, chunks: list[dict], verbose: bool = True) -> dict:
    """Write documents, sections and chunks, and link documents to their trial.

    Sections come from the first heading in each chunk's heading path. That is
    coarse — a nested subsection is folded into its parent — but it comes free from
    metadata already present, and a deeper hierarchy is only worth building once
    section-scoped queries prove useful.
    """
    by_document: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk.get("doc_id", "unknown")].append(chunk)

    counts = {"documents": 0, "sections": 0, "chunks": 0, "linked_to_trial": 0,
             "next_edges": 0}

    # Per-document facts, computed once up front — the NCT lookup is cheap
    # (string search over the opening few chunks), and every batch below
    # needs doc_id/source/nct_id together, so this is worked out once
    # rather than re-derived per batch.
    doc_facts: dict[str, dict] = {}
    for doc_id, doc_chunks in by_document.items():
        source = doc_chunks[0].get("source", "")
        opening = " ".join(c.get("text", "") for c in doc_chunks[:3])
        doc_facts[doc_id] = {
            "doc_id": doc_id, "source": source, "n_chunks": len(doc_chunks),
            "nct_id": find_nct_id(doc_id, source, opening),
        }

    # STEP 1 — documents, one UNWIND for every document in this call.
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (d:Document {docId: row.doc_id})
        SET d.sourceFile = row.source, d.nChunks = row.n_chunks,
            d.nctId = row.nct_id, d.origin = 'structure'
    """, list(doc_facts.values()))
    counts["documents"] = len(doc_facts)

    # STEP 2 — ABOUT edges. Only documents with a found nct_id are sent at
    # all — MATCHing a Trial that never had a Document row would just find
    # nothing, so filtering here is a smaller UNWIND, not a correctness
    # difference. Which doc_ids actually matched a real Trial node comes
    # back as one collected list rather than a per-row count, since a
    # MATCH that finds nothing drops that row from the result entirely —
    # collecting survivors and taking the length is the correct way to
    # count them, not summing a count(*) that would already have skipped
    # the misses.
    candidates = [row for row in doc_facts.values() if row["nct_id"]]
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i + BATCH_SIZE]
        result = session.run("""
            UNWIND $batch AS row
            MATCH (d:Document {docId: row.doc_id})
            MATCH (t:Trial {nctId: row.nct_id})
            MERGE (d)-[:ABOUT]->(t)
            RETURN collect(row.doc_id) AS linked
        """, batch=batch).single()
        counts["linked_to_trial"] += len(result["linked"]) if result else 0

    # STEP 3 — sections, deduplicated across the whole call first. The
    # original per-chunk loop tracked seen_sections per document as it
    # went; deduplicating up front here is the same logic, just computed
    # before batching rather than interleaved with per-chunk writes.
    sections: dict[str, dict] = {}
    for doc_id, doc_chunks in by_document.items():
        for chunk in doc_chunks:
            headings = chunk.get("headings") or []
            heading = headings[0] if headings else "(no heading)"
            key = section_key(doc_id, heading)
            if key not in sections:
                sections[key] = {"key": key, "heading": heading, "doc_id": doc_id,
                                 "search_key": normalise(heading)}
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (s:Section {sectionKey: row.key})
        SET s.heading = row.heading, s.docId = row.doc_id,
            s.key = row.search_key, s.origin = 'structure'
        WITH s, row MATCH (d:Document {docId: row.doc_id})
        MERGE (d)-[:HAS_SECTION]->(s)
    """, list(sections.values()))
    counts["sections"] = len(sections)

    # STEP 4 — chunks. Every chunk carries the section key it belongs to,
    # computed the same way as STEP 3 so the two agree on the identical
    # key for the identical (doc_id, heading) pair.
    chunk_rows = []
    for doc_id, doc_chunks in by_document.items():
        for chunk in doc_chunks:
            headings = chunk.get("headings") or []
            heading = headings[0] if headings else "(no heading)"
            chunk_rows.append({
                "chunk_id": chunk["chunk_id"], "doc_id": doc_id,
                "page": chunk.get("page"), "position": chunk.get("position"),
                "content_type": chunk.get("content_type"),
                "n_tokens": chunk.get("n_tokens"),
                "key": section_key(doc_id, heading),
            })
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (c:Chunk {chunkId: row.chunk_id})
        SET c.docId = row.doc_id, c.page = row.page, c.position = row.position,
            c.content_type = row.content_type, c.n_tokens = row.n_tokens,
            c.origin = 'structure'
        WITH c, row MATCH (s:Section {sectionKey: row.key})
        MERGE (s)-[:HAS_CHUNK]->(c)
    """, chunk_rows)
    counts["chunks"] = len(chunk_rows)

    # STEP 5 — reading-order edges, taken directly from each chunk's own
    # prev_id/next_id — not re-derived by sorting. See the module
    # docstring for why re-sorting on position alone can silently
    # misorder a table's fragments against its summary, which this avoids
    # by trusting the one place that ordering was already computed
    # correctly.
    next_rows = [{"a": chunk["chunk_id"], "b": chunk["next_id"]}
                for chunk in chunks if chunk.get("next_id")]
    _run_batched(session, """
        UNWIND $batch AS row
        MATCH (a:Chunk {chunkId: row.a}) MATCH (b:Chunk {chunkId: row.b})
        MERGE (a)-[:NEXT]->(b)
    """, next_rows)
    counts["next_edges"] = len(next_rows)

    if verbose:
        for key, value in counts.items():
            print(f"  {key:<18} {value}", flush=True)
        unlinked = counts["documents"] - counts["linked_to_trial"]
        if unlinked:
            # Worth naming: an unlinked document is one whose protocol content can
            # never be reached from a registry fact, which is half the point of
            # having both layers.
            print(f"  {unlinked} document(s) could not be linked to a registered "
                  "trial — no NCT number found, or not in the registry", flush=True)
    return counts
