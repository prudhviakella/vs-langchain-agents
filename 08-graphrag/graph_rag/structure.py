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

    for doc_id, doc_chunks in by_document.items():
        source = doc_chunks[0].get("source", "")
        # Look for the NCT number in the opening chunks, where a protocol states it.
        opening = " ".join(c.get("text", "") for c in doc_chunks[:3])
        nct_id = find_nct_id(doc_id, source, opening)

        session.run("""
            MERGE (d:Document {docId: $doc_id})
            SET d.sourceFile = $source, d.nChunks = $n_chunks,
                d.nctId = $nct_id, d.origin = 'structure'
        """, doc_id=doc_id, source=source, n_chunks=len(doc_chunks), nct_id=nct_id)
        counts["documents"] += 1

        # The join. Only created when the registry already holds that trial, so a
        # document naming an unregistered NCT number does not conjure a Trial node
        # with no facts attached.
        if nct_id:
            result = session.run("""
                MATCH (d:Document {docId: $doc_id})
                MATCH (t:Trial {nctId: $nct_id})
                MERGE (d)-[:ABOUT]->(t)
                RETURN count(*) AS linked
            """, doc_id=doc_id, nct_id=nct_id).single()
            if result and result["linked"]:
                counts["linked_to_trial"] += 1

        seen_sections = set()
        for chunk in doc_chunks:
            headings = chunk.get("headings") or []
            heading = headings[0] if headings else "(no heading)"
            key = section_key(doc_id, heading)

            if key not in seen_sections:
                session.run("""
                    MERGE (s:Section {sectionKey: $key})
                    SET s.heading = $heading, s.docId = $doc_id,
                        s.key = $search_key, s.origin = 'structure'
                    WITH s MATCH (d:Document {docId: $doc_id})
                    MERGE (d)-[:HAS_SECTION]->(s)
                """, key=key, heading=heading, doc_id=doc_id,
                     search_key=normalise(heading))
                seen_sections.add(key)
                counts["sections"] += 1

            # Chunks carry an id, a page and a type — and no text. The text lives in
            # the vector store; two copies would need keeping in sync, and a
            # traversal only needs to know which chunks to fetch.
            session.run("""
                MERGE (c:Chunk {chunkId: $chunk_id})
                SET c.docId = $doc_id, c.page = $page, c.position = $position,
                    c.content_type = $content_type, c.n_tokens = $n_tokens,
                    c.origin = 'structure'
                WITH c MATCH (s:Section {sectionKey: $key})
                MERGE (s)-[:HAS_CHUNK]->(c)
            """, chunk_id=chunk["chunk_id"], doc_id=doc_id,
                 page=chunk.get("page"), position=chunk.get("position"),
                 content_type=chunk.get("content_type"),
                 n_tokens=chunk.get("n_tokens"), key=key)
            counts["chunks"] += 1

    # Reading-order edges, taken directly from each chunk's own prev_id/next_id —
    # not re-derived by sorting. See the module docstring for why re-sorting on
    # position alone can silently misorder a table's fragments against its summary,
    # which this avoids by trusting the one place that ordering was already
    # computed correctly.
    for chunk in chunks:
        next_id = chunk.get("next_id")
        if not next_id:
            continue
        session.run("""
            MATCH (a:Chunk {chunkId: $a}) MATCH (b:Chunk {chunkId: $b})
            MERGE (a)-[:NEXT]->(b)
        """, a=chunk["chunk_id"], b=next_id)
        counts["next_edges"] += 1

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
