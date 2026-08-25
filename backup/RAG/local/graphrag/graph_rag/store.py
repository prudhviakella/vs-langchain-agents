"""Writing the graph to Neo4j and querying it.

Everything here is MERGE rather than CREATE, so a re-run converges on the same graph
instead of duplicating it. That property is what makes a rebuild safe after a failed
load, and it depends entirely on the uniqueness constraints below — without one, two
MERGE statements on the same key can both create a node.

Chunk nodes carry an id, a page and a document, and no text. The text lives in the
vector index, and storing it in both places would create two copies to keep in sync.
A traversal only needs to know *which* chunks to fetch.
"""

from collections import Counter

from . import config
from .schema import ENTITY_TYPES, RELATION_TYPES


def driver():
    """A verified Neo4j driver, for Aura or a local instance.

    Connectivity is checked here so a bad URI or password fails at connect time,
    rather than inside the first query where it arrives wrapped in a transaction
    error and reads like a Cypher problem.

    Aura needs the `neo4j+s://` scheme. Given `bolt://` it fails with a message
    about routing, which sends people looking at the wrong thing entirely — so
    that case is named explicitly below.
    """
    from neo4j import GraphDatabase

    if not config.NEO4J_PASSWORD:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. Aura shows the password once when the "
            "instance is created and never again; if it is lost, reset it from "
            "the Aura console.")

    aura = "neo4j.io" in config.NEO4J_URI
    if aura and not config.NEO4J_URI.startswith(("neo4j+s://", "neo4j+ssc://")):
        raise RuntimeError(
            f"{config.NEO4J_URI} looks like an Aura instance but does not use the "
            "neo4j+s:// scheme. Aura requires an encrypted connection, and a plain "
            "bolt:// URI fails with a routing error that has nothing to do with "
            "the real cause.")

    instance = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    instance.verify_connectivity()
    return instance


def create_constraints(session) -> None:
    """One uniqueness constraint per entity type, on the normalised key.

    These are what make MERGE idempotent, and they create the index that makes
    lookups by key fast. Both matter: without the constraint a concurrent load
    duplicates nodes, and without the index a traversal scans.
    """
    # Chunk, Document and Section constraints belong to the structure layer, and
    # the registry's to registry.py. Two modules declaring the same constraint is
    # how a schema drifts out of agreement with itself.
    for label in ENTITY_TYPES:
        session.run(f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.key IS UNIQUE")


def load(session, extractions: list[dict], verbose: bool = True) -> dict:
    """Write extracted entities and relations, linked to the chunks they came from.

    Chunk nodes are created by the structure layer, which runs first. This only
    MATCHes them — so an extraction referring to a chunk that was never loaded links
    nothing rather than creating a Chunk with no document and no section.

    Everything written here carries `source: "extracted"`, which is what separates
    these claims from the registry's facts.
    """
    counts: Counter = Counter()

    for extraction in extractions:
        chunk_id = extraction["chunk_id"]

        for entity in extraction["entities"]:
            # The label cannot be parameterised in Cypher, so it is interpolated.
            # That is safe only because it was checked against ENTITY_TYPES during
            # validation — never interpolate a label straight from a model.
            session.run(
                f"MERGE (e:{entity['type']} {{key: $key}}) "
                "SET e.name = $name, e.detail = coalesce(e.detail, $detail), "
                "    e.source = 'extracted' "
                "WITH e MATCH (c:Chunk {chunkId: $chunk_id}) "
                "MERGE (c)-[:MENTIONS]->(e)",
                key=entity["key"], name=entity["name"],
                detail=entity["detail"], chunk_id=chunk_id,
            )
            counts[entity["type"]] += 1

        for relation in extraction["relations"]:
            source_type, target_type = RELATION_TYPES[relation["type"]]
            session.run(
                f"MATCH (a:{source_type} {{key: $source}}) "
                f"MATCH (b:{target_type} {{key: $target}}) "
                f"MERGE (a)-[r:{relation['type']}]->(b) "
                # Counting mentions turns a repeated relation into evidence of
                # strength rather than a duplicate edge.
                "SET r.mentions = coalesce(r.mentions, 0) + 1, "
                "    r.source = 'extracted'",
                source=relation["source"], target=relation["target"],
            )
            counts[relation["type"]] += 1

    if verbose:
        for key, value in counts.most_common():
            print(f"    {key:<16} {value}", flush=True)
    return dict(counts)


def summary(session) -> dict:
    """Node and relationship counts, for checking a load did what it claims."""
    nodes = {row["label"]: row["n"] for row in session.run(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label, count(*) AS n ORDER BY n DESC")}
    relationships = {row["type"]: row["n"] for row in session.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC")}
    return {"nodes": nodes, "relationships": relationships}


def clear(session) -> None:
    """Delete every node and relationship. Only for rebuilding from scratch."""
    session.run("MATCH (n) DETACH DELETE n")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def find_chunks(session, terms: list[str], hops: int = 1,
                limit: int = 10) -> list[dict]:
    """Chunks reachable from any entity whose key contains one of the terms.

    Entities are matched by substring rather than embedded, which is the method's
    main weakness as well as its precision: a question naming no entity in the graph
    retrieves nothing at all, where dense retrieval would still return its best
    guess. Graph retrieval is exact and brittle; vector retrieval is fuzzy and always
    answers. Running both is why production systems keep two stores.
    """
    if not terms:
        return []

    # `hops` is interpolated because Cypher cannot parameterise a path length. It is
    # forced to an int first — never interpolate a caller's string into a query.
    hops = max(0, int(hops))
    rows = session.run(f"""
        MATCH (seed) WHERE NOT seed:Chunk AND seed.key IS NOT NULL
          AND any(t IN $terms WHERE seed.key CONTAINS t)
        MATCH (seed)-[*0..{hops}]-(near) WHERE NOT near:Chunk
        MATCH (near)<-[:MENTIONS]-(c:Chunk)
        RETURN DISTINCT c.chunkId AS chunkId, c.page AS page,
               collect(DISTINCT near.name)[..5] AS via
        LIMIT $limit
    """, terms=terms, limit=limit)
    return [dict(row) for row in rows]


def neighbours(session, name_fragment: str, limit: int = 20) -> list[dict]:
    """What an entity connects to. The query a vector store cannot express."""
    rows = session.run("""
        MATCH (a)-[r]->(b) WHERE NOT a:Chunk AND NOT b:Chunk
          AND a.key CONTAINS $fragment
        RETURN a.name AS source, type(r) AS relation, b.name AS target,
               labels(b)[0] AS target_type, r.mentions AS mentions
        ORDER BY mentions DESC LIMIT $limit
    """, fragment=name_fragment.lower(), limit=limit)
    return [dict(row) for row in rows]


def shared_neighbours(session, limit: int = 20) -> list[dict]:
    """Pairs of entities connected through a common third.

    A two-hop join: which trials share a sponsor, which drugs treat the same
    condition. There is no embedding of "shares a sponsor with" — this question has
    no vector formulation at all, which is the clearest argument for the graph.
    """
    rows = session.run("""
        MATCH (a)-[r1]->(shared)<-[r2]-(b)
        WHERE id(a) < id(b) AND NOT shared:Chunk
        RETURN labels(a)[0] AS a_type, a.name AS a,
               shared.name AS via, type(r1) AS relation,
               labels(b)[0] AS b_type, b.name AS b
        LIMIT $limit
    """, limit=limit)
    return [dict(row) for row in rows]


def hubs(session, limit: int = 20) -> list[dict]:
    """The most connected entities.

    Both a useful view and a sanity check: if every entity has degree 1, the graph is
    a pile of isolated nodes and nothing will traverse.
    """
    rows = session.run("""
        MATCH (e)-[r]-() WHERE NOT e:Chunk
        RETURN labels(e)[0] AS type, e.name AS name, count(r) AS degree
        ORDER BY degree DESC LIMIT $limit
    """, limit=limit)
    return [dict(row) for row in rows]


def communities(session, min_size: int = 2) -> list[dict]:
    """Connected components of the entity graph, largest first.

    Components are clustering at one level. Real GraphRAG uses Leiden for a
    hierarchy, which needs the Graph Data Science plugin; components need nothing and
    keep the mechanism visible, which is the better trade until communities have been
    shown to help on a given corpus.
    """
    rows = session.run("""
        MATCH (e) WHERE NOT e:Chunk
        OPTIONAL MATCH path = (e)-[*1..2]-(other)
        WHERE NOT other:Chunk
        WITH e, collect(DISTINCT other.name) AS members
        WHERE size(members) >= $min_size
        RETURN labels(e)[0] AS type, e.name AS seed,
               size(members) AS size, members[..25] AS members
        ORDER BY size DESC
    """, min_size=min_size)
    return [dict(row) for row in rows]
