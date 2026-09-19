#!/usr/bin/env python3
"""Load a JSON dump (from export_neo4j.py) into a Neo4j database.

    NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io \
    NEO4J_USER=neo4j \
    NEO4J_PASSWORD=... \
    python import_neo4j.py --file graph_dump.json

FLOW

    read the json file  ->  nodes (with export_id) + relationships   (STEP 1)
        |
        v
    is the target database already non-empty?
        |                          |
        no                         yes, and no --force
        |                          |
        v                          v
    continue                  refuse — see WHY BELOW               (STEP 2)
        |
        v
    temporary index on _importId, one per label found                (STEP 3)
        |
        v
    CREATE every node, batched by label combination,                 (STEP 4)
    each carrying a temporary _importId
        |
        v
    CREATE every relationship, batched by type, matching              (STEP 5)
    endpoints via _importId
        |
        v
    remove _importId from every node, drop the temporary index        (STEP 6)

WHY THIS REFUSES ON A NON-EMPTY DATABASE BY DEFAULT

A fresh Aura instance is empty, and that is the case this is built for.
Running this against a database that already has data in it would add a
second, disconnected copy of the graph on top of whatever is already
there — nothing gets overwritten, but the result is confusing and hard to
undo by hand. --force skips this check for anyone who genuinely wants
that.

WHY _importId IS REMOVED AFTERWARD

Without this, every node in the target database would carry a leftover
property no node in the source database has — the graph would not
actually match what was exported, only look like it does until someone
runs a query that includes it. Removed in STEP 6 specifically so the
result is indistinguishable from a graph built natively, not from an
import.
"""

import argparse
import gzip
import json
import os

BATCH_SIZE = 500


def driver():
    """Same as export_neo4j.py's own driver() — see its comment for why
    this is copied here rather than imported from graph_rag.store.
    """
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")

    if not password:
        raise SystemExit(
            "NEO4J_PASSWORD is not set. Aura shows the password once when "
            "the instance is created and never again; if it is lost, reset "
            "it from the Aura console.")

    valid_schemes = ("bolt://", "neo4j://", "neo4j+s://", "neo4j+ssc://")
    if not uri.startswith(valid_schemes):
        raise SystemExit(
            f"NEO4J_URI is {uri!r}, which is not a real Neo4j connection "
            "URI. Set it to your Aura instance's URI (starts with "
            "neo4j+s://) or a local instance's (bolt://).")

    if "neo4j.io" in uri and not uri.startswith(("neo4j+s://", "neo4j+ssc://")):
        raise SystemExit(
            f"{uri} looks like an Aura instance but does not use the "
            "neo4j+s:// scheme. Aura requires an encrypted connection.")

    instance = GraphDatabase.driver(uri, auth=(user, password))
    instance.verify_connectivity()
    return instance


def load_dump(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        dump = json.load(f)
    for key in ("nodes", "relationships"):
        if key not in dump:
            raise SystemExit(
                f"{path} is missing {key!r} — this does not look like a "
                "file export_neo4j.py produced.")
    return dump


def _label_clause(labels: list[str]) -> str:
    """`:Label1:Label2` — labels cannot be parameterised in Cypher, so this
    goes straight into the query text. Backtick-quoted so a label
    containing an unusual character (a space, a reserved word) still
    produces valid Cypher rather than a syntax error.
    """
    return "".join(f"`{label}`" for label in labels)


def ensure_empty_or_forced(session, force: bool) -> None:
    """STEP 2 — refuse to run against a non-empty database unless told to."""
    count = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    if count > 0 and not force:
        raise SystemExit(
            f"This database already has {count} node(s). Importing on top "
            "of it would add a second, disconnected copy of the graph "
            "rather than replace or merge anything. Use an empty database, "
            "or pass --force if you're sure you want to add to what's "
            "already there.")


def create_temp_indexes(session, all_labels: set[str]) -> None:
    """STEP 3 — one index per label actually present, so STEP 5's
    relationship-matching does not fall back to a full scan per lookup.
    Safe to run every time: IF NOT EXISTS makes this a no-op on a re-run.
    """
    for label in all_labels:
        session.run(f"CREATE INDEX import_temp_{label.lower()} IF NOT EXISTS "
                    f"FOR (n:`{label}`) ON (n._importId)")


def drop_temp_indexes(session, all_labels: set[str]) -> None:
    for label in all_labels:
        session.run(f"DROP INDEX import_temp_{label.lower()} IF EXISTS")


def create_nodes(session, nodes: list[dict]) -> None:
    """STEP 4 — batched by label combination, since Cypher needs labels in
    the query text rather than as a parameter, and different nodes in the
    file can have different labels.
    """
    by_labels: dict[tuple, list[dict]] = {}
    for node in nodes:
        key = tuple(sorted(node["labels"]))
        by_labels.setdefault(key, []).append(node)

    created = 0
    for labels, group in by_labels.items():
        clause = _label_clause(labels)
        for i in range(0, len(group), BATCH_SIZE):
            batch = [{"export_id": n["export_id"], "properties": n["properties"]}
                     for n in group[i:i + BATCH_SIZE]]
            session.run(
                f"UNWIND $batch AS row "
                f"CREATE (n:{clause}) "
                f"SET n = row.properties, n._importId = row.export_id",
                batch=batch,
            )
            created += len(batch)
            print(f"  created {created} of {len(nodes)} node(s)...",
                 end="\r", flush=True)
    print(f"  created {created} of {len(nodes)} node(s)" + " " * 10)


def create_relationships(session, relationships: list[dict]) -> None:
    """STEP 5 — batched by relationship type, same reason as node labels."""
    by_type: dict[str, list[dict]] = {}
    for rel in relationships:
        by_type.setdefault(rel["type"], []).append(rel)

    created = 0
    for rel_type, group in by_type.items():
        for i in range(0, len(group), BATCH_SIZE):
            batch = group[i:i + BATCH_SIZE]
            session.run(
                f"UNWIND $batch AS row "
                f"MATCH (a {{_importId: row.start}}) "
                f"MATCH (b {{_importId: row.end}}) "
                f"CREATE (a)-[r:`{rel_type}`]->(b) "
                f"SET r = row.properties",
                batch=batch,
            )
            created += len(batch)
            print(f"  created {created} of {len(relationships)} relationship(s)...",
                 end="\r", flush=True)
    print(f"  created {created} of {len(relationships)} relationship(s)" + " " * 10)


def cleanup(session) -> None:
    """STEP 6 — remove the temporary property so the result matches the
    source graph exactly, not the source graph plus import scaffolding.
    """
    session.run("MATCH (n) WHERE n._importId IS NOT NULL REMOVE n._importId")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", required=True,
                    help="dump file from export_neo4j.py (.json or .json.gz)")
    ap.add_argument("--force", action="store_true",
                    help="import even if the target database already has data")
    args = ap.parse_args()

    print(f"reading {args.file}")
    dump = load_dump(args.file)
    print(f"  {dump['node_count']} node(s), {dump['relationship_count']} "
         f"relationship(s), exported {dump.get('exported_at', 'unknown')}")

    all_labels = {label for node in dump["nodes"] for label in node["labels"]}

    print("connecting to Neo4j")
    drv = driver()
    try:
        with drv.session() as session:
            ensure_empty_or_forced(session, args.force)
            create_temp_indexes(session, all_labels)
            create_nodes(session, dump["nodes"])
            create_relationships(session, dump["relationships"])
            cleanup(session)
            drop_temp_indexes(session, all_labels)
    finally:
        drv.close()

    print(f"\ndone — {dump['node_count']} node(s) and "
         f"{dump['relationship_count']} relationship(s) now in the database")


if __name__ == "__main__":
    main()
