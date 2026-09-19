#!/usr/bin/env python3
"""Dump every node and relationship in a Neo4j database to one JSON file.

    NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io \
    NEO4J_USER=neo4j \
    NEO4J_PASSWORD=... \
    python export_neo4j.py --out graph_dump.json

FLOW

    connect to Neo4j
        |
        v
    MATCH (n) RETURN n            ->  every node, given a fresh          (STEP 1)
                                       sequential export_id (0, 1, 2...)
        |
        v
    MATCH (a)-[r]->(b) RETURN r   ->  every relationship, referring      (STEP 2)
                                       to its endpoints by export_id,
                                       not by Neo4j's own internal id
        |
        v
    write ONE json file: {nodes: [...], relationships: [...]}            (STEP 3)

WHY export_id EXISTS AND NEO4J'S OWN elementId() DOES NOT APPEAR IN THE FILE

elementId() (or, on an older server, id()) is a handle into THIS database —
it means nothing in a student's own, separate Neo4j instance, and reusing
it there would either collide with unrelated nodes or simply refer to
nothing. export_id is assigned fresh during this export, purely to let
import_neo4j.py reconnect the right two nodes when it recreates
relationships; it exists only inside this file and inside the target
database for the few minutes an import takes, then import_neo4j.py
removes it.

WHAT THIS DOES NOT DO

    - It does not care what labels or relationship types this particular
      graph uses. Every node and relationship is exported exactly as
      found — this is not specific to Document/Section/Chunk/Trial, and
      keeps working if that schema changes later.
    - It does not export indexes or constraints. import_neo4j.py creates
      its own temporary index for the reconnection step and drops it
      afterward; anything the source database's schema needs (uniqueness
      constraints, for instance) is a one-time setup step on the target,
      not something this file carries.
    - It does not compress the output itself. Pass --gzip if the file is
      too large to share as-is.
"""

import argparse
import gzip
import json
import os
from datetime import datetime, timezone

# Same batch size as the Pinecone dump tools, for the same reason: small
# enough that one query's parameter list never becomes the bottleneck,
# large enough that this does not take one round trip per node.
BATCH_SIZE = 500


def driver():
    """Same validation as graph_rag.store.driver(), copied rather than
    imported — these scripts run standalone, on a student's machine that
    does not have the graph_rag package installed at all.
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


def export_graph(session) -> dict:
    # STEP 1 — every node, given a fresh sequential id.
    #
    # elementId(n) is read here ONLY as a lookup key for this run, to build
    # the id_map below — it never appears in the output file itself.
    id_map: dict[str, int] = {}
    nodes = []
    result = session.run("MATCH (n) RETURN elementId(n) AS eid, "
                         "labels(n) AS labels, properties(n) AS props")
    for row in result:
        export_id = len(nodes)
        id_map[row["eid"]] = export_id
        nodes.append({"export_id": export_id, "labels": row["labels"],
                      "properties": dict(row["props"])})
        if len(nodes) % 500 == 0:
            print(f"  read {len(nodes)} node(s)...", end="\r", flush=True)
    print(f"  read {len(nodes)} node(s) total" + " " * 10)

    # STEP 2 — every relationship, referring to endpoints by export_id.
    relationships = []
    result = session.run("MATCH (a)-[r]->(b) RETURN elementId(a) AS start_eid, "
                         "elementId(b) AS end_eid, type(r) AS rel_type, "
                         "properties(r) AS props")
    for row in result:
        relationships.append({
            "start": id_map[row["start_eid"]],
            "end": id_map[row["end_eid"]],
            "type": row["rel_type"],
            "properties": dict(row["props"]),
        })
        if len(relationships) % 500 == 0:
            print(f"  read {len(relationships)} relationship(s)...",
                 end="\r", flush=True)
    print(f"  read {len(relationships)} relationship(s) total" + " " * 10)

    return {
        "node_count": len(nodes),
        "relationship_count": len(relationships),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nodes": nodes,
        "relationships": relationships,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output file path, e.g. graph_dump.json")
    ap.add_argument("--gzip", action="store_true",
                    help="gzip-compress the output — .gz is added to --out "
                         "automatically if not already there")
    args = ap.parse_args()

    if args.gzip and not args.out.endswith(".gz"):
        args.out += ".gz"

    print("connecting to Neo4j")
    drv = driver()
    try:
        with drv.session() as session:
            dump = export_graph(session)
    finally:
        drv.close()

    # STEP 3 — write the file.
    payload = json.dumps(dump, indent=2, default=str).encode("utf-8")
    if args.gzip:
        payload = gzip.compress(payload)
    with open(args.out, "wb") as f:
        f.write(payload)

    size_mb = len(payload) / 1024 / 1024
    print(f"\nwrote {dump['node_count']} node(s), "
         f"{dump['relationship_count']} relationship(s) to {args.out} "
         f"({size_mb:.1f} MB{' gzipped' if args.gzip else ''})")


if __name__ == "__main__":
    main()
