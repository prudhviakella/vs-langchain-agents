#!/usr/bin/env python3
"""Create the Neo4j fulltext index that find_entity_by_name queries.

    python setup_neo4j.py          (run after the trial-graph/neo4j secret
                                    holds the real credentials)

    trial-graph/neo4j secret ──► connect ──► CREATE FULLTEXT INDEX ... IF NOT EXISTS
                                                  │
                                                  v
                                          db.awaitIndex — wait until ONLINE

WHY THIS SCRIPT EXISTS

lambda_tools/handler.py queries a fulltext index named trial_entity_names.
Nothing created it: not the RAG pipeline, not graph_rag, not deploy.py.
Every find_entity_by_name call would have failed with "no such fulltext
schema index" — the one tool the prompt tells the model to call FIRST.

WHAT IT INDEXES — CHECKED AGAINST THE REAL GRAPH

    Trial      nctId, briefTitle, officialTitle, acronym
    Sponsor    name
    Disease    name
    CRO        name
    Site       facility
    Drug       name        (the graph currently holds ZERO Drug nodes;
                            indexed anyway so it works once they exist)

A node lacking one of the properties is simply not indexed on it.

WHAT THIS DOES NOT DO

    It does not change data. Index creation is a schema operation, safe to
    re-run: IF NOT EXISTS makes a second run a no-op.
"""
import json
import sys

import boto3
from neo4j import GraphDatabase

SECRET_NAME = "trial-graph/neo4j"
INDEX = "trial_entity_names"

CREATE = f"""
CREATE FULLTEXT INDEX {INDEX} IF NOT EXISTS
FOR (n:Trial|Sponsor|Disease|CRO|Site|Drug)
ON EACH [n.nctId, n.briefTitle, n.officialTitle, n.acronym, n.name, n.facility]
"""


def credentials() -> dict:
    secret = json.loads(boto3.client("secretsmanager").get_secret_value(
        SecretId=SECRET_NAME)["SecretString"])
    if secret.get("uri", "replace-me") == "replace-me":
        sys.exit(f"{SECRET_NAME} still holds placeholders. Set it first:\n"
                 f"  aws secretsmanager put-secret-value --secret-id {SECRET_NAME} "
                 '--secret-string \'{"uri":"neo4j+s://...","user":"neo4j","password":"..."}\'')
    return secret


def ensure_index() -> None:
    s = credentials()
    with GraphDatabase.driver(s["uri"], auth=(s.get("user", "neo4j"), s["password"])) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            # STEP 1 — create (no-op if it exists)
            session.run(CREATE).consume()
            # STEP 2 — fulltext indexes populate asynchronously; wait
            session.run("CALL db.awaitIndex($name, 300)", parameters={"name": INDEX}).consume()
            state = session.run("SHOW FULLTEXT INDEXES YIELD name, state "
                                "WHERE name = $name RETURN state",
                                parameters={"name": INDEX}).single()
    print(f"  index {INDEX!r}: {state['state'] if state else 'MISSING'}")
    if not state or state["state"] != "ONLINE":
        sys.exit(f"index {INDEX!r} is not ONLINE")


if __name__ == "__main__":
    ensure_index()
