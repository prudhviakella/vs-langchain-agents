#!/usr/bin/env python3
"""Create the Neo4j fulltext index both agents' name lookups query.

    python setup_neo4j.py          (run after the trial-graph/neo4j secret
                                    holds the real credentials)

    trial-graph/neo4j secret ──► connect
                                    │
                                    ├─ index exists with another analyzer? ──► DROP it
                                    v
                         CREATE FULLTEXT INDEX ... IF NOT EXISTS  (analyzer: english)
                                    │
                                    v
                         db.awaitIndex — wait until ONLINE

WHO QUERIES IT

    trial_graph   lambda_tools/handler.py  find_entity_by_name
    trial_search  lambda_tools/handler.py  resolve_trial

WHAT IT INDEXES — CHECKED AGAINST THE REAL GRAPH

    Trial      nctId, briefTitle, officialTitle, acronym
    Sponsor    name
    Disease    name
    CRO        name
    Site       facility
    Drug       name        (the graph currently holds ZERO Drug nodes;
                            indexed anyway so it works once they exist)

A node lacking one of the properties is simply not indexed on it.

WHY THE english ANALYZER

The index was first created with Neo4j's default, standard-no-stop-words.
It indexes every word, so filler words in a question matched real titles:

    "a name that matches nothing xyz"  -> 8 trials   ("a" is in 15 of 20 titles)
    "the glaucoma trial"               -> 5 trials   ("the" in 12)

Lucene's english analyzer drops English stop words (a, the, that, of, in,
with ...) and stems ("trials" -> "trial"), at index time AND query time —
queryNodes analyzes the query with the index's own analyzer. The filler
words then match nothing. This is the language's stop list, inside Lucene;
nothing about this corpus is written here.

WHY DROP, AND NOT ONLY "IF NOT EXISTS"

An index's analyzer cannot be changed in place. IF NOT EXISTS would keep
the old index and silently ignore the new setting, so an existing index
with a different analyzer is dropped and created again. Name lookups fail
for the few seconds until the new index is ONLINE.

WHAT THIS DOES NOT DO

    It does not change data. Index creation is a schema operation, safe to
    re-run: when the index already exists with the english analyzer, a
    second run changes nothing.
"""
import json
import sys

import boto3
from neo4j import GraphDatabase

SECRET_NAME = "trial-graph/neo4j"
INDEX = "trial_entity_names"
ANALYZER = "english"

CREATE = f"""
CREATE FULLTEXT INDEX {INDEX} IF NOT EXISTS
FOR (n:Trial|Sponsor|Disease|CRO|Site|Drug)
ON EACH [n.nctId, n.briefTitle, n.officialTitle, n.acronym, n.name, n.facility]
OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{ANALYZER}'}}}}
"""

CURRENT = """
SHOW FULLTEXT INDEXES YIELD name, state, options
WHERE name = $name
RETURN state, options
"""


def credentials() -> dict:
    secret = json.loads(boto3.client("secretsmanager").get_secret_value(
        SecretId=SECRET_NAME)["SecretString"])
    if secret.get("uri", "replace-me") == "replace-me":
        sys.exit(f"{SECRET_NAME} still holds placeholders. Set it first:\n"
                 f"  aws secretsmanager put-secret-value --secret-id {SECRET_NAME} "
                 '--secret-string \'{"uri":"neo4j+s://...","user":"neo4j","password":"..."}\'')
    return secret


def _analyzer(row) -> str | None:
    """The analyzer an existing index was created with, from SHOW's options map."""
    if row is None:
        return None
    return ((row["options"] or {}).get("indexConfig") or {}).get("fulltext.analyzer")


def ensure_index() -> None:
    s = credentials()
    with GraphDatabase.driver(s["uri"], auth=(s.get("user", "neo4j"), s["password"])) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            # STEP 1 — an index with another analyzer cannot be altered; drop it
            existing = session.run(CURRENT, parameters={"name": INDEX}).single()
            found = _analyzer(existing)
            if existing is not None and found != ANALYZER:
                print(f"  index {INDEX!r} uses analyzer {found!r}; recreating with {ANALYZER!r}")
                session.run(f"DROP INDEX {INDEX}").consume()

            # STEP 2 — create (no-op if it already exists with the right analyzer)
            session.run(CREATE).consume()

            # STEP 3 — fulltext indexes populate asynchronously; wait
            session.run("CALL db.awaitIndex($name, 300)", parameters={"name": INDEX}).consume()
            row = session.run(CURRENT, parameters={"name": INDEX}).single()

    state, analyzer = (row["state"], _analyzer(row)) if row else ("MISSING", None)
    print(f"  index {INDEX!r}: {state}, analyzer {analyzer!r}")
    if state != "ONLINE" or analyzer != ANALYZER:
        sys.exit(f"index {INDEX!r} is not ONLINE with analyzer {ANALYZER!r}")


if __name__ == "__main__":
    ensure_index()
