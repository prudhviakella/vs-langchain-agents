"""Build the graph for one document. The Stage 2 Batch job container entrypoint.

    python build_graph.py --bucket my-bucket --doc-id nct03164772-heart-failure

    structure   Document -> Section -> Chunk, from chunks.json          free
    registry    Trial, Sponsor, Drug, ... from ClinicalTrials.gov        free
    link        (:Document)-[:ABOUT]->(:Trial)                          needs both

Separate from Stage 1 deliberately. Stage 1 is the expensive, Docling-heavy
parse — minutes per document, real compute cost. Stage 2 is a few Cypher
writes and one cached HTTP call — seconds. Coupling them means a transient
Neo4j connection failure loses a nine-minute parse; separating them means
the graph can be rebuilt entirely — a schema fix, a new node type — without
re-parsing a single PDF.

Reads chunks.json from S3, not from Pinecone. graph_rag.chunks.load_chunks()
exists for pulling chunks back out of Pinecone's list()/fetch() API, which
needs the SDK-version-tolerant handling that API requires — Stage 1 already
wrote the same data to S3 as a plain JSON file, so reading it back is one
get_object call with no pagination and no SDK-shape quirks to work around.
"""

# MUST run before `from graph_rag import ...` — graph_rag.config reads every
# NEO4J_* and RUN_* setting with os.getenv(...) at import time, the same
# constraint as the Stage 1 worker and for the identical reason: setting an
# environment variable after the module holding it has already imported has
# no effect, and nothing warns you when that happens.
import os


def _bootstrap_config_from_parameter_store() -> None:
    prefix = os.environ.get("PARAM_PREFIX")
    if not prefix:
        return
    import boto3
    ssm = boto3.client("ssm")
    paginator = ssm.get_paginator("get_parameters_by_path")
    loaded = []
    for page in paginator.paginate(Path=prefix, Recursive=False):
        for param in page["Parameters"]:
            key = param["Name"].rsplit("/", 1)[-1]
            if key.startswith(("JOBDEF_", "QUEUE_")):
                continue
            os.environ[key] = param["Value"]
            loaded.append(key)
    print(f"loaded {len(loaded)} setting(s) from Parameter Store: "
          f"{', '.join(sorted(loaded))}", flush=True)


_bootstrap_config_from_parameter_store()

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import sys
import time
from datetime import datetime, timezone

DOCS_PREFIX = "docs"


def flatten(record: dict) -> dict:
    """A chunks.json record, as {"text": ..., "meta": {...}}, into the flat
    shape structure.load_structure() actually reads.

    THIS IS NOT A STYLE CHOICE

    graph_rag.structure.load_structure() was built against chunks pulled
    back out of Pinecone via graph_rag.chunks.load_chunks(), which returns
    Pinecone's own metadata dict directly — flat, with `doc_id`, `text`, and
    `headings` all as top-level keys. rag.inspect.write_chunk_report() writes
    a DIFFERENT shape to chunks.json: `text` at the top level, everything
    else nested one level down under `meta`.

    Passing chunks.json's records to load_structure() unflattened does not
    raise. `chunk.get("doc_id", "unknown")` silently returns "unknown" for
    every single chunk, and the entire document collapses into one Document
    node named "unknown" — confirmed by reading rag/chunking.py's own
    `{**meta, "text": ...}` pattern used internally for a different purpose,
    which is the same flattening this function performs.
    """
    return {**record["meta"], "text": record["text"]}


def load_chunks_from_s3(bucket: str, doc_id: str) -> list[dict]:
    """Read the full chunk records Stage 1 wrote specifically for this —
    NOT chunks.json.

    chunks.json (written by rag.inspect.write_chunk_report) is a
    deliberately metadata-only diagnostic report: it flattens each
    record's meta dict to the top level and explicitly drops the
    top-level text field, since the full text already lives in the
    companion chunks.md for a human to read. flatten() below still
    expects the original {"text": ..., "meta": {...}} shape — correctly,
    since that IS the real in-memory shape build_records() produces — but
    chunks.json was never that shape once written to disk. Confirmed
    directly: the first real run against actual data failed with
    `FAILED: 'meta'`, because chunks.json has no top-level text and no
    nested meta wrapper to find. graph_chunks.json is written by
    ingest.py's upload_graph_chunks() specifically to give this function
    the real, unreshaped shape — no reason to reshape flatten() itself,
    only to point this line at the file that actually matches what it
    already correctly expects.
    """
    import boto3
    s3 = boto3.client("s3")
    key = f"{DOCS_PREFIX}/{doc_id}/graph_chunks.json"
    obj = s3.get_object(Bucket=bucket, Key=key)
    records = json.loads(obj["Body"].read())
    return [flatten(r) for r in records]


def build_one(bucket: str, doc_id: str, session, audit_table: str | None = None) -> dict:
    """Run structure, registry, and link for one document.

    Returns a result dict rather than raising, matching Stage 1's own
    ingest_one() — one bad document must not fail the Batch job in a way
    that looks identical to every other kind of failure in the audit trail.
    """
    from graph_rag import config as gconfig
    from graph_rag import registry, structure

    started = time.time()
    print(f"\n{doc_id}: building graph", flush=True)

    try:
        chunks = load_chunks_from_s3(bucket, doc_id)
        print(f"  {len(chunks)} chunks loaded from s3://{bucket}/"
              f"{DOCS_PREFIX}/{doc_id}/chunks.json", flush=True)

        structure.create_constraints(session)
        struct_counts = structure.load_structure(session, chunks, verbose=True)

        nct_id = None
        if gconfig.RUN_REGISTRY:
            registry.create_constraints(session)
            opening = " ".join(c.get("text", "") for c in chunks[:3])
            nct_id = structure.find_nct_id(doc_id, chunks[0].get("source", ""), opening)
            if nct_id:
                registry.load_trials(session, [nct_id], verbose=True)
            else:
                print(f"  no NCT number found in {doc_id} — registry layer skipped "
                      "for this document", flush=True)

        # The link only forms on a SECOND pass: Trial nodes did not exist
        # when load_structure ran above (registry had not written them yet).
        # structure.load_structure's ABOUT edge uses MATCH, not MERGE, on
        # the Trial side deliberately — a document naming an unregistered
        # NCT number should not conjure a Trial node with no facts attached.
        # See structure.py's own docstring for the full reasoning; this
        # second call is what makes that design work as one clean step
        # rather than something a notebook has to remember to do twice.
        if nct_id:
            link_counts = structure.load_structure(session, chunks, verbose=False)
        else:
            link_counts = struct_counts

        elapsed = round(time.time() - started, 1)
        print(f"done in {elapsed}s: {struct_counts['documents']} document(s), "
              f"{struct_counts['sections']} section(s), "
              f"{struct_counts['chunks']} chunk(s), "
              f"{link_counts['linked_to_trial']} linked to a trial", flush=True)

        _audit(audit_table, doc_id, "COMPLETED", elapsed,
              documents=struct_counts["documents"],
              linked_to_trial=link_counts["linked_to_trial"])

        return {"doc_id": doc_id, "status": "ok", "seconds": elapsed,
               **link_counts}

    except Exception as exc:
        elapsed = round(time.time() - started, 1)
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        _audit(audit_table, doc_id, "FAILED", elapsed, error=str(exc)[:400])
        return {"doc_id": doc_id, "status": "failed", "seconds": elapsed,
               "error": str(exc)[:200]}


def _audit(audit_table: str | None, doc_id: str, status: str, elapsed: float,
          **fields) -> None:
    """A minimal audit write, deliberately not importing rag.audit.Audit —
    that class lives in the Stage 1 image's rag package, which this image
    does not carry (see graph_worker's own requirements: no Docling, no
    rag/, only graph_rag/). Small enough to duplicate directly rather than
    share a dependency across two images that otherwise have almost nothing
    in common.
    """
    if not audit_table:
        return
    import boto3
    table = boto3.resource("dynamodb").Table(audit_table)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    table.put_item(Item={
        "pk": f"DOC#{doc_id}", "sk": f"GRAPH#{run_id}",
        "status": status, "duration_s": str(elapsed),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: str(v) for k, v in fields.items()},
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the graph for one document")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--doc-id", required=True,
                    help="the doc_id whose chunks.json to read, e.g. "
                         "nct03164772-heart-failure")
    ap.add_argument("--audit-table", default=os.getenv("AUDIT_TABLE"))
    args = ap.parse_args()

    from graph_rag import config as gconfig
    from graph_rag import store

    try:
        driver = store.driver()
    except Exception as exc:
        # A connection failure — bad NEO4J_URI (including deploy.py's own
        # "replace-me" placeholder if never updated), a wrong password, an
        # Aura instance that is paused — happens before build_one ever
        # runs. Without this, it crashes unhandled with NO audit record at
        # all: the exception propagates straight out of main(), Batch sees
        # a non-zero exit and retries correctly, but scripts/status.py
        # would show "not yet run" for a document that genuinely tried and
        # failed — exactly the situation where the audit trail matters
        # most, since CloudWatch logs are the only place the real reason
        # would otherwise be visible.
        print(f"FAILED to connect to Neo4j: {exc}", file=sys.stderr, flush=True)
        _audit(args.audit_table, args.doc_id, "FAILED", 0.0, error=str(exc)[:400])
        return 1

    session = driver.session(database=gconfig.NEO4J_DATABASE)
    try:
        result = build_one(args.bucket, args.doc_id, session, args.audit_table)
    finally:
        session.close()
        driver.close()

    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
