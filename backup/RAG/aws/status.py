"""Report what the pipeline did, from the DynamoDB audit trail.

    python aws/status.py                  one line per document
    python aws/status.py --doc nct04368728  every stage of one document
    python aws/status.py --failed         failures only

The state machine's own history shows which tasks ran. This shows what happened
inside them — how many chunks each document produced, how long each stage took, and
where a failure occurred.
"""

import argparse

import boto3
from boto3.dynamodb.conditions import Key

import config

table = boto3.resource("dynamodb", region_name=config.REGION).Table(config.AUDIT_TABLE)


def by_status(status: str) -> list[dict]:
    """Every audit record with a given status, oldest first.

    Uses the GSI rather than scanning. On a corpus of any size a scan reads every
    record to find the handful that failed, which is both slow and billed per item.
    """
    return table.query(IndexName="status-ts-index",
                       KeyConditionExpression=Key("status").eq(status))["Items"]


def document_history(doc_id: str) -> list[dict]:
    """Every record for one document: each run, and each stage within each run.

    The partition key is the document, so this is a single query returning the
    complete history — which is why the table is keyed that way rather than by run.
    """
    return table.query(
        KeyConditionExpression=Key("pk").eq(f"DOC#{doc_id}"))["Items"]


def main() -> None:
    """Print a per-document summary, or the full stage history of one document."""
    ap = argparse.ArgumentParser(description="Query the ingestion audit trail")
    ap.add_argument("--doc", help="show every stage for one document")
    ap.add_argument("--failed", action="store_true", help="show failures only")
    args = ap.parse_args()

    if args.doc:
        # Sorted by sort key, which puts the run record before its stages and the
        # stages in the order they were written.
        for item in sorted(document_history(args.doc), key=lambda i: i["sk"]):
            stage = item.get("stage", "run")
            detail = item.get("error") or f"{item.get('duration_s', '')}s"
            print(f"{item['ts']}  {stage:<10} {item.get('status',''):<10} {detail}")
        return

    # STARTED is included deliberately. A run left in that state is a container that
    # died without writing a terminal record — invisible if you only look for
    # failures, and the most common thing to go wrong on a long parse.
    statuses = ["FAILED"] if args.failed else ["COMPLETED", "FAILED", "STARTED"]
    # STAGE# records are filtered out here: this view is one line per run, and the
    # stage detail is what --doc is for.
    rows = [i for s in statuses for i in by_status(s) if "STAGE#" not in i["sk"]]
    if not rows:
        print("no runs recorded")
        return

    print(f"{'document':<34}{'status':<11}{'pages':>6}{'chunks':>8}{'added':>7}{'time':>8}")
    print("-" * 76)
    for item in sorted(rows, key=lambda i: i["ts"]):
        doc = item["pk"].removeprefix("DOC#")
        # 'added' below the chunk count on a re-run is the incremental sync working:
        # only changed chunks were embedded.
        print(f"{doc:<34}{item.get('status',''):<11}"
              f"{item.get('pages','-'):>6}{item.get('chunks','-'):>8}"
              f"{item.get('added','-'):>7}{item.get('duration_s','-'):>7}s")
        if item.get("error"):
            print(f"    {item['error'][:110]}")


if __name__ == "__main__":
    main()
