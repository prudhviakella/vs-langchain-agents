#!/usr/bin/env python3
"""Check one document's progress through both stages.

    python scripts/status.py nct03164772-heart-failure

Reads the audit table directly rather than checking Pinecone or Neo4j —
this is the one place both stages agree to write their state, specifically
so a single command can answer "what happened to this document" without
needing three different services' credentials to find out.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import boto3

from infra import config

ddb = boto3.resource("dynamodb", region_name=config.REGION).Table(config.AUDIT_TABLE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("doc_id")
    args = ap.parse_args()

    response = ddb.query(
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": f"DOC#{args.doc_id}"},
    )
    items = response.get("Items", [])
    if not items:
        print(f"no record for {args.doc_id!r} — check the doc_id, or it has "
              "not been uploaded yet")
        return

    runs = sorted((i for i in items
                  if i["sk"].startswith("RUN#") and "#STAGE#" not in i["sk"]),
                 key=lambda i: i["sk"])
    stages = sorted((i for i in items if "#STAGE#" in i["sk"]),
                    key=lambda i: i["sk"])
    graph_runs = sorted((i for i in items if i["sk"].startswith("GRAPH#")),
                        key=lambda i: i["sk"])

    print(f"{args.doc_id}\n{'=' * len(args.doc_id)}")

    if runs:
        print("\nStage 1 (parse and index):")
        for run in runs:
            print(f"  {run['sk']:<24} {run.get('status', '?'):<12}"
                  f"tier={run.get('tier', '?'):<8}"
                  f"pages={run.get('pages', '-'):<6}"
                  f"chunks={run.get('chunks', '-')}")
            if run.get("error"):
                print(f"      error: {run['error']}")

    if stages:
        print("\n  stage detail (most recent run):")
        last_run_sk = runs[-1]["sk"] if runs else None
        for stage in stages:
            if last_run_sk and last_run_sk not in stage["sk"]:
                continue
            name = stage["sk"].split("#STAGE#")[-1]
            duration = stage.get("duration_s", "-")
            print(f"    {name:<12} {stage.get('status', '?'):<10} {duration}s")

    if graph_runs:
        print("\nStage 2 (graph build):")
        for run in graph_runs:
            print(f"  {run['sk']:<24} {run.get('status', '?'):<12}"
                  f"documents={run.get('documents', '-'):<4}"
                  f"linked_to_trial={run.get('linked_to_trial', '-')}")
            if run.get("error"):
                print(f"      error: {run['error']}")
    else:
        print("\nStage 2 (graph build): not yet run")


if __name__ == "__main__":
    main()
