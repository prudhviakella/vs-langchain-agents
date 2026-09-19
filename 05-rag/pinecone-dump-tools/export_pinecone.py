#!/usr/bin/env python3
"""Dump every vector in a Pinecone index to a single JSON file.

    PINECONE_API_KEY=... python export_pinecone.py \
        --index rag-docs --out dump.json

FLOW

    connect to Pinecone
        |
        v
    describe_index()  ->  dimension, metric
        |
        v
    index.list(namespace)  ->  pages of up to 100 IDs        (STEP 1)
        |
        v
    index.fetch(ids=page)  ->  values + metadata, per page   (STEP 2)
        |
        v
    write ONE json file: {header, vectors: [...]}            (STEP 3)

WHAT THIS DOES NOT DO

    - It does not touch OpenAI, Docling, or anything from the main
      ingestion pipeline. This only ever talks to Pinecone.
    - It does not export more than one namespace per run. Run it again
      with --namespace for a second one if the index uses more than one.
    - It does not compress the output itself. Pass --gzip if the file
      is too big to share as-is; nothing downstream needs it uncompressed.

WHY THE HEADER CARRIES DIMENSION AND METRIC

The whole reason this pair of scripts exists: not every student can set
up the full AWS pipeline, so one working index gets exported once and
every student imports the same data into their own account. A student's
import should not need to know or guess the source index's dimension or
metric — get either wrong when creating their own index and Pinecone
either rejects it outright (dimension) or silently returns ranked
results that mean nothing (metric only affects distance math, not
validity). Both are written into the file's own header so
import_pinecone.py never has to ask.
"""

import argparse
import gzip
import json
import os

from dotenv import load_dotenv
from datetime import datetime, timezone

from pinecone import Pinecone

# Matches Pinecone's own default list() page size and the documented
# fetch() batch ceiling (1000 IDs / 2MB per call) with real headroom —
# not chosen to be clever, chosen to be safely inside both documented
# limits without needing to reason about metadata size per vector.
PAGE_SIZE = 100


def export_index(index_name: str, namespace: str) -> dict:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if not pc.has_index(index_name):
        raise SystemExit(
            f"Index {index_name!r} does not exist in this Pinecone account. "
            "Check the name and that PINECONE_API_KEY points at the right "
            "account.")

    description = pc.describe_index(index_name)
    index = pc.Index(index_name)

    # STEP 1 — list every ID in the namespace, page by page.
    #
    # index.list() is a generator: the Python SDK paginates automatically,
    # yielding a list of IDs per page rather than one ID at a time. It
    # never returns values or metadata — only IDs — so fetch() further
    # down is what does the real work.
    all_ids: list[str] = []
    for page in index.list(namespace=namespace, limit=PAGE_SIZE):
        # Confirmed directly from real diagnostic output against a live
        # index — each page is NOT a plain list of ID strings. It yields
        # ListItem objects, each with the real ID on its own `.id`
        # attribute:
        #
        #   ListResponse(vectors=[ListItem(id='doc:abc:0'), ...], ...)
        #
        # extend(page) without unwrapping .id silently "worked" in the
        # sense that the running count came out correct (5764, matching
        # describe_index_stats() exactly) — nothing raised, since
        # iterating a page and appending its items is valid Python
        # regardless of what those items are. The bug only surfaced one
        # step later: fetch() was then called with a list of ListItem
        # OBJECTS instead of ID strings, matched nothing, and returned
        # cleanly empty — no error, just zero vectors, for every batch.
        all_ids.extend(item.id for item in page)
        print(f"  listed {len(all_ids)} id(s) so far...", end="\r", flush=True)
    print(f"  listed {len(all_ids)} id(s) total" + " " * 10)

    if not all_ids:
        print(f"  namespace {namespace!r} is empty — nothing to export")

    # STEP 2 — fetch values + metadata for every ID, one page at a time.
    #
    # fetch() takes a list of IDs and returns a dict keyed by ID. Batched
    # at PAGE_SIZE rather than fetching one ID at a time, and rather than
    # trying to fetch all_ids in one call — the latter risks the 2MB
    # per-request ceiling once vector count gets into the thousands.
    vectors = []
    for i in range(0, len(all_ids), PAGE_SIZE):
        batch_ids = all_ids[i:i + PAGE_SIZE]
        result = index.fetch(ids=batch_ids, namespace=namespace)
        for vector_id, record in result.vectors.items():
            vectors.append({
                "id": vector_id,
                "values": list(record.values),
                "metadata": dict(record.metadata) if record.metadata else {},
            })
        print(f"  fetched {len(vectors)} of {len(all_ids)} vector(s)...",
             end="\r", flush=True)
    if all_ids:
        print(f"  fetched {len(vectors)} of {len(all_ids)} vector(s)" + " " * 10)

    return {
        "index_name": index_name,
        "namespace": namespace,
        "dimension": description.dimension,
        "metric": description.metric,
        "vector_count": len(vectors),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vectors": vectors,
    }


def main() -> None:
    # Loaded first, before anything else — same convention as this
    # project's own notebooks (see 01_build_graph.ipynb's setup cell).
    # A .env file in the current directory is picked up automatically;
    # if one doesn't exist, this is a harmless no-op and env vars already
    # set in the shell (the NEO4J_URI=... prefix style) still work exactly
    # as before.
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", required=True, help="Pinecone index name to export from")
    ap.add_argument("--namespace", default="",
                    help="namespace to export (default: the unnamed default namespace)")
    ap.add_argument("--out", required=True, help="output file path, e.g. dump.json")
    ap.add_argument("--gzip", action="store_true",
                    help="gzip-compress the output — .gz is added to --out "
                         "automatically if not already there")
    args = ap.parse_args()

    if args.gzip and not args.out.endswith(".gz"):
        args.out += ".gz"

    if "PINECONE_API_KEY" not in os.environ:
        raise SystemExit(
            "PINECONE_API_KEY is not set. Get it from the Pinecone console "
            "(API Keys section) and run:\n"
            "  PINECONE_API_KEY=your-key python export_pinecone.py ...")

    # "__default__" is what Pinecone's own console displays as the
    # namespace name, and typing it into --namespace is the natural thing
    # to do after seeing it there — confirmed directly as a real failure,
    # not a hypothetical one: index.list(namespace="__default__") found
    # every ID correctly, but index.fetch() on those exact same IDs with
    # the identical namespace string returned nothing, for every batch.
    # Pinecone's own GitHub (pinecone-io/cli#88) explains why: "__default__"
    # is a human-readable DISPLAY alias, not the real wire value — the
    # actual default namespace is an empty string, and converting the
    # literal word back to that only happens in newer API versions,
    # inconsistently across different calls in others. Normalized here so
    # this never has to be explained or hit twice.
    if args.namespace == "__default__":
        args.namespace = ""

    print(f"exporting index {args.index!r}, namespace {args.namespace!r}")
    dump = export_index(args.index, args.namespace)

    # STEP 3 — write the file.
    payload = json.dumps(dump, indent=2).encode("utf-8")
    if args.gzip:
        payload = gzip.compress(payload)
    with open(args.out, "wb") as f:
        f.write(payload)

    size_mb = len(payload) / 1024 / 1024
    print(f"\nwrote {dump['vector_count']} vector(s), "
         f"dimension={dump['dimension']}, metric={dump['metric']!r}, "
         f"to {args.out} ({size_mb:.1f} MB{' gzipped' if args.gzip else ''})")


if __name__ == "__main__":
    main()
