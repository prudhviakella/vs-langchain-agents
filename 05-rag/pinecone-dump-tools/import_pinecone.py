#!/usr/bin/env python3
"""Load a JSON dump (from export_pinecone.py) into a Pinecone index.

    PINECONE_API_KEY=... python import_pinecone.py \
        --file dump.json --index my-index

FLOW

    read the json file  ->  header (dimension, metric) + vectors    (STEP 1)
        |
        v
    does --index already exist in THIS Pinecone account?
        |                                    |
        no                                   yes
        |                                    |
        v                                    v
    create it, matching the           check its dimension matches
    file's own dimension/metric       the file's — refuse if not    (STEP 2)
        |                                    |
        +------------------+-----------------+
                           |
                           v
                  upsert in batches                                  (STEP 3)

WHAT THIS DOES NOT DO

    - It does not merge with or overwrite unrelated existing data by
      deleting anything first. Upserting a vector ID this index already
      has replaces that one vector; everything else in the index is
      left alone.
    - It does not create a NEW Pinecone account or API key. Every
      student needs their own free Pinecone account and API key before
      running this — that part cannot be automated away.
    - It does not verify the vectors are correct for any particular use
      case, only that they load into an index shaped the way they were
      exported from.

WHY A DIMENSION MISMATCH REFUSES RATHER THAN WARNS

Pinecone's own behaviour is the reason: pushing vectors into an index of
the wrong dimension fails outright, which is at least honest. But
querying an EXISTING index that happens to already have the right
dimension for a different reason — say, a student re-used an index from
an earlier exercise — would succeed and return ranked results that mean
nothing, with no error anywhere. This script would rather stop and say
exactly why than let that happen silently.
"""

import argparse
import gzip
import json
import os

from dotenv import load_dotenv
import sys
import time

from pinecone import Pinecone, ServerlessSpec

# Same batch size export_pinecone.py lists and fetches at — safely inside
# Pinecone's documented upsert ceiling (1000 IDs / 2MB per call) without
# needing to reason about metadata size per vector.
BATCH_SIZE = 100


def load_dump(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        dump = json.load(f)

    for key in ("dimension", "metric", "vectors"):
        if key not in dump:
            raise SystemExit(
                f"{path} is missing {key!r} — this does not look like a "
                "file export_pinecone.py produced. Re-export it, or check "
                "you passed the right file.")
    return dump


def ensure_index(pc: Pinecone, index_name: str, dimension: int, metric: str,
                 cloud: str, region: str) -> None:
    """STEP 2 — create the target index if it does not exist yet, or
    verify an existing one actually matches the data about to be loaded
    into it.
    """
    if not pc.has_index(index_name):
        print(f"  index {index_name!r} does not exist — creating it "
             f"(dimension={dimension}, metric={metric!r}, "
             f"cloud={cloud}, region={region})")
        pc.create_index(
            name=index_name, dimension=dimension, metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Creation is asynchronous — never wait without a deadline, or a
        # failed creation spins here silently forever.
        deadline = time.time() + 180
        while not pc.describe_index(index_name).status.get("ready", False):
            if time.time() > deadline:
                raise SystemExit(f"{index_name} not ready after 180s — "
                                 "check the Pinecone console for its status.")
            time.sleep(2)
        print(f"  index {index_name!r} ready")
        return

    description = pc.describe_index(index_name)
    if description.dimension != dimension:
        raise SystemExit(
            f"Index {index_name!r} already exists with dimension "
            f"{description.dimension}, but this file's vectors are "
            f"dimension {dimension}. Loading them anyway would either be "
            "rejected outright or, worse, silently produce meaningless "
            "search results. Use a different --index name, or delete the "
            "existing index first if you're sure it should be replaced.")
    if description.metric != metric:
        print(f"  WARNING: index {index_name!r} uses metric "
             f"{description.metric!r}, but this file's vectors were "
             f"computed for {metric!r}. Loading will succeed, but search "
             "scores may not mean what you expect.", file=sys.stderr)
    print(f"  index {index_name!r} already exists — dimension matches, "
         "loading into it")


def main() -> None:
    # Loaded first, before anything else — same convention as this
    # project's own notebooks (see 01_build_graph.ipynb's setup cell).
    # A .env file in the current directory is picked up automatically;
    # if one doesn't exist, this is a harmless no-op and env vars already
    # set in the shell (the NEO4J_URI=... prefix style) still work exactly
    # as before.
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", required=True,
                    help="dump file from export_pinecone.py (.json or .json.gz)")
    ap.add_argument("--index", required=True,
                    help="Pinecone index name to load into — created if it "
                         "does not exist yet")
    ap.add_argument("--namespace", default=None,
                    help="namespace to load into (default: the same "
                         "namespace the file was exported from)")
    ap.add_argument("--cloud", default="aws",
                    help="cloud to create the index in, if it doesn't exist yet (default: aws)")
    ap.add_argument("--region", default="us-east-1",
                    help="region to create the index in, if it doesn't exist yet (default: us-east-1)")
    args = ap.parse_args()

    if "PINECONE_API_KEY" not in os.environ:
        raise SystemExit(
            "PINECONE_API_KEY is not set. Get your own key from the "
            "Pinecone console (API Keys section) and run:\n"
            "  PINECONE_API_KEY=your-key python import_pinecone.py ...")

    # STEP 1 — read the file.
    print(f"reading {args.file}")
    dump = load_dump(args.file)
    namespace = args.namespace if args.namespace is not None else dump["namespace"]
    # See export_pinecone.py's own comment on this exact line — confirmed
    # directly that Pinecone's fetch() mishandles the literal string
    # "__default__" even though list() accepts it, so it is normalized to
    # the real empty-string namespace value everywhere in these two
    # scripts rather than just where the bug was first found.
    if namespace == "__default__":
        namespace = ""
    print(f"  {dump['vector_count']} vector(s), dimension={dump['dimension']}, "
         f"metric={dump['metric']!r}, exported {dump.get('exported_at', 'unknown')}")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    ensure_index(pc, args.index, dump["dimension"], dump["metric"],
                args.cloud, args.region)

    # STEP 3 — upsert in batches.
    index = pc.Index(args.index)
    vectors = dump["vectors"]
    loaded = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        loaded += len(batch)
        print(f"  loaded {loaded} of {len(vectors)} vector(s)...",
             end="\r", flush=True)
    print(f"  loaded {loaded} of {len(vectors)} vector(s)" + " " * 10)

    print(f"\ndone — {loaded} vector(s) now in index {args.index!r}, "
         f"namespace {namespace!r}")


if __name__ == "__main__":
    main()
