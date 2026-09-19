"""Ingest one PDF into the vector index. The Stage 1 Batch job container entrypoint.

    python ingest.py --bucket my-bucket --key raw/small/report.pdf   (Batch)
    python ingest.py --pdf report.pdf                                (local)
    python ingest.py --dir pdfs/ --skip-done                         (local batch)

    parse     PDF  -> DoclingDocument     rag.docling_io
    inspect   verify, and write reports   rag.inspect
    figures   store rendered images       rag.docling_io
    chunk     records with metadata       rag.chunking
    index     three-way diff to Pinecone  rag.sync

Identical on a laptop and inside a Batch task; the differences are where the
PDF comes from, whether the cache round-trips through S3, and whether audit
records are written to DynamoDB.

There is deliberately no distributed code here: no queue polling, no
locking, no heartbeats. On AWS, parallelism lives entirely in AWS Batch,
which the classifier Lambda submits one job to per document. Locally, --dir
just loops.
"""

# ─────────────────────────────────────────────────────────────────────────────
# MUST run before `from rag import ...` below, and it must be the FIRST thing
# this file does — not inside a function called later. rag.config reads
# every setting with os.getenv(...) at MODULE level, the instant it is
# imported, not lazily. Loading Parameter Store values after that import has
# already happened leaves every setting frozen at whatever was already in
# the environment, which for a container is nothing.
#
# This is the same class of bug found twice already in this project's
# history: once for load_dotenv() ordering locally, once for a Fargate task
# definition that supplied secrets but no tuning parameters at all. Both
# failures were silent — the pipeline ran, just with every default, and
# nothing said so. This bootstrap exists specifically so a cloud run cannot
# repeat it.
# ─────────────────────────────────────────────────────────────────────────────
import os


def _bootstrap_config_from_parameter_store() -> None:
    """Load every tuning parameter under PARAM_PREFIX into the process
    environment, before anything downstream reads it.

    A no-op when PARAM_PREFIX is unset, which is the local-development case
    — nothing here should require AWS credentials just to run against a
    local PDF with .env supplying settings the normal way.
    """
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
            # A job-definition ARN or queue name (JOBDEF_SMALL, QUEUE_LARGE)
            # lives under the same prefix for the classifier's convenience
            # but is not a rag.config setting — skip anything the worker
            # itself has no use for, so an unrelated future parameter under
            # this prefix cannot silently redefine something in rag.config.
            if key.startswith(("JOBDEF_", "QUEUE_")):
                continue
            os.environ[key] = param["Value"]
            loaded.append(key)
    print(f"loaded {len(loaded)} setting(s) from Parameter Store: "
          f"{', '.join(sorted(loaded))}", flush=True)


_bootstrap_config_from_parameter_store()

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rag.audit import Audit, Stage
from rag.chunking import build_records, document_date
from rag.config import CACHE_DIR, REPORT_DIR, slugify
from rag.docling_io import parse_pdf, save_figures
from rag.index import open_index, scan_document, write_manifest
from rag.inspect import inspect, write_chunk_report, write_extraction_report
from rag.sync import sync

# Defined here rather than imported from the top-level infra.config module,
# which is not copied into the container image — only rag/ and this file
# are. Duplicated constants, matching infra/config.py exactly, are worth it
# to keep the image from needing the whole provisioning package.
DOCS_PREFIX = "docs"
CACHE_PREFIX = "cache"


def sync_cache(bucket: str | None, direction: str) -> None:
    """Round-trip rag's local cache directory through S3.

    WHY THIS EXISTS

    A Spot-reclaimed task's local disk is gone with the instance. Without
    this, every retry re-describes every figure and re-embeds every chunk —
    real time and real API cost, repeated on every interruption.

    Worse than the cost: figure descriptions are measured to be
    non-deterministic (12 of 17 changed across two identical runs at
    temperature=0, earlier in this project), and chunk ids are content-
    addressed. An uncached description on retry produces a DIFFERENT chunk
    id for the same figure, which sync.py then reports as one chunk added
    and one removed — the index churns on every retry, not just the cost
    does.

    WHY `aws s3 sync` AND NOT A DIFFERENT MECHANISM

    It is the simplest thing that is actually correct: it diffs by size and
    mtime, transfers only what changed, and needs no bookkeeping of which
    hashes belong to which document — the cache is content-addressed and
    shared across every document by design.

    THE REAL LIMITATION, STATED PLAINLY

    This downloads (or uploads) the WHOLE cache directory on every job. At
    a few thousand documents' worth of cache entries that is a fast diff of
    mostly-unchanged small files. At a genuinely large corpus — the kind
    this project's own scaling discussion aims at — syncing millions of
    cache entries to every worker becomes its own bottleneck. The next
    evolution at that scale is a real lookup (DynamoDB holding hash ->
    S3 key) rather than a full local mirror; not built here because this
    project's measured corpus does not yet need it, and building it before
    it is needed is exactly the kind of unmeasured complexity this whole
    session has argued against elsewhere.
    """
    if not bucket:
        return
    src, dst = (f"s3://{bucket}/{CACHE_PREFIX}/", str(CACHE_DIR)) \
        if direction == "down" else (str(CACHE_DIR), f"s3://{bucket}/{CACHE_PREFIX}/")
    result = subprocess.run(
        ["aws", "s3", "sync", src, dst, "--quiet"],
        capture_output=True, text=True)
    if result.returncode != 0:
        # A cache sync failure should never abort the document — worst case
        # is a cache miss, which costs time and money, not correctness.
        print(f"  cache sync ({direction}) failed, continuing without it: "
              f"{result.stderr[:200]}", file=sys.stderr, flush=True)


def upload_reports(doc_id: str, bucket: str | None) -> int:
    """Copy this document's reports to S3, under its own prefix.

    A Batch task's filesystem is destroyed when the task exits. The
    extraction and chunk reports — the artifacts actually read to find out
    why a parse went wrong — are written to local disk by the rag package,
    so without this they exist only inside a container that is already
    gone. What survives otherwise is a DynamoDB record naming the stage
    that failed, CloudWatch logs, and nothing to read.

    Uploaded rather than written directly from rag.inspect, deliberately:
    the package stays free of cloud specifics, the local and AWS paths run
    the same lines, and a stage that fails part-way still leaves whatever
    it managed to write behind to be collected.

    Missing files are skipped rather than raising. This runs after a
    failure as well as after a success, and the reports it can find are
    worth more than consistency about the ones it cannot.
    """
    if not bucket:
        return 0
    import boto3

    s3 = boto3.client("s3")
    uploaded = 0
    # doc_id / f"{doc_id}.{name}" — matches write_extraction_report and
    # write_chunk_report's actual output path (out_dir = REPORT_DIR /
    # doc_id, via rag.inspect.report_dir), confirmed by reading those
    # functions directly. The previous version looked for
    # REPORT_DIR / f"{doc_id}.{name}" — one directory level too shallow —
    # so every local.exists() check silently failed, uploaded stayed 0,
    # and nothing was ever flagged as wrong: a run showed COMPLETED in
    # DynamoDB with a real chunk count while S3 held nothing at all.
    for name in ("extract.md", "extract.json", "chunks.md", "chunks.json"):
        local = REPORT_DIR / doc_id / f"{doc_id}.{name}"
        if not local.exists():
            continue
        content_type = "application/json" if name.endswith(".json") else "text/markdown"
        s3.put_object(Bucket=bucket, Key=f"{DOCS_PREFIX}/{doc_id}/{name}",
                      Body=local.read_bytes(), ContentType=content_type)
        uploaded += 1
    return uploaded


def upload_graph_chunks(doc_id: str, bucket: str | None, records: list[dict]) -> bool:
    """Write the full, unmodified chunk records to their own S3 key, for
    Stage 2 to read.

    NOT the same data as chunks.json, and this function exists specifically
    because that distinction was missed once already. rag.inspect.
    write_chunk_report()'s JSON output is deliberately metadata-only — it
    flattens each record's `meta` dict to the top level and explicitly
    drops the top-level `text` field, because the point of that file is a
    compact, diagnostic report; the full text already lives in the
    companion chunks.md.

    graph_worker/build_graph.py's flatten() was written assuming chunks.json
    carried the complete {"text": ..., "meta": {...}} shape. It does not,
    and never did — confirmed directly: the first real Stage 2 run against
    real data failed with `FAILED: 'meta'`, because the actual file has
    no top-level text and no nested meta wrapper at all, just meta's own
    fields flattened up with text removed. Patching flatten() alone could
    not have fixed this — the data it needs was never in that file to
    begin with. This function gives Stage 2 a genuinely separate artifact
    instead, so chunks.json keeps its original, deliberate diagnostic
    shape and Stage 2 gets what it actually needs.

    Written as records exactly as build_records() produced them — no
    reshaping — so flatten() on the reading side keeps working unchanged;
    only which file it reads had to change.
    """
    if not bucket:
        return False
    import boto3
    import json

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=f"{DOCS_PREFIX}/{doc_id}/graph_chunks.json",
        Body=json.dumps(records, default=str).encode("utf-8"),
        ContentType="application/json")
    return True


def ingest_one(pdf: Path, index, bucket: str | None = None,
               audit_table: str | None = None) -> dict:
    """Run all five stages against one PDF and return what happened.

    Returns a result dict rather than raising, so a batch can record a
    failure and carry on. A corpus always contains one file that is
    encrypted, corrupt, or a scan of a fax, and losing the other nineteen to
    it is not a useful default.

    The index is passed in rather than opened here: opening it probes the
    embedding dimension with an API call, and doing that once per document
    in a batch is twenty pointless round trips.
    """
    doc_id = slugify(pdf.stem)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit = Audit(audit_table, doc_id, run_id)

    print(f"\n{doc_id}  run {run_id}", flush=True)
    started = time.time()
    audit.run("STARTED", source=pdf.name, source_key=os.environ.get("_SOURCE_KEY", ""))

    try:
        with Stage(audit, "parse"):
            doc = parse_pdf(pdf)
            markdown = doc.export_to_markdown()

        with Stage(audit, "inspect"):
            report = inspect(doc, markdown)
            write_extraction_report(doc, doc_id, pdf)

        with Stage(audit, "figures"):
            figure_uris = save_figures(doc, doc_id, bucket)
            print(f"  {len(figure_uris)} figure images stored", flush=True)

        with Stage(audit, "chunk"):
            records = build_records(doc, pdf, doc_id,
                                    document_date(pdf, markdown[:4000]), figure_uris)
            write_chunk_report(records, doc_id)

        with Stage(audit, "index"):
            plan = sync(index, doc_id, records)

        elapsed = round(time.time() - started, 1)

        # Uploaded BEFORE the COMPLETED write below, deliberately — that
        # write is what the graph trigger Lambda watches for (see
        # lambda_graph_trigger), and Stage 2 reads chunks.json straight
        # from S3. Writing COMPLETED first would let the trigger fire,
        # and Stage 2 launch, before this upload has necessarily finished
        # — in practice Stage 2 needs a Spot instance to start first,
        # which almost always takes longer than uploading four small
        # files, but "almost always" is a race, not a guarantee, and this
        # closes it structurally instead of relying on that timing
        # holding.
        #
        # Wrapped in its own try/except: a document that genuinely parsed,
        # chunked, and indexed successfully should not be marked FAILED
        # just because the follow-up report upload hit a transient S3
        # problem. The trade-off is real and accepted — a failure here is
        # rare (this ran successfully seconds ago against the same
        # bucket) and logged loudly, rather than silently
        # double-guarded against.
        try:
            n = upload_reports(doc_id, bucket)
            if n:
                print(f"  {n} report(s) -> s3://{bucket}/{DOCS_PREFIX}/{doc_id}/",
                      flush=True)
        except Exception as exc:
            print(f"  report upload failed, continuing: {exc}",
                 file=sys.stderr, flush=True)

        try:
            # Same ordering reasoning as upload_reports above, but more
            # critical here: Stage 2 depends ENTIRELY on this file — it is
            # not a diagnostic extra the way the four reports are. Must
            # land in S3 before the COMPLETED write below, since that
            # write is what the graph trigger Lambda fires on.
            upload_graph_chunks(doc_id, bucket, records)
        except Exception as exc:
            print(f"  graph_chunks.json upload failed, continuing: {exc}",
                 file=sys.stderr, flush=True)

        audit.run("COMPLETED", duration_s=str(elapsed), pages=str(report["pages"]),
                  chunks=str(len(records)), added=str(plan["added"]),
                  removed=str(plan["removed"]))
        write_manifest(doc_id, len(records), extra={"source": pdf.name})

        print(f"done in {elapsed}s", flush=True)

        return {"doc_id": doc_id, "status": "ok", "seconds": elapsed,
                "pages": report["pages"], "chunks": len(records),
                "added": plan["added"], "removed": plan["removed"],
                "tables_suspect": report.get("tables_suspect", 0),
                "undescribed": report["pictures"] - report["pictures_described"]}

    except Exception as exc:
        elapsed = round(time.time() - started, 1)
        # source_key carried on the FAILED record too — this is what
        # lambda_error_handler needs to find the PDF in raw/ and move it to
        # error/. Losing it here would mean a failed document's audit trail
        # exists but nothing ever moves the file.
        audit.run("FAILED", duration_s=str(elapsed), error=str(exc)[:400],
                  source_key=os.environ.get("_SOURCE_KEY", ""))
        try:
            n = upload_reports(doc_id, bucket)
            if n:
                print(f"  {n} partial report(s) -> "
                      f"s3://{bucket}/{DOCS_PREFIX}/{doc_id}/", flush=True)
        except Exception as upload_exc:
            print(f"  could not upload reports: {upload_exc}", file=sys.stderr,
                  flush=True)
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return {"doc_id": doc_id, "status": "failed", "seconds": elapsed,
                "error": str(exc)[:200]}


def already_indexed(index, pdf: Path) -> bool:
    """Whether this document already has chunks in the index.

    The index is the real state, so this is what --skip-done asks. The
    manifest would be cheaper to read but records what a previous run
    INTENDED; if it was killed between the sync and the manifest write, the
    manifest is wrong and the index is not.
    """
    return bool(scan_document(index, slugify(pdf.stem)))


def summarise(results: list[dict]) -> None:
    """Print one line per document, then the totals."""
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print("\n" + "=" * 84)
    print(f"{'document':<34}{'status':<9}{'pages':>6}{'chunks':>7}"
          f"{'added':>7}{'suspect':>9}{'no desc':>8}{'time':>8}")
    print("-" * 84)
    for result in results:
        if result["status"] == "ok":
            print(f"{result['doc_id'][:33]:<34}{'ok':<9}{result['pages']:>6}"
                  f"{result['chunks']:>7}{result['added']:>7}"
                  f"{result['tables_suspect']:>9}{result['undescribed']:>8}"
                  f"{result['seconds']:>7.0f}s")
        elif result["status"] == "skipped":
            print(f"{result['doc_id'][:33]:<34}{'skipped':<9}")
        else:
            print(f"{result['doc_id'][:33]:<34}{'FAILED':<9}  {result['error'][:44]}")

    print("-" * 84)
    print(f"{len(ok)} ok, {len(failed)} failed, {len(skipped)} skipped   "
          f"{sum(r['pages'] for r in ok)} pages, "
          f"{sum(r['chunks'] for r in ok)} chunks, "
          f"{sum(r['seconds'] for r in ok) / 60:.0f} min")

    troubled = [r for r in ok if r["tables_suspect"] or r["undescribed"]]
    if troubled:
        print(f"\n{len(troubled)} document(s) parsed with problems — read their "
              "extraction reports:")
        for result in troubled:
            parts = []
            if result["tables_suspect"]:
                parts.append(f"{result['tables_suspect']} table(s) with bad structure")
            if result["undescribed"]:
                parts.append(f"{result['undescribed']} figure(s) with no description")
            print(f"  {result['doc_id']}: {', '.join(parts)}")

    if failed:
        print(f"\n{len(failed)} failed. Re-run with --skip-done to retry only these.")


def main() -> int:
    """Resolve the input, ingest, and report.

    Returns 0 or 1 rather than raising, because the exit code is what Batch
    reads to decide whether the job succeeded, failed, or should be retried.
    """
    ap = argparse.ArgumentParser(description="Ingest PDFs into the vector index")
    ap.add_argument("--pdf", help="one local PDF")
    ap.add_argument("--dir", help="a folder of PDFs, processed in name order")
    ap.add_argument("--bucket", help="S3 bucket holding the PDF, and for figures/cache/reports")
    ap.add_argument("--key", help="S3 key of the PDF")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip documents already present in the index")
    ap.add_argument("--limit", type=int,
                    help="stop after this many documents, for a trial run")
    ap.add_argument("--audit-table", default=os.getenv("AUDIT_TABLE"))
    args = ap.parse_args()

    if args.dir:
        pdfs = sorted(Path(args.dir).glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"no PDFs found in {args.dir}")
    elif args.bucket and args.key:
        import boto3
        local = Path("/tmp") / Path(args.key).name
        boto3.client("s3").download_file(args.bucket, args.key, str(local))
        pdfs = [local]
        # Threaded through to ingest_one via the environment rather than a
        # new parameter, so the FAILED audit record can carry the exact S3
        # key lambda_error_handler needs — the alternative was passing it
        # through three function signatures for one value.
        os.environ["_SOURCE_KEY"] = args.key
    elif args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        ap.error("provide --pdf, --dir, or --bucket and --key")

    if args.limit:
        pdfs = pdfs[:args.limit]

    sync_cache(args.bucket, "down")

    index = open_index(create=True)

    print(f"{len(pdfs)} document(s) to ingest", flush=True)
    results = []
    try:
        for n, pdf in enumerate(pdfs, 1):
            if args.skip_done and already_indexed(index, pdf):
                print(f"\n[{n}/{len(pdfs)}] {pdf.name}  already indexed, skipping",
                      flush=True)
                results.append({"doc_id": slugify(pdf.stem), "status": "skipped"})
                continue
            print(f"\n[{n}/{len(pdfs)}] {pdf.name}", flush=True)
            results.append(ingest_one(pdf, index, args.bucket, args.audit_table))
    finally:
        # In finally, not just on success: a mid-run crash should still
        # upload whatever figures and embeddings it computed before dying —
        # otherwise a task that gets 90% through a large document and then
        # OOMs pays for that 90% again on the retry.
        sync_cache(args.bucket, "up")

    if len(pdfs) > 1:
        summarise(results)

    return 1 if any(r["status"] == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
