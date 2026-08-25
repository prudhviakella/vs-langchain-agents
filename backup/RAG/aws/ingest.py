"""Ingest PDFs into the vector index.

The entrypoint. It runs the five stages in order and does nothing else — the work
lives in the `rag` package, one module per stage.

    python ingest.py --pdf report.pdf                  one local file
    python ingest.py --dir pdfs/                       a folder
    python ingest.py --dir pdfs/ --skip-done           resume an interrupted run
    python ingest.py --bucket my-bucket --key raw/small/report.pdf

    parse     PDF  -> DoclingDocument     rag.docling_io
    inspect   verify, and write reports   rag.inspect
    figures   store rendered images       rag.docling_io
    chunk     records with metadata       rag.chunking
    index     three-way diff to Pinecone  rag.sync

Identical on a laptop and inside a Fargate task; the differences are where the PDF
comes from and whether audit records are written to DynamoDB.

There is deliberately no distributed code here: no queue polling, no locking, no
heartbeats. On AWS, parallelism lives entirely in the state machine, which runs many
copies of this script at once. Locally, --dir just loops.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rag.audit import Audit, Stage
from rag.chunking import build_records, document_date
from rag.config import slugify
from rag.docling_io import parse_pdf, save_figures
from rag.index import open_index, scan_document, write_manifest
from rag.inspect import inspect, write_chunk_report, write_extraction_report
from rag.sync import sync


def ingest_one(pdf: Path, index, bucket: str | None = None,
               audit_table: str | None = None) -> dict:
    """Run all five stages against one PDF and return what happened.

    Returns a result dict rather than raising, so a batch can record a failure and
    carry on. A corpus always contains one file that is encrypted, corrupt, or a
    scan of a fax, and losing the other nineteen to it is not a useful default.

    The index is passed in rather than opened here: opening it probes the embedding
    dimension with an API call, and doing that once per document in a batch is
    twenty pointless round trips.
    """
    doc_id = slugify(pdf.stem)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit = Audit(audit_table, doc_id, run_id)

    print(f"\n{doc_id}  run {run_id}", flush=True)
    started = time.time()
    audit.run("STARTED", source=pdf.name)

    try:
        with Stage(audit, "parse"):
            doc = parse_pdf(pdf)
            # Rendering the whole document to markdown is expensive on long PDFs.
            # Do it once here and reuse it for the checks and the date.
            markdown = doc.export_to_markdown()

        with Stage(audit, "inspect"):
            report = inspect(doc, markdown)
            # Written before anything is chunked or embedded, so a bad parse can be
            # caught by reading rather than by wondering why answers are wrong.
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
        # Numbers written as strings because DynamoDB's Python resource layer
        # rejects floats, and this keeps the audit records uniformly typed.
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
        # One broad handler: the Stage context manager has already recorded which
        # stage failed and why, so this only closes out the run record.
        elapsed = round(time.time() - started, 1)
        audit.run("FAILED", duration_s=str(elapsed), error=str(exc)[:400])
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return {"doc_id": doc_id, "status": "failed", "seconds": elapsed,
                "error": str(exc)[:200]}


def already_indexed(index, pdf: Path) -> bool:
    """Whether this document already has chunks in the index.

    The index is the real state, so this is what `--skip-done` asks. The manifest
    would be cheaper to read but records what a previous run *intended*; if it was
    killed between the sync and the manifest write, the manifest is wrong and the
    index is not.
    """
    return bool(scan_document(index, slugify(pdf.stem)))


def summarise(results: list[dict]) -> None:
    """Print one line per document, then the totals.

    The per-document columns are the ones worth scanning a batch for: a document
    with suspect tables or undescribed figures parsed, but not well, and that is
    invisible in a pass/fail count.
    """
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

    # Worth surfacing separately: these documents succeeded but produced content the
    # index cannot use well, which a pass/fail count hides entirely.
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

    Returns 0 or 1 rather than raising, because the exit code is what the ECS task
    reports and what the state machine reads to decide success or failure.
    """
    ap = argparse.ArgumentParser(description="Ingest PDFs into the vector index")
    ap.add_argument("--pdf", help="one local PDF")
    ap.add_argument("--dir", help="a folder of PDFs, processed in name order")
    ap.add_argument("--bucket", help="S3 bucket holding the PDF, or for figure images")
    ap.add_argument("--key", help="S3 key of the PDF")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip documents already present in the index")
    ap.add_argument("--limit", type=int,
                    help="stop after this many documents, for a trial run")
    ap.add_argument("--audit-table", default=os.getenv("AUDIT_TABLE"))
    args = ap.parse_args()

    # Three input modes, one code path afterwards.
    if args.dir:
        pdfs = sorted(Path(args.dir).glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"no PDFs found in {args.dir}")
    elif args.bucket and args.key:
        import boto3
        local = Path("/tmp") / Path(args.key).name
        boto3.client("s3").download_file(args.bucket, args.key, str(local))
        pdfs = [local]
    elif args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        ap.error("provide --pdf, --dir, or --bucket and --key")

    if args.limit:
        pdfs = pdfs[:args.limit]

    # Opened once. Doing it per document would probe the embedding dimension with an
    # API call twenty times over.
    index = open_index(create=True)

    print(f"{len(pdfs)} document(s) to ingest", flush=True)
    results = []
    for n, pdf in enumerate(pdfs, 1):
        if args.skip_done and already_indexed(index, pdf):
            print(f"\n[{n}/{len(pdfs)}] {pdf.name}  already indexed, skipping", flush=True)
            results.append({"doc_id": slugify(pdf.stem), "status": "skipped"})
            continue
        print(f"\n[{n}/{len(pdfs)}] {pdf.name}", flush=True)
        results.append(ingest_one(pdf, index, args.bucket, args.audit_table))

    if len(pdfs) > 1:
        summarise(results)

    # Non-zero if anything failed, so this can gate a script.
    return 1 if any(r["status"] == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
