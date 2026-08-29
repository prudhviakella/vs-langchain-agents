"""Ingest every PDF in a folder.

    pdfs/*.pdf
        |
        v
    for each, in size order (smallest first)
        |
        +-- already done and unchanged?  ->  skip, no parse, no cost
        |
        +-- parse -> reports -> records -> sync -> index
        |
        +-- append one row to the run log
        |
        v
    reports/_run.json          one row per document, machine-readable
    reports/_run.md            the same as a table you can read

    python ingest_all.py                     everything in pdfs/
    python ingest_all.py --dry-run           what would run, and what would be skipped
    python ingest_all.py --only NCT03164772  one document, matched on filename
    python ingest_all.py --force             re-ingest even if already indexed

WHAT THIS DOES NOT DO

    It does NOT retry. A document that fails is recorded and the run continues;
    fix the cause and run again, and the ones that succeeded are skipped.

    It does NOT parallelise. Every enrichment is a model pass on CPU and they
    already saturate the machine; running two documents at once makes both
    slower and the memory profile worse.

    It does NOT clear any cache. Figure descriptions are keyed on the rendered
    image bytes, so a re-run of an unchanged document costs no vision calls.

WHY SMALLEST FIRST

    A 15-page document that fails takes two minutes to fail. A 250-page one
    takes twenty. Ordering by size means a systemic problem — a missing key, a
    changed API, a bad setting — surfaces in the first two minutes rather than
    the twentieth.

WHY ONE DOCUMENT AT A TIME, COMMITTED AS IT GOES

    2,785 pages is hours of CPU. A crash on document 18 must not cost the
    first 17. Each document is parsed, reported, embedded and synced to
    completion before the next one starts, so the index and the manifest are
    always consistent with what has actually finished.
"""

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from rag import chunking, config, docling_io
from rag import index as index_module
from rag import inspect as inspection
from rag import sync as sync_module

# Set to an S3 bucket to store figure images. None keeps everything local.
BUCKET = None

RUN_JSON = config.REPORT_DIR / "_run.json"
RUN_MD = config.REPORT_DIR / "_run.md"


def load_log() -> dict:
    """Previous runs, keyed by doc_id. Empty on the first run."""
    if RUN_JSON.exists():
        try:
            return {row["doc_id"]: row for row in json.loads(RUN_JSON.read_text())}
        except Exception:
            # A corrupt log must not stop an ingest. Worst case it re-does work
            # that sync() will then find unchanged and skip anyway.
            return {}
    return {}


def already_done(log: dict, doc_id: str, pdf: Path) -> bool:
    """True when this exact file has been ingested successfully.

        same doc_id + same size + same mtime  ->  skip

    Size and mtime rather than a content hash: hashing a 32 MB PDF on every
    run to save a lookup is the wrong trade, and an edited protocol changes
    both. If you replace a file with one of identical size and timestamp, use
    --force.
    """
    row = log.get(doc_id)
    if not row or row.get("status") != "ok":
        return False
    stat = pdf.stat()
    return (row.get("bytes") == stat.st_size
            and abs(row.get("mtime", 0) - stat.st_mtime) < 1)


def ingest_one(pdf: Path, index) -> dict:
    """Parse, report, chunk and index one PDF.

        1  parse            six models; the only slow step
        2  inspect          counts, layout quality, problems
        3  extract report   written before anything is embedded
        4  records          chunking, dropping, merging
        5  chunk report     every chunk exactly as it will be stored
        6  sync             embed and upsert only what changed

    Returns one log row. Raises nothing — the caller records failures.
    """
    doc_id = config.slugify(pdf.stem)
    started = time.time()

    # STEP 1 — parse. describe_figures() runs inside this, against its cache.
    doc = docling_io.parse_pdf(pdf)
    parsed_at = time.time()

    # STEP 2/3 — what did the parse produce, and what went wrong?
    markdown = doc.export_to_markdown()
    report = inspection.inspect(doc, markdown)
    inspection.write_extraction_report(doc, doc_id, pdf)

    # STEP 4 — records. Still nothing embedded.
    doc_date = chunking.document_date(pdf, markdown[:4000])
    figure_uris = docling_io.save_figures(doc, doc_id, BUCKET)
    records = chunking.build_records(doc, pdf, doc_id, doc_date, figure_uris)

    # STEP 5 — the chunk report, readable next to the PDF.
    inspection.write_chunk_report(records, doc_id)

    # STEP 6 — the only step that spends money on embeddings.
    plan = sync_module.sync(index, doc_id, records)

    types = {}
    for record in records:
        kind = record["meta"]["content_type"]
        types[kind] = types.get(kind, 0) + 1

    stat = pdf.stat()
    return {
        "doc_id": doc_id,
        "file": pdf.name,
        "status": "ok",
        "pages": len(doc.pages),
        "bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "doc_date": doc_date,
        "records": len(records),
        "types": types,
        "problems": len(report.get("problems", [])) if isinstance(report, dict) else None,
        "parse_seconds": round(parsed_at - started),
        "total_seconds": round(time.time() - started),
        "added": plan.get("added"),
        "removed": plan.get("removed"),
        "unchanged": plan.get("unchanged"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_log(rows: list[dict]) -> None:
    """Write the run log, JSON and Markdown, after every document.

    After EVERY document, not at the end. A run that dies on document 18
    should still leave a readable record of the first 17.
    """
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_JSON.write_text(json.dumps(rows, indent=2))

    header = (f"{'document':<44}{'pages':>6}{'recs':>6}{'text':>6}{'tab':>5}"
              f"{'fig':>5}{'prob':>6}{'parse':>7}{'add':>6}{'status':>8}")
    lines = ["# corpus ingest", "",
             f"- run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"- documents: {len(rows)}",
             f"- ok: {sum(1 for r in rows if r['status'] == 'ok')}",
             f"- failed: {sum(1 for r in rows if r['status'] != 'ok')}",
             "", "```", header, "-" * len(header)]
    for r in sorted(rows, key=lambda r: r.get("file", "")):
        t = r.get("types") or {}
        lines.append(
            f"{r.get('file', '')[:43]:<44}{r.get('pages') or '':>6}"
            f"{r.get('records') or '':>6}{t.get('text', '') or '':>6}"
            f"{t.get('table', '') or '':>5}{t.get('figure', '') or '':>5}"
            f"{r.get('problems') if r.get('problems') is not None else '':>6}"
            f"{(str(r.get('parse_seconds')) + 's') if r.get('parse_seconds') else '':>7}"
            f"{r.get('added') if r.get('added') is not None else '':>6}"
            f"{r.get('status', ''):>8}")
    lines += ["```", ""]

    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        lines += ["## failures", ""]
        for r in failed:
            lines.append(f"- **{r.get('file')}** — {r.get('error', '')[:300]}")
        lines.append("")
    RUN_MD.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", nargs="?", default="pdfs",
                        help="folder of PDFs (default: pdfs)")
    parser.add_argument("--only", default=None,
                        help="substring of the filename; ingest just that one")
    parser.add_argument("--force", action="store_true",
                        help="re-ingest documents already recorded as ok")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would run, parse nothing")
    args = parser.parse_args()

    folder = Path(args.folder)
    pdfs = sorted(folder.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if args.only:
        pdfs = [p for p in pdfs if args.only.lower() in p.name.lower()]
    if not pdfs:
        print(f"no PDFs matched in {folder.resolve()}")
        return

    log = load_log()
    queue, skipped = [], []
    for pdf in pdfs:
        doc_id = config.slugify(pdf.stem)
        if not args.force and already_done(log, doc_id, pdf):
            skipped.append(pdf)
        else:
            queue.append(pdf)

    total_mb = sum(p.stat().st_size for p in queue) / 1e6
    print(f"{len(pdfs)} PDF(s) in {folder.resolve()}")
    print(f"  to ingest : {len(queue)}  ({total_mb:.0f} MB)")
    print(f"  skipped   : {len(skipped)} already ingested and unchanged")
    for pdf in skipped:
        print(f"      - {pdf.name}")
    print()

    if args.dry_run:
        for i, pdf in enumerate(queue, 1):
            print(f"  {i:2}. {pdf.name}  ({pdf.stat().st_size / 1e6:.1f} MB)")
        return

    # Opened once. open_index() probes the embedding dimension at connect time,
    # and doing that per document would be 20 wasted API calls.
    index = index_module.open_index(create=True)

    rows = list(log.values())
    started = time.time()

    for i, pdf in enumerate(queue, 1):
        print(f"\n{'=' * 72}")
        print(f"[{i}/{len(queue)}] {pdf.name}  ({pdf.stat().st_size / 1e6:.1f} MB)")
        print("=" * 72, flush=True)
        try:
            row = ingest_one(pdf, index)
            print(f"  done in {row['total_seconds']}s "
                  f"({row['records']} records, {row['added']} embedded)", flush=True)
        except Exception as exc:
            # One bad document must not end the run. The traceback goes to the
            # console, a one-line reason goes in the log, and the next document
            # starts.
            traceback.print_exc()
            row = {"doc_id": config.slugify(pdf.stem), "file": pdf.name,
                   "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "bytes": pdf.stat().st_size, "mtime": pdf.stat().st_mtime,
                   "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

        rows = [r for r in rows if r.get("doc_id") != row["doc_id"]] + [row]
        write_log(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\n{'=' * 72}")
    print(f"{len(ok)} ok, {len(bad)} failed, "
          f"{round(time.time() - started)}s this run")
    print(f"  pages   : {sum(r.get('pages') or 0 for r in ok)}")
    print(f"  records : {sum(r.get('records') or 0 for r in ok)}")
    print(f"  log     : {RUN_MD}")
    for r in bad:
        print(f"  FAILED  : {r['file']} — {r.get('error', '')[:120]}")


if __name__ == "__main__":
    main()
