"""Sort PDFs by page count into three size tiers and upload them to S3.

    python aws/sort_and_upload.py --dir ./pdfs --dry-run
    python aws/sort_and_upload.py --dir ./pdfs

Page count, not file size, predicts processing cost: a 33 MB scanned report can be
shorter than a 4 MB text one, and it is pages that drive both parse time and peak
memory. Each tier maps to a Fargate task definition with different CPU and memory,
so a 15-page document does not reserve a 16 GB container.
"""

import argparse
import json
from pathlib import Path

import boto3
from pypdf import PdfReader

import config

# Upper page bound per tier. The last has no bound, so anything larger falls into
# it. These are the boundaries the tier CPU and memory in config.py were sized for;
# moving one without moving the other gives you an under- or over-provisioned lane.
TIERS = [("small", 50), ("medium", 150), ("large", None)]


def page_count(pdf: Path) -> int:
    """Number of pages, read from the PDF structure without rendering anything.

    Malformed cross-reference tables are common in generated PDFs and produce
    warnings on almost every file, which would bury the useful output. They do not
    affect the page count, so they are suppressed.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return len(PdfReader(str(pdf)).pages)


def tier_for(pages: int) -> str:
    """Which lane a document belongs in."""
    for name, upper in TIERS:
        if upper is None or pages <= upper:
            return name
    return "large"


def main() -> None:
    """Sort a directory of PDFs into tiers, upload them, and write the manifest.

    Always run with --dry-run first. The printed distribution is what tells you
    whether the concurrency settings in config.py make sense for this corpus: if
    most documents land in one lane, that lane alone decides total wall clock and
    the other two are decoration.
    """
    ap = argparse.ArgumentParser(description="Sort PDFs into tiers and upload to S3")
    ap.add_argument("--dir", required=True, help="directory containing PDFs")
    ap.add_argument("--dry-run", action="store_true", help="sort and report, do not upload")
    args = ap.parse_args()

    pdfs = sorted(Path(args.dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found in {args.dir}")

    manifest = {name: [] for name, _ in TIERS}
    print(f"{'file':<44}{'pages':>7}  tier")
    print("-" * 62)

    for pdf in pdfs:
        pages = page_count(pdf)
        tier = tier_for(pages)
        key = f"{config.RAW_PREFIX}/{tier}/{pdf.name}"
        # The local path rides along so the upload loop below never has to re-derive
        # which file produced which entry — which would mean reading every PDF
        # header a second time.
        manifest[tier].append({"bucket": config.BUCKET, "key": key,
                               "pages": pages, "name": pdf.name,
                               "path": str(pdf)})
        print(f"{pdf.name:<44}{pages:>7}  {tier}")

    # This summary is the point of --dry-run. Compare the page totals against the
    # concurrency in config.py: lane time is roughly pages / concurrency, and the
    # slowest lane is your wall clock.
    print("-" * 62)
    for tier, items in manifest.items():
        pages = sum(i["pages"] for i in items)
        print(f"{tier:<10} {len(items):>3} documents  {pages:>6} pages")

    if args.dry_run:
        print("\ndry run, nothing uploaded")
        return

    s3 = boto3.client("s3", region_name=config.REGION)
    for items in manifest.values():
        for item in items:
            s3.upload_file(item["path"], config.BUCKET, item["key"])
            print(f"uploaded s3://{config.BUCKET}/{item['key']}")

    # The state machine reads bucket and key from each item. The local path is
    # dropped so the manifest describes only what exists in S3 — otherwise a run
    # started from another machine would carry paths that do not resolve.
    body = json.dumps({tier: [{k: v for k, v in i.items() if k != "path"} for i in items]
                       for tier, items in manifest.items()}, indent=2)
    s3.put_object(Bucket=config.BUCKET, Key=config.MANIFEST_KEY, Body=body.encode())
    Path("manifest.json").write_text(body)
    print(f"\nmanifest → s3://{config.BUCKET}/{config.MANIFEST_KEY} and ./manifest.json")


if __name__ == "__main__":
    main()
