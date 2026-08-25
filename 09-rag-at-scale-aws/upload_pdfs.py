"""Upload PDFs to S3, sorted into size tiers.

    python upload_pdfs.py --dir ./pdfs --dry-run     see the split, upload nothing
    python upload_pdfs.py --dir ./pdfs               upload and write the manifest

This is the only step between having PDFs on your machine and running the pipeline.
It does three things:

    1. counts the pages in each PDF
    2. sorts them into small, medium and large
    3. uploads each to s3://BUCKET/raw/<tier>/ and writes a manifest

The manifest is what `run.py` reads to start the job.

Page count decides the tier, not file size. A 33 MB scanned report can be shorter
than a 4 MB text one, and it is pages that drive both parse time and memory. Each
tier maps to a Fargate task definition with different CPU and memory, so a 15-page
document does not reserve a 16 GB container.
"""

import argparse
import json
import warnings
from pathlib import Path

import boto3
from pypdf import PdfReader

import config

# Upper page bound per tier. The last has no bound, so anything larger falls into
# it. These boundaries are what the CPU and memory in config.py were sized for;
# moving one without the other gives you an under- or over-provisioned lane.
TIERS = [("small", 50), ("medium", 150), ("large", None)]


def page_count(pdf: Path) -> int:
    """Number of pages, read from the PDF structure without rendering anything.

    Malformed cross-reference tables are common in generated PDFs and warn on almost
    every file. They do not affect the page count, so the noise is suppressed.
    """
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
    """Sort a folder of PDFs into tiers, upload them, and write the manifest.

    Always run with --dry-run first. The printed distribution tells you whether the
    concurrency settings in config.py suit this corpus: if most documents land in
    one lane, that lane alone decides your total wall clock and the other two are
    decoration.
    """
    ap = argparse.ArgumentParser(description="Upload PDFs to S3, sorted into tiers")
    ap.add_argument("--dir", required=True, help="folder containing the PDFs")
    ap.add_argument("--dry-run", action="store_true",
                    help="sort and report, upload nothing")
    args = ap.parse_args()

    pdfs = sorted(Path(args.dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found in {args.dir}")

    manifest = {name: [] for name, _ in TIERS}
    print(f"{'file':<48}{'pages':>7}  tier")
    print("-" * 66)

    for pdf in pdfs:
        pages = page_count(pdf)
        tier = tier_for(pages)
        key = f"{config.RAW_PREFIX}/{tier}/{pdf.name}"
        # The local path rides along so the upload loop never has to work out which
        # file produced which entry — which would mean reading every PDF twice.
        manifest[tier].append({"bucket": config.BUCKET, "key": key,
                               "pages": pages, "name": pdf.name, "path": str(pdf)})
        print(f"{pdf.name:<48}{pages:>7}  {tier}")

    # This summary is the point of --dry-run. Compare the page totals against the
    # concurrency in config.py: a lane's time is roughly pages / concurrency, and
    # the slowest lane is your wall clock.
    print("-" * 66)
    for tier, items in manifest.items():
        pages = sum(i["pages"] for i in items)
        concurrency = config.TIERS[tier]["concurrency"]
        print(f"{tier:<10}{len(items):>4} documents{pages:>7} pages"
              f"{concurrency:>4} at a time")

    if args.dry_run:
        print("\ndry run — nothing uploaded")
        return

    s3 = boto3.client("s3", region_name=config.REGION)
    print()
    for items in manifest.values():
        for item in items:
            s3.upload_file(item["path"], config.BUCKET, item["key"])
            print(f"uploaded  {item['key']}")

    # The state machine reads bucket and key from each entry. The local path is
    # dropped, so the manifest describes only what exists in S3 — otherwise a run
    # started from another machine would carry paths that do not resolve.
    body = json.dumps({tier: [{k: v for k, v in i.items() if k != "path"}
                              for i in items]
                       for tier, items in manifest.items()}, indent=2)
    s3.put_object(Bucket=config.BUCKET, Key=config.MANIFEST_KEY, Body=body.encode())
    Path("manifest.json").write_text(body)

    print(f"\nmanifest written to s3://{config.BUCKET}/{config.MANIFEST_KEY}")
    print("next: python run.py")


if __name__ == "__main__":
    main()
