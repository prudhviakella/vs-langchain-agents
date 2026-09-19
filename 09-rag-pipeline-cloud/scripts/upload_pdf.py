#!/usr/bin/env python3
"""Upload a PDF to raw/, which starts the pipeline automatically.

    python scripts/upload_pdf.py path/to/document.pdf

That's the whole interface. There is no --tier flag and no separate
"submit" step — landing the file under raw/ is what the S3 event
notification is wired to, so the classifier runs, decides the tier, and
submits the Batch job with no further action needed. This script exists
only because "upload a file to a specific S3 key" is otherwise a few lines
of boilerplate to remember correctly every time.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import boto3

from infra import config

s3 = boto3.client("s3", region_name=config.REGION)
sts = boto3.client("sts", region_name=config.REGION)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf", type=Path, help="local PDF to upload")
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"{args.pdf} does not exist")
    if args.pdf.suffix.lower() != ".pdf":
        raise SystemExit(f"{args.pdf} is not a .pdf — the trigger filters on "
                         "that suffix and would silently ignore it")

    account_id = sts.get_caller_identity()["Account"]
    bucket = config.bucket_name(account_id)
    key = f"{config.RAW_PREFIX}/{args.pdf.name}"

    print(f"uploading {args.pdf} -> s3://{bucket}/{key}")
    s3.upload_file(str(args.pdf), bucket, key)
    print("uploaded — the classifier will run within a few seconds")

    # Same regex as rag.config.slugify / lambda_classifier's own copy of it,
    # verified identical earlier in this project — reproduced here only to
    # print a correct doc_id hint, not as a third independent copy anything
    # else depends on.
    import re
    doc_id = re.sub(r"[^a-z0-9]+", "-", args.pdf.stem.lower()).strip("-")[:48]
    print(f"\ncheck progress with:\n  python scripts/status.py {doc_id}")


if __name__ == "__main__":
    main()
