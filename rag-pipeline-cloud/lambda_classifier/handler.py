"""Triggered by S3 when a PDF lands under raw/. Classifies it by size and
submits the matching Stage 1 Batch job.

    S3 raw/<file>.pdf
        |
        v
    this Lambda
        |
        +-- read page count (pypdf, no rendering)
        +-- consult DynamoDB for this document's own processing history
        +-- pick small / medium / large
        |
        v
    batch:SubmitJob on the matching queue

WHY HISTORY BEFORE PAGE COUNT

Page count predicts processing cost badly on this corpus — measured directly
earlier in this project: a 192-page document took 9 minutes, a 114-page one
took 19.6 hours. A page-count rule alone would size that second document as
"large" and still be wrong, because the actual driver was something else
entirely (page complexity, not page count).

So the classifier looks first for a PREVIOUS run of this exact document (by
doc_id, which is stable across re-uploads of the same file) and reuses
whatever tier its actual measured duration implies. Only a document with no
history falls back to the page-count table in config.py, which is a starting
guess, not a promise.

This means a Spot-interrupted or resized-and-retried document self-corrects
over successive runs: badly-classified once, correctly classified from then
on.
"""

import os
from datetime import datetime, timezone

import boto3

BUCKET = os.environ["BUCKET"]
AUDIT_TABLE = os.environ["AUDIT_TABLE"]
PARAM_PREFIX = os.environ["PARAM_PREFIX"]

# region_name passed explicitly rather than left to boto3's default-session
# resolution. Lambda's runtime always sets AWS_REGION, but boto3's implicit
# resolution was verified to check AWS_DEFAULT_REGION first and NOT fall
# back to AWS_REGION reliably — confirmed directly by running this exact
# code with only AWS_REGION set, which raised NoRegionError. Passing it
# explicitly removes any dependency on which variable a given boto3 version
# happens to check.
_REGION = os.environ["AWS_REGION"]
s3 = boto3.client("s3", region_name=_REGION)
ddb = boto3.resource("dynamodb", region_name=_REGION).Table(AUDIT_TABLE)
ssm = boto3.client("ssm", region_name=_REGION)
batch = boto3.client("batch", region_name=_REGION)

# Page-count fallback for a document with no processing history. Mirrors
# infra/config.py's STAGE1_TIERS boundaries; duplicated rather than imported
# because this Lambda's deployment package does not include the infra
# module — it is a thin, standalone function by design, not a slice of the
# provisioning code.
PAGE_CEILINGS = [("small", 50), ("medium", 150), ("large", None)]

# Ordered so "the next tier up" is just the next entry — used only by
# escalate_for_ocr below, never to walk downward.
TIER_ORDER = ["small", "medium", "large"]

# Escalate one tier when at least this share of pages have no real text
# layer. Set well above zero deliberately: one scanned exhibit page glued
# into an otherwise-digital protocol should not retier the whole document —
# this is meant to catch a document that is substantially or wholly a scan,
# where OCR runs across most of it rather than a handful of pages.
ESCALATE_OCR_FRACTION = 0.2

# Loaded once per COLD START, not per invocation. A warm Lambda container is
# reused across many S3 events, and Parameter Store has its own request
# rate limits — calling it on every invocation costs latency for no benefit,
# since job definition ARNs and queue names essentially never change
# between one PDF landing and the next.
_JOBDEFS = {}
_QUEUES = {}


def _load_batch_resources() -> None:
    if _JOBDEFS:
        return
    for tier in ("SMALL", "MEDIUM", "LARGE"):
        _JOBDEFS[tier.lower()] = ssm.get_parameter(
            Name=f"{PARAM_PREFIX}/JOBDEF_{tier}")["Parameter"]["Value"]
        _QUEUES[tier.lower()] = ssm.get_parameter(
            Name=f"{PARAM_PREFIX}/QUEUE_{tier}")["Parameter"]["Value"]


def slugify(text: str) -> str:
    """Must match rag.config.slugify EXACTLY, character for character — this
    is how doc_id is derived on both the classifier side and the worker
    side, and a mismatch here means history lookups silently never find a
    match, forever, with no error raised anywhere to say so.

    Copied verbatim rather than re-derived from the description, after
    discovering by direct comparison that a first attempt using a
    \\w-based character class diverged from the real implementation on
    every filename containing an underscore — \\w already matches
    underscore, so it was never being replaced. Confirmed identical output
    against the real function before trusting this copy.
    """
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def scan_pdf(bucket: str, key: str) -> dict:
    """One pass over the PDF: page count, and two cheap content signals used
    to adjust the page-count guess before any history exists.

    WHAT THIS CAN AND CANNOT TELL YOU

    PyMuPDF gives real answers, fast (measured: 0.66s for a 113-page
    document) for two things:

        ocr_fraction   share of pages with almost no extractable text layer
                       — a genuine scanned page, not a proxy for one. This
                       is the signal actually worth escalating on: OCR only
                       runs on regions with no text layer at all, and is
                       measured elsewhere in this project to be
                       considerably slower per page than digital text.

        image_bytes    total size of embedded images, reported for
                       visibility, NOT used to escalate the compute tier.
                       Figure description is a REMOTE API call — more vCPU
                       on the worker does not make OpenAI respond faster.
                       A weaker, honest signal than OCR fraction, so it
                       stays out of the tier decision and only appears in
                       the log for whoever is tuning ESCALATE_OCR_FRACTION
                       later.

    It CANNOT tell you real table count or complexity — that needs the kind
    of trained layout model Docling's TableFormer is, which is exactly the
    expensive thing Stage 1 exists to run. A cheap heuristic here (counting
    vector rectangles, the usual proxy) would be a guess dressed up as a
    measurement, so this does not attempt it.
    """
    import pymupdf

    obj = s3.get_object(Bucket=bucket, Key=key)
    doc = pymupdf.open(stream=obj["Body"].read(), filetype="pdf")

    thin_text_pages = 0
    image_count = 0
    image_bytes = 0
    for page in doc:
        if len(page.get_text("text").strip()) < 50:
            thin_text_pages += 1
        images = page.get_images(full=True)
        image_count += len(images)
        for img in images:
            try:
                image_bytes += len(doc.extract_image(img[0])["image"])
            except Exception:
                # A malformed or unsupported image entry costs nothing to
                # skip — this is a classification hint, not the real parse,
                # and Stage 1's own parser will surface a genuine problem.
                pass

    pages = len(doc)
    return {"pages": pages,
           "ocr_fraction": thin_text_pages / pages if pages else 0.0,
           "image_count": image_count, "image_bytes": image_bytes}


def tier_from_history(doc_id: str) -> str | None:
    """The tier implied by this document's most recent successful run, or
    None if it has never completed one.

    Queries the most recent RUN record (not a stage record) for this doc_id.
    A COMPLETED run's duration is compared against the same page-ceiling-
    style logic, but on measured seconds rather than page count, using a
    rough seconds-per-tier expectation: a run that took over an hour was
    undersized for whatever tier it ran in last time.
    """
    response = ddb.query(
        KeyConditionExpression="pk = :pk AND begins_with(sk, :sk)",
        ExpressionAttributeValues={":pk": f"DOC#{doc_id}", ":sk": "RUN#"},
        ScanIndexForward=False,  # most recent first
        Limit=5,
    )
    for item in response.get("Items", []):
        if item.get("status") != "COMPLETED":
            continue
        duration = float(item.get("duration_s", 0))
        # A document that took over an hour needs more headroom than
        # whatever tier it ran in, regardless of what that tier was —
        # escalate rather than trust the same classification twice.
        if duration > 3600:
            return "large"
        if duration > 600:
            return "medium"
        return "small"
    return None


def tier_from_pages(pages: int) -> str:
    for tier, ceiling in PAGE_CEILINGS:
        if ceiling is None or pages <= ceiling:
            return tier
    return "large"


def escalate_for_ocr(tier: str, ocr_fraction: float) -> str:
    """Bump the page-count guess up one tier for a substantially-scanned
    document.

    Only ever moves up, and only by one step — this adjusts a guess, it
    does not replace it. A document already at "large" stays there; there
    is nowhere further to escalate to in this project's three tiers.
    """
    if ocr_fraction < ESCALATE_OCR_FRACTION:
        return tier
    index = TIER_ORDER.index(tier)
    return TIER_ORDER[min(index + 1, len(TIER_ORDER) - 1)]


def handler(event, context):
    _load_batch_resources()
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        filename = key.rsplit("/", 1)[-1]
        doc_id = slugify(filename.removesuffix(".pdf"))

        scan = scan_pdf(bucket, key)
        pages = scan["pages"]
        history_tier = tier_from_history(doc_id)
        if history_tier:
            # Real measured history always wins outright — a content-based
            # guess is worth nothing next to what the document actually did
            # last time it ran.
            tier = history_tier
            reason = "history"
        else:
            page_tier = tier_from_pages(pages)
            tier = escalate_for_ocr(page_tier, scan["ocr_fraction"])
            reason = ("page count" if tier == page_tier
                      else f"page count + {scan['ocr_fraction']:.0%} scanned")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_name = f"{doc_id}-{run_id}"[:128]  # Batch job name length limit

        response = batch.submit_job(
            jobName=job_name,
            jobQueue=_QUEUES[tier],
            jobDefinition=_JOBDEFS[tier],
            containerOverrides={
                # NOT ["python", "ingest.py", "--bucket", ...] — the worker
                # image's Dockerfile already sets
                # ENTRYPOINT ["python", "ingest.py"], and ECS always
                # PREPENDS the entrypoint to whatever command is given here;
                # it never replaces it. Including "python", "ingest.py"
                # again produced, at actual container launch:
                #
                #   python ingest.py python ingest.py --bucket ... --key ...
                #
                # which argparse then rejected as "unrecognized arguments:
                # python ingest.py" — confirmed directly from the real
                # CloudWatch log line, not inferred. This field is only the
                # ARGUMENTS to append after the entrypoint, not a
                # standalone command.
                "command": ["--bucket", bucket, "--key", key],
            },
        )

        ddb.put_item(Item={
            "pk": f"DOC#{doc_id}", "sk": f"RUN#{run_id}",
            "status": "SUBMITTED", "tier": tier,
            "classification_reason": reason, "pages": str(pages),
            "ocr_fraction": f"{scan['ocr_fraction']:.2f}",
            "image_count": str(scan["image_count"]),
            "batch_job_id": response["jobId"], "source_key": key,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

        print(f"{doc_id}: {pages} pages, {scan['image_count']} image(s) "
              f"({scan['image_bytes']/1024/1024:.1f}MB), "
              f"{scan['ocr_fraction']:.0%} scanned -> {tier} ({reason}), "
              f"job {response['jobId']}", flush=True)
        results.append({"doc_id": doc_id, "tier": tier,
                        "job_id": response["jobId"]})

    return {"classified": results}
