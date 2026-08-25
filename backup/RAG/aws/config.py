"""Names and sizes for every AWS resource this project creates.

One module rather than constants scattered across scripts, because setup, run,
status and teardown all have to agree on what things are called. A name that drifts
between setup and teardown leaves an orphaned resource billing quietly.

Change PROJECT to run a second, isolated copy of the whole stack — useful for
letting a class create their own without colliding.
"""

import os

# Every resource name derives from this, so one value namespaces the whole stack.
PROJECT = os.getenv("PROJECT", "rag-pipeline")
REGION = os.getenv("AWS_REGION", "us-east-1")

# S3 bucket names are globally unique across all AWS accounts, not per-account, so
# the account id is appended to avoid collisions with anyone else's rag-pipeline.
BUCKET = os.getenv("BUCKET", f"{PROJECT}-{os.getenv('AWS_ACCOUNT_ID', 'docs')}")

RAW_PREFIX = "raw"            # uploaded PDFs, under raw/<tier>/
MANIFEST_KEY = "manifest.json"  # the tier-sorted document list run.py reads

AUDIT_TABLE = f"{PROJECT}-audit"
CLUSTER = f"{PROJECT}-cluster"
LOG_GROUP = f"/ecs/{PROJECT}"
ECR_REPO = PROJECT
SECRET_NAME = f"{PROJECT}/api-keys"
STATE_MACHINE = f"{PROJECT}-ingest"

# Three roles, not one. See create_roles() in setup.py for why the execution and
# task roles are separate — it is the least obvious part of the ECS model.
TASK_ROLE = f"{PROJECT}-task-role"
EXEC_ROLE = f"{PROJECT}-exec-role"
SFN_ROLE = f"{PROJECT}-sfn-role"

# ─────────────────────────────────────────────────────────────────────────────
# Tiers
#
# Page count, not file size, decides the lane: a 33 MB scanned report can be
# shorter than a 4 MB text one, and it is pages that drive parse time and memory.
#
# Concurrency multiplies into the account's Fargate vCPU quota. Peak demand is
# sum(cpu/1024 * concurrency) across all lanes, because the Parallel state starts
# every branch at once — 44 vCPU for these values. New accounts default to 6. Run
# aws/preflight.py before the first run.
#
# The concurrency values come from measured lane load, not intuition. On the
# 20-document reference corpus 11 documents are large, so that lane alone decides
# total wall clock: at concurrency 1 it runs about 52 minutes against 70 sequential,
# which is not worth building. At 3 it runs about 17. Fargate bills per task-second,
# so the higher setting costs the same and finishes sooner. Re-derive this for your
# own corpus with `upload_pdfs.py --dry-run`.
#
# Fargate accepts only specific cpu/memory pairs; these are valid combinations.
# ─────────────────────────────────────────────────────────────────────────────
TIERS = {
    "small":  {"cpu": "2048", "memory": "4096",  "concurrency": 4},
    "medium": {"cpu": "4096", "memory": "8192",  "concurrency": 3},
    "large":  {"cpu": "8192", "memory": "16384", "concurrency": 3},
}

# Wall-clock ceiling per document. A large clinical protocol with many figures can
# run well over an hour, so this is generous — but it exists so a hung task fails
# its branch rather than holding a lane open indefinitely.
TASK_TIMEOUT_SECONDS = 7200
