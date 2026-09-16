"""Triggered by the audit table's DynamoDB Stream. When a document's Stage 1
run record transitions to COMPLETED, submits the corresponding Stage 2
(graph build) job.

    DynamoDB Stream (NEW_AND_OLD_IMAGES)
        |
        v
    this Lambda
        |
        +-- is this a Stage 1 RUN record (not a STAGE record, not a
        |   Stage 2 GRAPH record)?
        +-- did status just BECOME COMPLETED (not already COMPLETED)?
        |
        v
    submit_job(graph queue, --bucket, --doc-id)

WHY THIS EXISTS

Stage 1 and Stage 2 were always designed to be decoupled — see the parent
README's own reasoning: a transient Neo4j problem should not cost a
nine-minute re-parse, and the whole graph can be rebuilt from S3 without
re-parsing anything. But decoupled still needs *something* to connect
them, and until this Lambda existed nothing did: the queue, job
definition, and container all existed, but no path from "Stage 1
finished" to "Stage 2 starts" did. Confirmed directly — grepped the whole
codebase for any submit_job call targeting the graph queue and found
none, before this file existed.

WHY "JUST BECAME COMPLETED" AND NOT "IS COMPLETED"

Same reasoning as lambda_error_handler's own FAILED-transition check, just
for the opposite status. The stream delivers every write, and NEW_AND_OLD_
IMAGES (configured in infra/storage.py) is what makes a genuine transition
distinguishable from a record that was already COMPLETED before this
invocation ever ran — the second case must never resubmit Stage 2 for a
document that already has a graph.

WHY GRAPH# RECORDS CAN NEVER FALSE-TRIGGER THIS

Stage 2's own audit write (build_graph.py's build_one, on success) ALSO
sets status to "COMPLETED" — on a record whose sk starts with "GRAPH#",
never "RUN#". The filter below only acts on sk starting with "RUN#" (and
not containing "#STAGE#"), which is naturally never true for a "GRAPH#..."
key — no extra special-case needed to keep a completed Stage 2 run from
re-triggering itself into a submit loop.
"""

import os
import sys

import boto3

BUCKET = os.environ["BUCKET"]
PARAM_PREFIX = os.environ["PARAM_PREFIX"]

# See lambda_classifier/handler.py's comment on this exact line for why
# region_name is passed explicitly rather than left to boto3's default
# resolution.
_REGION = os.environ["AWS_REGION"]
ssm = boto3.client("ssm", region_name=_REGION)
batch = boto3.client("batch", region_name=_REGION)

# Looked up once per cold start and cached, same reasoning as the
# classifier's own _JOBDEFS/_QUEUES: a graph job definition ARN and queue
# name essentially never change between one Stage 1 completion and the
# next.
_GRAPH_JOBDEF = {}
_GRAPH_QUEUE = {}


def _load_graph_resources() -> None:
    if _GRAPH_JOBDEF:
        return
    _GRAPH_JOBDEF["arn"] = ssm.get_parameter(
        Name=f"{PARAM_PREFIX}/JOBDEF_GRAPH")["Parameter"]["Value"]
    _GRAPH_QUEUE["name"] = ssm.get_parameter(
        Name=f"{PARAM_PREFIX}/QUEUE_GRAPH")["Parameter"]["Value"]


def _attr(image: dict, name: str) -> str | None:
    """Pull a plain string out of a DynamoDB Streams record image, which
    wraps every value in a type descriptor ({"S": "value"}) rather than
    storing it directly."""
    field = image.get(name)
    return field.get("S") if field else None


def handler(event, context):
    _load_graph_resources()
    submitted = []

    for record in event.get("Records", []):
        if record.get("eventName") != "MODIFY":
            # A COMPLETED status is never a document's first-ever write —
            # STARTED always precedes it in the same run — so this is
            # never an INSERT. REMOVE is not something this table's
            # access pattern produces.
            continue

        new_image = record["dynamodb"].get("NewImage", {})
        old_image = record["dynamodb"].get("OldImage", {})

        sk = _attr(new_image, "sk") or ""
        if not sk.startswith("RUN#") or "#STAGE#" in sk:
            # Not a Stage 1 run-level record — either a per-stage record
            # (parse/inspect/figures/chunk/index) or a Stage 2 GRAPH#
            # record. Only the run-level record's own status is the
            # source of truth for whether Stage 1, as a whole, finished.
            continue

        new_status = _attr(new_image, "status")
        old_status = _attr(old_image, "status")
        if new_status != "COMPLETED" or old_status == "COMPLETED":
            continue

        pk = _attr(new_image, "pk") or ""
        doc_id = pk.removeprefix("DOC#")
        run_id = sk.removeprefix("RUN#")

        job_name = f"{doc_id}-graph-{run_id}"[:128]  # Batch job name length limit
        try:
            batch.submit_job(
                jobName=job_name,
                jobQueue=_GRAPH_QUEUE["name"],
                jobDefinition=_GRAPH_JOBDEF["arn"],
                containerOverrides={
                    # NOT ["python", "build_graph.py", ...] — the graph
                    # worker's Dockerfile already sets
                    # ENTRYPOINT ["python", "build_graph.py"], which ECS
                    # always prepends. Checked directly before writing
                    # this, having made exactly the opposite mistake once
                    # already today in lambda_classifier/handler.py.
                    "command": ["--bucket", BUCKET, "--doc-id", doc_id],
                },
            )
            print(f"{doc_id}: Stage 1 completed, submitted graph job "
                 f"{job_name}", flush=True)
            submitted.append(doc_id)
        except batch.exceptions.ClientException as exc:
            # Logged, not raised — a Lambda invocation failing here would
            # be retried by the stream's own retry policy against a
            # transition that already happened and will never be seen
            # again in this exact form. Better to surface it in
            # CloudWatch than to spin retries against an unrepeatable
            # event.
            print(f"{doc_id}: failed to submit graph job: {exc}",
                 file=sys.stderr, flush=True)

    return {"submitted": submitted}
