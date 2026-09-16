"""Triggered by the audit table's DynamoDB Stream. When a document's run
record transitions to FAILED, copies the source PDF to error/ and deletes it
from raw/.

    DynamoDB Stream (NEW_AND_OLD_IMAGES)
        |
        v
    this Lambda
        |
        +-- is this a RUN record (not a STAGE record)?
        +-- did status just BECOME FAILED (not already FAILED)?
        |
        v
    copy raw/<file>.pdf -> error/<file>.pdf, delete the raw/ copy

WHY "JUST BECAME FAILED" AND NOT "IS FAILED"

The stream delivers every write to the table, including retries. Batch's own
retry (RETRY_ATTEMPTS in config.py) means a document can transition
STARTED -> FAILED -> STARTED again -> COMPLETED across two attempts. If this
handler acted on every record where status happens to equal FAILED, it would
move the PDF out of raw/ after the FIRST attempt's failure — deleting the
input a still-pending retry needs to read.

NEW_AND_OLD_IMAGES on the stream (configured in infra/storage.py) is what
makes the distinction possible: this only fires the move when OLD.status was
NOT already FAILED and NEW.status IS FAILED, which is the transition into a
final failure, not a transient one Batch is about to retry on its own.

A record with no OLD image at all (the very first write for a document) can
never be a FAILED transition — nothing fails on its first ever write — so
that case is skipped without needing special-casing.
"""

import os

import boto3

BUCKET = os.environ["BUCKET"]

# See lambda_classifier/handler.py's comment on this exact line for why
# region_name is passed explicitly rather than left to boto3's default
# resolution.
s3 = boto3.client("s3", region_name=os.environ["AWS_REGION"])


def _attr(image: dict, name: str) -> str | None:
    """Pull a plain string out of a DynamoDB Streams record image, which
    wraps every value in a type descriptor ({"S": "value"}) rather than
    storing it directly."""
    field = image.get(name)
    return field.get("S") if field else None


def handler(event, context):
    moved = []

    for record in event.get("Records", []):
        if record.get("eventName") != "MODIFY":
            # INSERT records have no OLD image and so can never represent a
            # FAILED transition (see module docstring); REMOVE is not
            # something this table's access pattern produces.
            continue

        new_image = record["dynamodb"].get("NewImage", {})
        old_image = record["dynamodb"].get("OldImage", {})

        sk = _attr(new_image, "sk") or ""
        if "#STAGE#" in sk:
            # A stage record transitioning, e.g. STARTED -> OK. Only the
            # RUN-level record's own status is the source of truth for
            # whether the whole document failed.
            continue

        new_status = _attr(new_image, "status")
        old_status = _attr(old_image, "status")
        if new_status != "FAILED" or old_status == "FAILED":
            continue

        pk = _attr(new_image, "pk") or ""
        doc_id = pk.removeprefix("DOC#")
        source_key = _attr(new_image, "source_key") or _attr(new_image, "source")
        if not source_key:
            print(f"{doc_id}: FAILED but no source_key on the record, "
                  "cannot locate the PDF to move", flush=True)
            continue

        error_key = source_key.replace("raw/", "error/", 1)
        try:
            s3.copy_object(Bucket=BUCKET,
                           CopySource={"Bucket": BUCKET, "Key": source_key},
                           Key=error_key)
            s3.delete_object(Bucket=BUCKET, Key=source_key)
            print(f"{doc_id}: moved {source_key} -> {error_key}", flush=True)
            moved.append(doc_id)
        except s3.exceptions.NoSuchKey:
            # The document may have been reprocessed and its raw/ copy
            # already cleaned up by something else between the failure being
            # recorded and this handler running. Not an error worth failing
            # the Lambda invocation over.
            print(f"{doc_id}: {source_key} no longer in raw/, "
                  "nothing to move", flush=True)

    return {"moved": moved}
