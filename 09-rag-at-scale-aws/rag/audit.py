"""The audit trail, written to DynamoDB when running on AWS.

Two record shapes in one table:

    pk = DOC#<doc_id>   sk = RUN#<run_id>                  one per run
    pk = DOC#<doc_id>   sk = RUN#<run_id>#STAGE#<stage>    one per stage

The stage records are what make a killed container debuggable. A run record alone
says the document failed; the stage records say it failed during `parse`, forty
minutes in, which points at memory rather than at credentials.
"""

import time
from datetime import datetime, timezone

class Audit:
    """Writes stage records to DynamoDB. A no-op when running locally.

    Making this a no-op rather than a separate code path means the notebook and the
    Fargate task execute the same lines. There is no "local mode" behaving
    differently, and therefore no local-only bug.
    """

    def __init__(self, table_name: str | None, doc_id: str, run_id: str):
        self.table = None
        self.doc_id, self.run_id = doc_id, run_id
        if table_name:
            # Imported lazily so the local path never needs boto3 installed.
            import boto3
            self.table = boto3.resource("dynamodb").Table(table_name)

    def _put(self, sk: str, **fields) -> None:
        """Write one record. Silently does nothing when no table is configured."""
        if self.table is None:
            return
        self.table.put_item(Item={
            "pk": f"DOC#{self.doc_id}", "sk": sk,
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # DynamoDB rejects None, so empty fields are dropped rather than nulled.
            **{k: v for k, v in fields.items() if v is not None},
        })

    def stage(self, name: str, status: str, **fields) -> None:
        """Record a stage transition and echo it to the log."""
        self._put(f"RUN#{self.run_id}#STAGE#{name}", stage=name, status=status, **fields)
        # flush on every print: CloudWatch buffers stdout, and an unflushed buffer is
        # lost when a container is killed — exactly when the logs matter most.
        print(f"  [{status:9s}] {name}", flush=True)

    def run(self, status: str, **fields) -> None:
        """Record a run-level transition: STARTED, COMPLETED or FAILED."""
        self._put(f"RUN#{self.run_id}", status=status, **fields)


class Stage:
    """Times a stage and records start, success or failure.

        with Stage(audit, "parse"):
            doc = parse_pdf(...)

    Writing the STARTED record before the work begins is the point. A stage with a
    start and no end is a container that died mid-stage, which no after-the-fact
    logging can tell you.
    """

    def __init__(self, audit: Audit, name: str):
        self.audit, self.name = audit, name

    def __enter__(self) -> "Stage":
        """Start the timer and write the STARTED record before any work runs."""
        self.started = time.time()
        self.audit.stage(self.name, "STARTED")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Record the outcome, then re-raise so the caller still sees the error."""
        elapsed = round(time.time() - self.started, 1)
        if exc_type is None:
            self.audit.stage(self.name, "OK", duration_s=str(elapsed))
        else:
            self.audit.stage(self.name, "FAILED", duration_s=str(elapsed),
                             error=str(exc)[:400])
        # False re-raises. The audit record is a side effect, not a handler: main()
        # still needs the exception so the task exits non-zero.
        return False
