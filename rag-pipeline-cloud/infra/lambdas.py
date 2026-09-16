"""Package and deploy all three Lambda functions, and wire all three triggers.

Three functions, two different packaging needs:

    classifier       needs pymupdf, which is not in the base Lambda Python
                     runtime — its dependencies are installed into the
                     deployment zip directly.
    error_handler    needs only boto3, already present in every Lambda
                     Python runtime — packaged as a bare zip of the handler.
    graph_trigger    same as error_handler — only boto3, bare zip.

Three triggers wired here, each requiring BOTH a resource-based permission on
the Lambda (so the calling service is allowed to invoke it) and the
subscription itself on the other side. Missing either half is a known way
for "the trigger does nothing" to happen with no error anywhere:

    S3 -> classifier                   permission granted here;
                                       notification config lives in storage.py
    DynamoDB Stream -> error_handler   both halves live here together,
                                       since Lambda event source mappings
                                       are configured lambda-side, unlike
                                       S3 notifications which are bucket-side
    DynamoDB Stream -> graph_trigger   a second, independent subscriber on
                                       the SAME stream — one watches for a
                                       transition to FAILED, the other for a
                                       transition to COMPLETED; each has its
                                       own iterator position and neither
                                       affects what the other sees
"""

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from . import config

lam = boto3.client("lambda", region_name=config.REGION)

HERE = Path(__file__).resolve().parent.parent


def exists(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except ClientError:
        return False


def _build_zip(source_dir: Path, requirements: Path | None) -> bytes:
    """A deployment package: the handler plus, if requirements.txt exists,
    its dependencies installed alongside it.

    pip installs into a temp directory rather than the source tree directly,
    so re-running deploy never accumulates stale installed packages inside
    the project's own source folder.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as build_dir:
        build_path = Path(build_dir)
        for py_file in source_dir.glob("*.py"):
            (build_path / py_file.name).write_bytes(py_file.read_bytes())

        if requirements and requirements.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                "-r", str(requirements), "--target", str(build_path),
                "--only-binary=:all:", "--platform", "manylinux2014_x86_64",
                "--python-version", "3.11"],
                check=True,
            )

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in build_path.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(build_path))
        return buffer.getvalue()


def deploy_function(name: str, source_dir: Path, role_arn: str, handler: str,
                    env: dict, timeout: int = 60, memory: int = 256) -> str:
    """Create or update a function, returning its ARN.

    Update rather than skip on an existing function — a re-deploy after a
    code change to the handler must actually ship that change, not silently
    leave the old version running because a name-exists check short-circuited
    everything else.
    """
    print(f"-- Lambda {name} --")
    package = _build_zip(source_dir, source_dir / "requirements.txt")

    if exists(lam.get_function, FunctionName=name):
        lam.update_function_code(FunctionName=name, ZipFile=package)
        waiter = lam.get_waiter("function_updated")
        waiter.wait(FunctionName=name)
        lam.update_function_configuration(
            FunctionName=name, Environment={"Variables": env},
            Timeout=timeout, MemorySize=memory)
        print("   code and configuration updated")
    else:
        lam.create_function(
            FunctionName=name, Runtime="python3.11", Role=role_arn,
            Handler=handler, Code={"ZipFile": package},
            Environment={"Variables": env}, Timeout=timeout,
            MemorySize=memory,
        )
        waiter = lam.get_waiter("function_active")
        waiter.wait(FunctionName=name)
        print("   created")

    return lam.get_function(FunctionName=name)["Configuration"]["FunctionArn"]


def wire_s3_trigger(function_name: str, function_arn: str, bucket: str,
                    account_id: str) -> None:
    """Grant S3 permission to invoke the classifier.

    This is HALF of the S3 trigger — the other half, the bucket's
    notification configuration itself, is storage.wire_upload_trigger().
    Both must exist; this half alone means S3 is never told to call
    anything, and that half alone means the call would be rejected for lack
    of permission. Neither failure raises an error pointing at the other
    half — this is worth stating plainly because it is exactly the kind of
    thing that "works in testing" (invoked directly) and silently does
    nothing end-to-end.
    """
    statement_id = "AllowS3Invoke"
    try:
        lam.add_permission(
            FunctionName=function_name, StatementId=statement_id,
            Action="lambda:InvokeFunction", Principal="s3.amazonaws.com",
            SourceArn=f"arn:aws:s3:::{bucket}",
            SourceAccount=account_id,
        )
        print(f"   S3 invoke permission granted to {function_name}")
    except lam.exceptions.ResourceConflictException:
        print(f"   S3 invoke permission already present on {function_name}")


def wire_stream_trigger(function_name: str, stream_arn: str) -> None:
    """Subscribe the error handler to the audit table's DynamoDB Stream."""
    existing = lam.list_event_source_mappings(
        FunctionName=function_name, EventSourceArn=stream_arn)
    if existing["EventSourceMappings"]:
        print(f"   stream trigger already wired for {function_name}")
        return
    lam.create_event_source_mapping(
        EventSourceArn=stream_arn, FunctionName=function_name,
        StartingPosition="LATEST", BatchSize=10,
    )
    print(f"   stream trigger wired for {function_name}")


def deploy_all(account_id: str, bucket: str, roles: dict, stream_arn: str) -> dict:
    """Deploy all three functions and wire all three triggers."""
    # 900 seconds is not a chosen value — it is Lambda's hard ceiling for a
    # standard function, unconditional and not raisable even by AWS support
    # request (confirmed against AWS's own current documentation: fixed at
    # 900s per invocation). Set here mainly for the classifier: it downloads
    # the whole PDF via s3.get_object before scanning it with PyMuPDF, and an
    # unusually large document — hundreds of MB, heavy on embedded images —
    # could plausibly take longer than the previous 30s budget on a cold
    # start. Raising the ceiling costs nothing when unused; Lambda bills by
    # actual duration, not by the configured timeout.
    classifier_arn = deploy_function(
        config.CLASSIFIER_FUNCTION, HERE / "lambda_classifier",
        roles["classifier"], "handler.handler",
        env={"BUCKET": bucket, "AUDIT_TABLE": config.AUDIT_TABLE,
            "PARAM_PREFIX": config.PARAM_PREFIX},
        timeout=900, memory=256,
    )
    # Raised to match, for consistency, though its own work (an S3 copy and
    # a delete) never approaches even the previous 30s. Worth knowing the
    # real tradeoff: a genuine hang in this function now takes 15 minutes to
    # surface as a timeout instead of 30 seconds — a worse debugging signal
    # for a failure mode this function is unlikely to ever hit. Lower this
    # back down if that matters more than having one uniform value.
    error_handler_arn = deploy_function(
        config.ERROR_HANDLER_FUNCTION, HERE / "lambda_error_handler",
        roles["error_handler"], "handler.handler",
        env={"BUCKET": bucket},
        timeout=900, memory=128,
    )
    # Only boto3, same as error_handler — no requirements.txt needed.
    # Its own work (one GetParameter lookup, cached per cold start, then
    # one SubmitJob call) never approaches even a fraction of this budget;
    # matched to the other two mainly for one uniform value across all
    # three functions rather than a specific need of its own.
    graph_trigger_arn = deploy_function(
        config.GRAPH_TRIGGER_FUNCTION, HERE / "lambda_graph_trigger",
        roles["graph_trigger"], "handler.handler",
        env={"BUCKET": bucket, "PARAM_PREFIX": config.PARAM_PREFIX},
        timeout=900, memory=128,
    )

    wire_s3_trigger(config.CLASSIFIER_FUNCTION, classifier_arn, bucket, account_id)
    from . import storage
    storage.wire_upload_trigger(bucket, classifier_arn)

    wire_stream_trigger(config.ERROR_HANDLER_FUNCTION, stream_arn)
    # Same stream, a second independent subscriber — DynamoDB Streams
    # support multiple event source mappings against one stream, each
    # with its own iterator position, so this Lambda seeing every write
    # does not affect what the error handler sees or vice versa.
    wire_stream_trigger(config.GRAPH_TRIGGER_FUNCTION, stream_arn)

    return {"classifier": classifier_arn, "error_handler": error_handler_arn,
           "graph_trigger": graph_trigger_arn}
