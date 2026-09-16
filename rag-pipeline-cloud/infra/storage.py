"""S3 bucket and DynamoDB audit table.

The bucket is created EMPTY — no prefixes pre-populated. The prefixes in
config.py (raw/, docs/, cache/, registry/, error/) come into existence the
first time something is written to them, which is how S3 works; there is
nothing to provision for a "folder".

The S3 -> Lambda trigger is the one piece that makes the pipeline self-
starting: upload a PDF under raw/, and the classifier runs with no other
action needed. Wiring it requires the Lambda's resource policy to grant s3
permission to invoke it, done in lambdas.py, and the bucket notification
configuration, done here — both have to exist for the trigger to fire, and
getting only one of them is a common way this silently does nothing.
"""

import boto3
from botocore.exceptions import ClientError

from . import config

s3 = boto3.client("s3", region_name=config.REGION)
ddb = boto3.client("dynamodb", region_name=config.REGION)


def exists(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except ClientError:
        return False


def create_bucket(bucket: str) -> None:
    """Create the document bucket, empty, with public access blocked.

    us-east-1 is the one region where CreateBucket must NOT be given a
    location constraint — passing one there is an error, not a no-op.
    """
    print(f"-- S3 bucket {bucket} --")
    if exists(s3.head_bucket, Bucket=bucket):
        print("   exists")
        return
    kwargs = {"Bucket": bucket}
    if config.REGION != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {
            "LocationConstraint": config.REGION}
    s3.create_bucket(**kwargs)
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True},
    )
    # Versioning protects against a re-upload of the same filename silently
    # replacing a document that a run is mid-way through reading.
    s3.put_bucket_versioning(
        Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
    print("   created, empty, public access blocked")


def wire_upload_trigger(bucket: str, classifier_function_arn: str) -> None:
    """Configure the bucket to invoke the classifier when a PDF lands under
    raw/.

    Only the notification CONFIGURATION lives here. The matching resource-
    based permission that lets S3 actually invoke the function is added in
    lambdas.py, on the Lambda side — both halves are required, and this is a
    known way for "the trigger does nothing" to happen silently, because
    neither half errors when the other is missing.

    put_bucket_notification_configuration REPLACES the whole configuration
    rather than adding to it, so if this bucket ever needs a second trigger,
    both must be included in the same call.
    """
    print("-- S3 event notification: raw/*.pdf -> classifier --")
    s3.put_bucket_notification_configuration(
        Bucket=bucket,
        NotificationConfiguration={
            "LambdaFunctionConfigurations": [{
                "LambdaFunctionArn": classifier_function_arn,
                "Events": ["s3:ObjectCreated:*"],
                "Filter": {"Key": {"FilterRules": [
                    {"Name": "prefix", "Value": f"{config.RAW_PREFIX}/"},
                    {"Name": "suffix", "Value": ".pdf"},
                ]}},
            }],
        },
    )
    print("   wired")


def create_audit_table() -> str:
    """The per-document, per-stage audit table, with a stream for the
    error-handling Lambda to react to.

    Two record shapes in one table, same design as the earlier Fargate
    version of this project:

        pk = DOC#<doc_id>   sk = RUN#<run_id>                  one per run
        pk = DOC#<doc_id>   sk = RUN#<run_id>#STAGE#<stage>    one per stage

    NEW_AND_OLD_IMAGES on the stream, not NEW_IMAGE alone, because the error
    handler needs to distinguish "this write just SET status to FAILED" from
    "this record has always been FAILED" — without the old image both look
    identical and a status write during retry re-triggers the error path.
    """
    print(f"-- DynamoDB table {config.AUDIT_TABLE} --")
    if exists(ddb.describe_table, TableName=config.AUDIT_TABLE):
        print("   exists")
    else:
        ddb.create_table(
            TableName=config.AUDIT_TABLE,
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"},
                      {"AttributeName": "sk", "KeyType": "RANGE"}],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
            StreamSpecification={"StreamEnabled": True,
                                 "StreamViewType": "NEW_AND_OLD_IMAGES"},
        )
        waiter = ddb.get_waiter("table_exists")
        waiter.wait(TableName=config.AUDIT_TABLE)
        print("   created, with a stream")

    return ddb.describe_table(TableName=config.AUDIT_TABLE)["Table"] \
              .get("LatestStreamArn", "")
