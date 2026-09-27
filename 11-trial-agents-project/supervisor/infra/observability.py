"""Turn on CloudWatch Transaction Search — once per account and region.

    ensure_transaction_search()
        │
        ├─ X-Ray already sends segments to CloudWatchLogs?  -> done, nothing changes
        │
        ├─ logs:PutResourcePolicy   let X-Ray write spans into
        │                            aws/spans and /aws/application-signals/data
        └─ xray:UpdateTraceSegmentDestination -> CloudWatchLogs

WHY THIS IS IN THE DEPLOY

It is the silent-failure gate of AgentCore observability. The agents run under
opentelemetry-instrument and export spans either way; without Transaction
Search, none of them appear in CloudWatch or the GenAI Observability dashboard,
and nothing reports an error. A manual console step in a README is exactly the
step that gets skipped.

THE POLICY

As documented for AgentCore observability. AWS's own pages print the
aws:SourceArn condition two ways (logs:… and xray:…); X-Ray is the service
that writes the spans, and the CloudWatch CloudFormation guide uses xray:…,
so that is used here.

WHAT THIS DOES NOT DO

    It does not change the indexing (sampling) rule — the account default
    applies. It never turns Transaction Search off. Spans can take about ten
    minutes to become searchable after it is first enabled.
"""
import json

import boto3

POLICY_NAME = "TransactionSearchXRayAccess"


def ensure_transaction_search() -> str:
    session = boto3.Session()
    region = session.region_name or "us-east-1"
    account = session.client("sts").get_caller_identity()["Account"]
    xray = session.client("xray", region_name=region)

    # STEP 1 — already on? Then leave everything as it is.
    current = xray.get_trace_segment_destination()
    if current.get("Destination") == "CloudWatchLogs":
        print(f"  Transaction Search already enabled ({current.get('Status', 'ACTIVE')})")
        return "already-enabled"

    # STEP 2 — let X-Ray write spans into CloudWatch Logs.
    logs = session.client("logs", region_name=region)
    base = f"arn:aws:logs:{region}:{account}:log-group"
    logs.put_resource_policy(policyName=POLICY_NAME, policyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "TransactionSearchXRayAccess", "Effect": "Allow",
            "Principal": {"Service": "xray.amazonaws.com"},
            "Action": "logs:PutLogEvents",
            "Resource": [f"{base}:aws/spans:*", f"{base}:/aws/application-signals/data:*"],
            "Condition": {"ArnLike": {"aws:SourceArn": f"arn:aws:xray:{region}:{account}:*"},
                          "StringEquals": {"aws:SourceAccount": account}}}]}))

    # STEP 3 — send trace segments there.
    xray.update_trace_segment_destination(Destination="CloudWatchLogs")
    print("  Transaction Search enabled — spans may take ~10 minutes to appear")
    return "enabled"
