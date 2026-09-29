"""The supervisor's memory store: one DynamoDB table and one Pinecone index.

    ensure_table()          trial-agents-memory      PK user_id, SK memory_id
                            on-demand; created once, found after
    ensure_index(api_key)   trial-agents-memory      1536 dims, cosine,
                            serverless aws us-east-1; namespaces "semantic"
                            and "episodic" appear on first write

WHY PINECONE IS CALLED OVER REST

The deploy runs on a laptop that has boto3 but not necessarily the pinecone
package. Two HTTPS calls with the standard library (describe, create) keep
the deployer's dependencies unchanged. The supervisor container, which
reads and writes vectors, does use the pinecone package.

WHY THE CERTIFICATE BUNDLE IS PASSED EXPLICITLY

urllib verifies HTTPS against the operating system's CA bundle. The
python.org installer for macOS ships with NONE until its "Install
Certificates.command" is run, so every HTTPS call fails with
CERTIFICATE_VERIFY_FAILED — while boto3 works, because it carries its own
bundle. This module uses certifi's bundle when installed, else botocore's
own cacert.pem (always present where boto3 is): the deploy then works the
same on every Python install.

WHY 1536 / COSINE

Memories are embedded with text-embedding-3-small, the same model as the
protocol index: 1536 dimensions. Cosine matches how the recall threshold
(0.90 = the same fact restated) was chosen.

WHAT THIS DOES NOT DO

    It never deletes or recreates an existing table or index — memories
    already stored survive every redeploy.
"""
import json
import ssl
import time
import urllib.error
import urllib.request

import boto3

TABLE = "trial-agents-memory"
INDEX = "trial-agents-memory"
DIMENSION = 1536
PINECONE_API = "https://api.pinecone.io"


def ca_bundle() -> str:
    """certifi's bundle if installed, else the one botocore ships."""
    try:
        import certifi
        return certifi.where()
    except ImportError:
        from botocore.httpsession import DEFAULT_CA_BUNDLE
        return DEFAULT_CA_BUNDLE


_TLS = ssl.create_default_context(cafile=ca_bundle())


def ensure_table(region: str) -> str:
    """The memory table's ARN; created on first deploy."""
    ddb = boto3.client("dynamodb", region_name=region)
    try:
        return ddb.describe_table(TableName=TABLE)["Table"]["TableArn"]
    except ddb.exceptions.ResourceNotFoundException:
        pass
    print(f"  creating DynamoDB table {TABLE!r}")
    ddb.create_table(
        TableName=TABLE, BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"},
                              {"AttributeName": "memory_id", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"},
                   {"AttributeName": "memory_id", "KeyType": "RANGE"}])
    ddb.get_waiter("table_exists").wait(TableName=TABLE)
    return ddb.describe_table(TableName=TABLE)["Table"]["TableArn"]


def _pinecone(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        PINECONE_API + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Api-Key": api_key, "Content-Type": "application/json",
                 "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30, context=_TLS) as response:
        return json.loads(response.read() or b"{}")


def ensure_index(api_key: str, region: str = "us-east-1") -> str:
    """The index name, ready to use; created on first deploy."""
    try:
        described = _pinecone("GET", f"/indexes/{INDEX}", api_key)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        print(f"  creating Pinecone index {INDEX!r} ({DIMENSION} dims, cosine)")
        _pinecone("POST", "/indexes", api_key, {
            "name": INDEX, "dimension": DIMENSION, "metric": "cosine",
            "spec": {"serverless": {"cloud": "aws", "region": region}}})
        described = {}
    for _ in range(60):                             # a new index takes ~10-60 s
        if (described.get("status") or {}).get("ready"):
            print(f"  Pinecone index {INDEX!r} ready")
            return INDEX
        time.sleep(5)
        described = _pinecone("GET", f"/indexes/{INDEX}", api_key)
    raise RuntimeError(f"Pinecone index {INDEX} not ready after 5 minutes")
