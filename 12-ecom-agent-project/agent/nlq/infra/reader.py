"""The read-only database user the agent's queries run as.

    your laptop (its IP is allowed on RDS) ──as ecom_admin (RDS-managed secret)──► RDS
        CREATE / ALTER ROLE ecom_reader LOGIN PASSWORD <generated>
        GRANT USAGE ON SCHEMA ecom · SELECT ON ALL TABLES · DEFAULT PRIVILEGES for future tables
        ALTER ROLE ecom_reader SET default_transaction_read_only = on
    Secrets Manager  ecom-nlq/db-reader = {"username": "ecom_reader", "password": …}

WHAT THIS DOES NOT DO
    It grants nothing but SELECT on schema ecom: the reader cannot write, cannot
    create objects, and cannot see other schemas. The password is generated
    here, stored only in Secrets Manager, and never printed.
"""
import json
import secrets as pysecrets
import string

import boto3
import psycopg
from psycopg import sql

READER = "ecom_reader"
SECRET_NAME = "ecom-nlq/db-reader"


def _admin_password(region: str, secret_arn: str) -> dict:
    return json.loads(boto3.client("secretsmanager", region_name=region)
                      .get_secret_value(SecretId=secret_arn)["SecretString"])


def ensure_reader(deployment: dict) -> str:
    """Create or refresh ecom_reader; returns the reader secret's ARN."""
    region = deployment["region"]
    sm = boto3.client("secretsmanager", region_name=region)
    try:
        password = json.loads(sm.get_secret_value(SecretId=SECRET_NAME)["SecretString"])["password"]
        arn = sm.describe_secret(SecretId=SECRET_NAME)["ARN"]
    except sm.exceptions.ResourceNotFoundException:
        password = "".join(pysecrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        arn = sm.create_secret(Name=SECRET_NAME, Description="read-only DB user for the ecom NLQ agent",
                               SecretString=json.dumps({"username": READER, "password": password}))["ARN"]
        print(f"  secret {SECRET_NAME} created")
    admin = _admin_password(region, deployment["secret_arn"])
    with psycopg.connect(host=deployment["host"], port=deployment["port"], dbname=deployment["database"],
                         user=admin["username"], password=admin["password"], sslmode="require",
                         connect_timeout=15, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (READER,)).fetchone()
        verb = "ALTER" if exists else "CREATE"
        conn.execute(sql.SQL(f"{verb} ROLE {{}} LOGIN PASSWORD {{}}").format(sql.Identifier(READER), sql.Literal(password)))
        for stmt in ("GRANT USAGE ON SCHEMA ecom TO {r}",
                     "GRANT SELECT ON ALL TABLES IN SCHEMA ecom TO {r}",
                     "ALTER DEFAULT PRIVILEGES IN SCHEMA ecom GRANT SELECT ON TABLES TO {r}",
                     "ALTER ROLE {r} SET default_transaction_read_only = on"):
            conn.execute(sql.SQL(stmt.replace("{r}", "{}")).format(sql.Identifier(READER)))
    print(f"  {READER} {'refreshed' if exists else 'created'}: SELECT on schema ecom only, read-only by default")
    return arn
