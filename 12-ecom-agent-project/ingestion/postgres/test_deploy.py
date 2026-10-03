"""Tests for deploy.py against moto's simulated AWS — no real account touched.

    first deploy      subnet group, security group with ONE /32 rule, encrypted public
                      instance with an RDS-managed secret, deployment.json without a password
    re-run            nothing duplicated
    new IP            a second /32 rule; the first is kept
    destroy           instance, subnet group, security group and deployment.json gone
    hand-off          a secret with only username + password, address from PG* variables,
                      connects the loader to a real PostgreSQL (skipped if none is running)

    python -m pytest test_deploy.py -q
"""
import json

import boto3
import psycopg
import pytest
from moto import mock_aws

import deploy
import load_postgres

REGION = "us-east-1"
BASE = ["--region", REGION, "--skip-load", "--allow-cidr", "203.0.113.7/32"]


@pytest.fixture(autouse=True)
def aws(tmp_path, monkeypatch):
    for k in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setattr(deploy, "DEPLOYMENT", tmp_path / "deployment.json")
    with mock_aws():
        yield


def rules(ec2):
    sg = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": ["ecom-nlq-db-sg"]}])["SecurityGroups"]
    return sg, sorted(r["CidrIp"] for p in sg[0]["IpPermissions"] for r in p["IpRanges"])


def test_first_deploy_creates_a_locked_down_public_instance():
    deploy.main(BASE)
    rds, ec2 = boto3.client("rds", REGION), boto3.client("ec2", REGION)
    db = rds.describe_db_instances(DBInstanceIdentifier="ecom-nlq-db")["DBInstances"][0]
    assert db["PubliclyAccessible"] and db["StorageEncrypted"] and db["Engine"] == "postgres"
    assert db["DBInstanceClass"] == "db.t4g.micro" and db["DBName"] == "ecom"
    assert db["MasterUserSecret"]["SecretArn"]                          # password managed by RDS
    sg, cidrs = rules(ec2)
    assert len(sg) == 1 and cidrs == ["203.0.113.7/32"]                 # one IP, never 0.0.0.0/0
    assert db["VpcSecurityGroups"][0]["VpcSecurityGroupId"] == sg[0]["GroupId"]
    info = json.loads(deploy.DEPLOYMENT.read_text(encoding="utf-8"))
    assert info["host"] == db["Endpoint"]["Address"] and info["secret_arn"] == db["MasterUserSecret"]["SecretArn"]
    secret = json.loads(boto3.client("secretsmanager", REGION).get_secret_value(SecretId=info["secret_arn"])["SecretString"])
    assert secret["password"] not in deploy.DEPLOYMENT.read_text(encoding="utf-8")     # the file never holds it


def test_rerun_duplicates_nothing():
    deploy.main(BASE)
    deploy.main(BASE)
    rds, ec2 = boto3.client("rds", REGION), boto3.client("ec2", REGION)
    assert len(rds.describe_db_instances()["DBInstances"]) == 1
    assert len(rds.describe_db_subnet_groups()["DBSubnetGroups"]) == 1
    sg, cidrs = rules(ec2)
    assert len(sg) == 1 and cidrs == ["203.0.113.7/32"]


def test_a_new_ip_is_added_and_the_old_kept():
    deploy.main(BASE)
    deploy.main(["--region", REGION, "--skip-load", "--allow-cidr", "198.51.100.20/32"])
    _, cidrs = rules(boto3.client("ec2", REGION))
    assert cidrs == ["198.51.100.20/32", "203.0.113.7/32"]


def test_destroy_removes_everything():
    deploy.main(BASE)
    deploy.main(["--region", REGION, "--destroy", "--yes"])
    rds, ec2 = boto3.client("rds", REGION), boto3.client("ec2", REGION)
    assert rds.describe_db_instances()["DBInstances"] == []
    assert rds.describe_db_subnet_groups()["DBSubnetGroups"] == []
    assert ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": ["ecom-nlq-db-sg"]}])["SecurityGroups"] == []
    assert not deploy.DEPLOYMENT.exists()


def test_load_needs_cleaned_data_first(tmp_path):
    with pytest.raises(SystemExit, match="run `python run.py prepare` first"):
        deploy.main(["--region", REGION, "--allow-cidr", "203.0.113.7/32", "--data", str(tmp_path)])


def local_postgres() -> bool:
    try:
        psycopg.connect(host="localhost", dbname="ecom", user="ecom_loader", password="localtest",
                        sslmode="disable", connect_timeout=3).close()
        return True
    except psycopg.OperationalError:
        return False


@pytest.mark.skipif(not local_postgres(), reason="no local PostgreSQL with ecom_loader")
def test_secret_with_only_username_and_password_connects(monkeypatch):
    """The RDS-managed secret shape: credentials from the secret, address from PG*."""
    arn = boto3.client("secretsmanager", REGION).create_secret(
        Name="rds!db-test", SecretString=json.dumps({"username": "ecom_loader", "password": "localtest"}))["ARN"]
    monkeypatch.setenv("PGHOST", "localhost")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "ecom")
    monkeypatch.setenv("PGSSLMODE", "disable")
    with load_postgres.connect(arn) as conn:
        assert conn.execute("SELECT current_user").fetchone()[0] == "ecom_loader"
    monkeypatch.delenv("PGHOST")
    with pytest.raises(SystemExit, match="no host and PGHOST is not set"):
        load_postgres.connect(arn)
