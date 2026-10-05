"""Deploy the NLQ agent to AgentCore — and everything it needs to reach RDS safely.

    STEP 0  preflight      AWS identity, Docker with linux/arm64; CloudWatch Transaction Search on
    STEP 1  database       ingestion/postgres/deployment.json, or the RDS instance ecom-nlq-db looked up in AWS
    STEP 2  reader user    ecom_reader: SELECT on schema ecom only; password -> ecom-nlq/db-reader
    STEP 3  API keys       ecom-agents/openai, ecom-agents/pinecone  (from ingestion/.env; shared by all agents)
    STEP 4  network        Lambda SG · RDS admits it on 5432 · Secrets Manager endpoint in the VPC
    STEP 5  SQL Lambda     ecom-nlq-sql in the VPC: handler + sql_exec + the agent's guard.py + psycopg
    STEP 6  Gateway        ecom-nlq-gateway (AWS_IAM / SigV4) -> target nlq-sql-tools
    STEP 7  runtime        image (linux/arm64) -> ECR -> AgentCore runtime ecom_nlq (A2A)
    STEP 8  deployment.json

    cd agent/nlq
    python deploy.py

    agent ──SigV4──► Gateway ──► Lambda (VPC) ──5432──► RDS (as ecom_reader)
      └── OpenAI, Pinecone directly (public runtime: no NAT gateway needed)

SAFE TO RE-RUN
    Every step looks before it creates. A re-run rebuilds the image and the
    Lambda code, and leaves everything else as it is.

WHAT THIS DOES NOT DO
    It does not create RDS (ingestion/postgres/deploy.py does) or Pinecone (the
    ingestion pipeline does). It does not deploy the supervisor or chart_gen.
"""
import json
import os
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RDS_DEPLOYMENT = PROJECT / "ingestion" / "postgres" / "deployment.json"
ENV_FILE = PROJECT / "ingestion" / ".env"
KEY_SECRETS = {"OPENAI_API_KEY": "ecom-agents/openai", "PINECONE_API_KEY": "ecom-agents/pinecone"}


def _env_file() -> None:
    """Keys from ingestion/.env; a variable already in the environment wins."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _key_secret(sm, env_var: str, name: str) -> str:
    """The secret {"api_key": …}, created or updated from the environment."""
    value = os.environ.get(env_var)
    try:
        current = json.loads(sm.get_secret_value(SecretId=name)["SecretString"]).get("api_key")
        arn = sm.describe_secret(SecretId=name)["ARN"]
        if value and value != current:
            sm.put_secret_value(SecretId=name, SecretString=json.dumps({"api_key": value}))
            print(f"  {name} updated from {env_var}")
        else:
            print(f"  {name} exists")
        return arn
    except sm.exceptions.ResourceNotFoundException:
        if not value:
            raise SystemExit(f"{env_var} is not set — add it to ingestion/.env")
        print(f"  {name} created from {env_var}")
        return sm.create_secret(Name=name, SecretString=json.dumps({"api_key": value}))["ARN"]


def _rds_info(session, region: str) -> dict:
    """The database to reach: ingestion's deployment.json when present, else the RDS
    instance looked up directly in AWS — the instance is the source of truth, the
    file only a record of it (a missing file must not block a deploy)."""
    if RDS_DEPLOYMENT.exists():
        return json.loads(RDS_DEPLOYMENT.read_text(encoding="utf-8"))
    instance = os.environ.get("RDS_INSTANCE", "ecom-nlq-db")
    rds = session.client("rds")
    try:
        db = rds.describe_db_instances(DBInstanceIdentifier=instance)["DBInstances"][0]
    except rds.exceptions.DBInstanceNotFoundFault:
        raise SystemExit(f"no RDS instance {instance!r} and no {RDS_DEPLOYMENT} — run `python run.py deploy` in ingestion/ first")
    if not db.get("MasterUserSecret"):
        raise SystemExit(f"{instance} has no RDS-managed master secret — deploy it with ingestion/postgres/deploy.py")
    print(f"  {RDS_DEPLOYMENT.name} not found — using the RDS instance {instance!r} found in AWS")
    return {"instance": db["DBInstanceIdentifier"], "region": region, "host": db["Endpoint"]["Address"],
            "port": db["Endpoint"]["Port"], "database": db.get("DBName", "postgres"), "master_user": db["MasterUsername"],
            "secret_arn": db["MasterUserSecret"]["SecretArn"],
            "security_group": db["VpcSecurityGroups"][0]["VpcSecurityGroupId"]}


def main() -> None:
    _env_file()
    session = boto3.session.Session()
    region = session.region_name or os.environ.get("AWS_REGION", "us-east-1")

    from infra import preflight, observability
    print("=== STEP 0: preflight ===")
    preflight.run(needs_gateway=True, session=session)
    observability.ensure_transaction_search()

    print("\n=== STEP 1: database ===")
    rds = _rds_info(session, region)
    print(f"  {rds['instance']}: {rds['host']}:{rds['port']}/{rds['database']}")
    rds_sg = rds["security_group"]

    from infra import reader
    print("\n=== STEP 2: read-only database user ===")
    reader_arn = reader.ensure_reader(rds)

    print("\n=== STEP 3: API key secrets ===")
    sm = session.client("secretsmanager")
    key_arns = {env: _key_secret(sm, env, name) for env, name in KEY_SECRETS.items()}

    from infra import network
    print("\n=== STEP 4: network (VPC) ===")
    net = network.ensure(rds_sg, region)

    from infra import iam, lambda_deploy
    print("\n=== STEP 5: SQL Lambda ===")
    lambda_role = iam.lambda_role([reader_arn])          # ARN prefix: IAM resources must be ARNs
    lambda_arn = lambda_deploy.deploy(lambda_role, {
        "PGHOST": rds["host"], "PGPORT": str(rds["port"]), "PGDATABASE": rds["database"],
        "PGUSER": reader.READER, "PGSSLMODE": "require", "READER_SECRET_ID": reader_arn},
        net["subnet_ids"], net["lambda_sg"])

    from infra import gateway
    print("\n=== STEP 6: Gateway ===")
    gw = gateway.create_gateway(iam.gateway_role(lambda_arn))
    gateway.create_lambda_target(gw["gatewayId"], lambda_arn)
    lambda_deploy.allow_gateway_invoke(gw["gatewayArn"])

    from infra import runtime_deploy, runtime_iam
    print("\n=== STEP 7: runtime ===")
    repo_uri, repo_arn = runtime_deploy.ensure_ecr_repo()
    runtime_deploy.ecr_login()
    image = runtime_deploy.build_and_push(repo_uri, str(HERE / "agent_code"))
    role = runtime_iam.runtime_role(gateway_arn=gw["gatewayArn"], ecr_repo_arn=repo_arn,
                                    secret_arns=list(key_arns.values()))
    runtime_arn = runtime_deploy.deploy_runtime(image, role, {
        "AWS_REGION": region, "NLQ_MODEL": os.environ.get("NLQ_MODEL", "gpt-6-sol"), "NLQ_DB_MODE": "gateway",
        "NLQ_GATEWAY_URL": gw["gatewayUrl"], "NLQ_GATEWAY_TARGET": gateway.TARGET_NAME,
        "OPENAI_SECRET_ID": key_arns["OPENAI_API_KEY"], "PINECONE_SECRET_ID": key_arns["PINECONE_API_KEY"],
        "PINECONE_INDEX": os.environ.get("PINECONE_INDEX", "ecom-kb")})

    print("\n=== STEP 8: deployment.json ===")
    info = {"agent": "nlq", "runtime_arn": runtime_arn, "region": region, "gateway_url": gw["gatewayUrl"],
            "lambda": lambda_deploy.FUNCTION_NAME, "reader_secret": reader.SECRET_NAME}
    (HERE / "deployment.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"\ndone. NLQ runtime: {runtime_arn}")


if __name__ == "__main__":
    main()
