"""Create every AWS resource this pipeline needs.

    python aws/setup.py

Safe to re-run. Every step checks whether the resource already exists first, because
setup fails part-way often — a permissions gap, an expired session, a quota — and
re-running has to pick up where it stopped rather than erroring on what already
exists.

boto3 rather than CloudFormation, deliberately. A template is the better production
answer, but it is 600 lines of YAML that nobody reads. This is the same resources in
a script you can follow top to bottom, with teardown.py as the counterpart.
"""

import json
import time

import boto3
from botocore.exceptions import ClientError

import config

s3 = boto3.client("s3", region_name=config.REGION)
ddb = boto3.client("dynamodb", region_name=config.REGION)
ecr = boto3.client("ecr", region_name=config.REGION)
ecs = boto3.client("ecs", region_name=config.REGION)
iam = boto3.client("iam", region_name=config.REGION)
logs = boto3.client("logs", region_name=config.REGION)
sfn = boto3.client("stepfunctions", region_name=config.REGION)
sm = boto3.client("secretsmanager", region_name=config.REGION)
ec2 = boto3.client("ec2", region_name=config.REGION)

ACCOUNT = boto3.client("sts", region_name=config.REGION).get_caller_identity()["Account"]


def step(message: str) -> None:
    """Print a section header so a long setup run reads as a checklist."""
    print(f"\n── {message}")


def exists(fn, *args, **kwargs) -> bool:
    """True if an AWS describe/head call succeeds.

    Every creation step is guarded by one of these so the whole script is
    idempotent. Setup fails part-way often — a permissions gap, an expired session,
    a quota — and re-running has to pick up where it stopped rather than erroring on
    the resources that already exist.
    """
    """True if an AWS describe/head call succeeds, False on any client error."""
    try:
        fn(*args, **kwargs)
        return True
    except ClientError:
        return False


# ── Storage and state ────────────────────────────────────────────────────────
def create_bucket() -> None:
    """Create the document bucket, blocking all public access.

    us-east-1 is the one region where CreateBucket must NOT be given a location
    constraint; passing one there is an error. Public access is blocked explicitly
    rather than relying on account defaults, because the bucket holds source
    documents.
    """
    step(f"S3 bucket {config.BUCKET}")
    if exists(s3.head_bucket, Bucket=config.BUCKET):
        print("   exists")
        return
    kwargs = {"Bucket": config.BUCKET}
    # us-east-1 is the one region where CreateBucket must NOT carry a location
    # constraint. Passing one there is an InvalidLocationConstraint error.
    if config.REGION != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": config.REGION}
    s3.create_bucket(**kwargs)
    s3.put_public_access_block(
        Bucket=config.BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True},
    )
    print("   created")


def create_audit_table() -> None:
    """Create the audit table.

    Keyed pk=DOC#<doc_id>, sk=RUN#<run> or RUN#<run>#STAGE#<stage>, so one query
    returns a document's entire history in order. The GSI on status is what makes
    "show me everything that failed today" a query rather than a full scan.

    On-demand billing: ingestion writes in short bursts separated by hours, which is
    the worst possible shape for provisioned capacity.
    """
    step(f"DynamoDB table {config.AUDIT_TABLE}")
    if exists(ddb.describe_table, TableName=config.AUDIT_TABLE):
        print("   exists")
        return
    # pk = DOC#<doc_id>, sk = RUN#<run> or RUN#<run>#STAGE#<stage>. Keying on the
    # document means one query returns its whole history in order, which is what
    # makes `status.py --doc` a single call rather than a scan.
    #
    # On-demand billing: ingestion writes in short bursts separated by hours, which
    # is the worst possible shape for provisioned capacity.
    ddb.create_table(
        TableName=config.AUDIT_TABLE,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"},
                   {"AttributeName": "sk", "KeyType": "RANGE"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
                              {"AttributeName": "sk", "AttributeType": "S"},
                              {"AttributeName": "status", "AttributeType": "S"},
                              {"AttributeName": "ts", "AttributeType": "S"}],
        # Without this index, "what failed today" means scanning every record in
        # the table — slow, and billed per item read.
        GlobalSecondaryIndexes=[{
            "IndexName": "status-ts-index",
            "KeySchema": [{"AttributeName": "status", "KeyType": "HASH"},
                          {"AttributeName": "ts", "KeyType": "RANGE"}],
            "Projection": {"ProjectionType": "ALL"},
        }],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.get_waiter("table_exists").wait(TableName=config.AUDIT_TABLE)
    print("   created")


def create_secret() -> None:
    """Create the API-key secret with placeholder values.

    Placeholders rather than prompting, so setup stays non-interactive and can run
    in CI. preflight.py checks that the placeholders were replaced before a run can
    start, which is where the real guard lives.
    """
    step(f"Secrets Manager {config.SECRET_NAME}")
    if exists(sm.describe_secret, SecretId=config.SECRET_NAME):
        print("   exists (update it with the AWS console or CLI if keys changed)")
        return
    # Placeholders rather than prompting, so setup stays non-interactive and can
    # run in CI. preflight.py is what enforces that they were replaced.
    sm.create_secret(
        Name=config.SECRET_NAME,
        SecretString=json.dumps({"OPENAI_API_KEY": "replace-me",
                                 "PINECONE_API_KEY": "replace-me"}),
    )
    print("   created — set the real keys before running:")
    print(f'   aws secretsmanager put-secret-value --secret-id {config.SECRET_NAME} \\')
    print('     --secret-string \'{"OPENAI_API_KEY":"sk-…","PINECONE_API_KEY":"pc-…"}\'')


def create_log_group() -> None:
    """Create the log group with a retention policy.

    Log groups default to never expiring, which quietly accumulates cost forever.
    Thirty days outlives any debugging session.
    """
    step(f"CloudWatch log group {config.LOG_GROUP}")
    try:
        logs.create_log_group(logGroupName=config.LOG_GROUP)
        # Log groups default to never expiring, which accumulates cost silently and
        # forever. Thirty days outlives any debugging session.
        logs.put_retention_policy(logGroupName=config.LOG_GROUP, retentionInDays=30)
        print("   created")
    except logs.exceptions.ResourceAlreadyExistsException:
        print("   exists")


def create_ecr_repo() -> str:
    """Create the container repository and return its URI.

    The URI is derived from account and region rather than read back, so it is
    available even when the repository already existed.
    """
    step(f"ECR repository {config.ECR_REPO}")
    try:
        ecr.create_repository(repositoryName=config.ECR_REPO)
        print("   created")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print("   exists")
    # Derived rather than read back, so it resolves whether or not the repository
    # was just created.
    uri = f"{ACCOUNT}.dkr.ecr.{config.REGION}.amazonaws.com/{config.ECR_REPO}"
    print(f"   {uri}")
    return uri


# ── IAM ──────────────────────────────────────────────────────────────────────
def upsert_role(name: str, service: str, policy: dict, managed: list[str] = ()) -> str:
    """Create or update one IAM role with its trust policy and permissions.

    put_role_policy overwrites by name, so re-running applies changed permissions to
    an existing role rather than failing or accumulating duplicates.
    """
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]}
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust))
        print(f"   created {name}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"   exists {name}")
    for arn in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=arn)
    # put_role_policy overwrites by name, so re-running setup applies changed
    # permissions rather than failing or accumulating duplicate policies.
    iam.put_role_policy(RoleName=name, PolicyName=f"{name}-inline",
                        PolicyDocument=json.dumps(policy))
    return f"arn:aws:iam::{ACCOUNT}:role/{name}"


def create_roles() -> tuple[str, str, str]:
    """Create the three roles this pipeline needs.

    The split between execution and task role is the part worth understanding. The
    execution role belongs to the ECS agent: it pulls the image, writes logs, and
    injects secrets into the container environment before your code starts. The task
    role is what the running code itself assumes. Code that needs S3 gets it through
    the task role; it never needs the secrets permission, because the agent has
    already placed those values in the environment.
    """
    step("IAM roles")

    exec_arn = upsert_role(
        config.EXEC_ROLE, "ecs-tasks.amazonaws.com",
        # The execution role is used by the ECS agent, not your code: it pulls the
        # image, writes logs, and injects secrets into the container environment.
        {"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"],
            "Resource": f"arn:aws:secretsmanager:{config.REGION}:{ACCOUNT}:secret:{config.SECRET_NAME}-*"}]},
        managed=["arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"],
    )

    task_arn = upsert_role(
        config.TASK_ROLE, "ecs-tasks.amazonaws.com",
        # The task role is what ingest.py itself uses.
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow",
             "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
             "Resource": [f"arn:aws:s3:::{config.BUCKET}",
                          f"arn:aws:s3:::{config.BUCKET}/*"]},
            {"Effect": "Allow",
             "Action": ["dynamodb:PutItem", "dynamodb:Query", "dynamodb:GetItem"],
             "Resource": [f"arn:aws:dynamodb:{config.REGION}:{ACCOUNT}:table/{config.AUDIT_TABLE}",
                          f"arn:aws:dynamodb:{config.REGION}:{ACCOUNT}:table/{config.AUDIT_TABLE}/index/*"]},
        ]},
    )

    sfn_arn = upsert_role(
        config.SFN_ROLE, "states.amazonaws.com",
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["ecs:RunTask", "ecs:StopTask",
                                           "ecs:DescribeTasks"], "Resource": "*"},
            # Step Functions launches tasks that assume these roles, so it needs
            # permission to hand them over. Scoped to these two ARNs rather than "*".
            {"Effect": "Allow", "Action": "iam:PassRole",
             "Resource": [exec_arn, task_arn]},
            # RunTask.sync needs these to receive task-completion events.
            {"Effect": "Allow",
             "Action": ["events:PutTargets", "events:PutRule", "events:DescribeRule"],
             "Resource": "*"},
            {"Effect": "Allow", "Action": ["s3:GetObject"],
             "Resource": f"arn:aws:s3:::{config.BUCKET}/*"},
        ]},
    )
    return exec_arn, task_arn, sfn_arn


# ── Compute ──────────────────────────────────────────────────────────────────
def default_network() -> tuple[list[str], str]:
    """Use the account's default VPC, so no networking has to be created.

    A purpose-built VPC with private subnets and a NAT gateway is the production
    answer, but a NAT gateway costs more per month than everything else here
    combined. The default VPC with public IPs is the right trade for a pipeline that
    only makes outbound calls.
    """
    """Use the account's default VPC so no networking has to be created."""
    # A purpose-built VPC with private subnets is the production answer, but a NAT
    # gateway costs more per month than everything else here combined. The default
    # VPC with public IPs is the right trade for a pipeline that only calls outward.
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise SystemExit("no default VPC in this region; create one or supply subnet ids")
    vpc_id = vpcs[0]["VpcId"]
    subnets = [s["SubnetId"] for s in ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]]
    groups = ec2.describe_security_groups(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]},
                 {"Name": "group-name", "Values": ["default"]}])["SecurityGroups"]
    return subnets, groups[0]["GroupId"]


def create_cluster() -> None:
    """Create the ECS cluster. Idempotent — create_cluster returns the existing one."""
    step(f"ECS cluster {config.CLUSTER}")
    ecs.create_cluster(clusterName=config.CLUSTER, capacityProviders=["FARGATE"])
    print("   ready")


def register_task_definitions(image_uri: str, exec_arn: str, task_arn: str) -> dict:
    """Register one task definition per tier: same image, different CPU and memory.

    Secrets are declared rather than passed as environment values, so the keys never
    appear in the task definition, in the console, or in CloudTrail. The agent
    resolves them at launch.

    Registering always creates a new revision rather than mutating, so re-running
    after a config change is safe and the previous revision stays available.
    """
    step("ECS task definitions")
    secret_arn = sm.describe_secret(SecretId=config.SECRET_NAME)["ARN"]
    arns = {}
    for tier, spec in config.TIERS.items():
        family = f"{config.PROJECT}-{tier}"
        response = ecs.register_task_definition(
            family=family,
            requiresCompatibilities=["FARGATE"],
            networkMode="awsvpc",
            cpu=spec["cpu"], memory=spec["memory"],
            executionRoleArn=exec_arn, taskRoleArn=task_arn,
            runtimePlatform={"cpuArchitecture": "X86_64",
                             "operatingSystemFamily": "LINUX"},
            containerDefinitions=[{
                "name": "ingest",
                "image": f"{image_uri}:latest",
                "essential": True,
                "environment": [
                    {"name": "AWS_REGION", "value": config.REGION},
                    {"name": "AUDIT_TABLE", "value": config.AUDIT_TABLE},
                ],
                # Declared as secrets, not environment values, so the keys never
                # appear in the task definition, the console, or CloudTrail. The ECS
                # agent resolves them into the container at launch.
                "secrets": [
                    {"name": "OPENAI_API_KEY",
                     "valueFrom": f"{secret_arn}:OPENAI_API_KEY::"},
                    {"name": "PINECONE_API_KEY",
                     "valueFrom": f"{secret_arn}:PINECONE_API_KEY::"},
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {"awslogs-group": config.LOG_GROUP,
                                "awslogs-region": config.REGION,
                                "awslogs-stream-prefix": tier},
                },
            }],
        )
        arns[tier] = response["taskDefinition"]["taskDefinitionArn"]
        print(f"   {family}: {spec['cpu']} cpu / {spec['memory']} MB")
    return arns


# ── Orchestration ────────────────────────────────────────────────────────────
def lane(tier: str, task_def_arn: str, subnets: list[str], sg: str) -> dict:
    """One Map state: run every document of this tier, N at a time.

    MaxConcurrency is the whole parallelism model. RunTask.sync makes Step Functions
    wait for the container to exit and surface its exit code.
    """
    spec = config.TIERS[tier]
    return {
        "Type": "Map",
        "ItemsPath": f"$.{tier}",
        # This single field is the entire parallelism model. There is no queue, no
        # worker pool, no scheduler to configure.
        "MaxConcurrency": spec["concurrency"],
        "ResultPath": None,
        "Iterator": {
            "StartAt": "RunIngest",
            "States": {
                "RunIngest": {
                    "Type": "Task",
                    # .sync blocks until the container exits and reads its exit
                    # code. Without it the state machine fires and forgets, reports
                    # success immediately, and you get parallelism with no idea
                    # whether anything worked.
                    "Resource": "arn:aws:states:::ecs:runTask.sync",
                    "TimeoutSeconds": config.TASK_TIMEOUT_SECONDS,
                    "Parameters": {
                        "Cluster": config.CLUSTER,
                        "TaskDefinition": task_def_arn,
                        "LaunchType": "FARGATE",
                        "NetworkConfiguration": {"AwsvpcConfiguration": {
                            "Subnets": subnets, "SecurityGroups": [sg],
                            "AssignPublicIp": "ENABLED"}},
                        "Overrides": {"ContainerOverrides": [{
                            "Name": "ingest",
                            # Command is an array, so each element cannot carry its own
                            # ".$" suffix. States.Array builds the whole list in one
                            # intrinsic expression with the item fields substituted in.
                            "Command.$": (
                                "States.Array('python','ingest.py',"
                                "'--bucket',$.bucket,'--key',$.key)"
                            ),
                        }]},
                    },
                    "Retry": [
                        # Container OOM or a transient AWS error. Two attempts only:
                        # each retry can cost an hour of compute, and re-ingestion is
                        # idempotent so a partial run leaves nothing to clean up.
                        {"ErrorEquals": ["States.TaskFailed", "States.Timeout"],
                         "IntervalSeconds": 30, "MaxAttempts": 1, "BackoffRate": 2.0},
                    ],
                    "Catch": [
                        # One bad document must not stop the other nineteen.
                        {"ErrorEquals": ["States.ALL"], "Next": "RecordFailure"},
                    ],
                    "End": True,
                },
                "RecordFailure": {"Type": "Pass", "End": True},
            },
        },
        "End": True,
    }


def create_state_machine(task_defs: dict, sfn_arn: str,
                         subnets: list[str], sg: str) -> str:
    """Create or update the state machine.

    Update rather than fail when it exists, so changing lane concurrency is a matter
    of editing config.py and re-running setup.
    """
    step(f"Step Functions {config.STATE_MACHINE}")
    definition = {
        "Comment": "Ingest PDFs in three size lanes, in parallel",
        "StartAt": "ProcessAllTiers",
        "States": {
            "ProcessAllTiers": {
                "Type": "Parallel",
                "Branches": [
                    {"StartAt": f"{tier}Lane",
                     "States": {f"{tier}Lane": lane(tier, task_defs[tier], subnets, sg)}}
                    for tier in config.TIERS
                ],
                "End": True,
            }
        },
    }
    body = json.dumps(definition)
    arn = f"arn:aws:states:{config.REGION}:{ACCOUNT}:stateMachine:{config.STATE_MACHINE}"
    try:
        sfn.create_state_machine(name=config.STATE_MACHINE, definition=body,
                                 roleArn=sfn_arn, type="STANDARD")
        print("   created")
    except sfn.exceptions.StateMachineAlreadyExists:
        sfn.update_state_machine(stateMachineArn=arn, definition=body, roleArn=sfn_arn)
        print("   updated")
    return arn


def main() -> None:
    """Create every resource, in dependency order, and print what to do next."""
    print(f"project {config.PROJECT} · region {config.REGION} · account {ACCOUNT}")
    create_bucket()
    create_audit_table()
    create_secret()
    create_log_group()
    image_uri = create_ecr_repo()
    exec_arn, task_arn, sfn_arn = create_roles()
    create_cluster()
    subnets, sg = default_network()
    print(f"   using default VPC: {len(subnets)} subnets")

    # IAM is eventually consistent across regions. A role used within a second or
    # two of creation is frequently rejected with a misleading "not authorized to
    # perform: iam:PassRole", which sends people to check policies that are already
    # correct. Ten seconds is cheaper than that debugging session.
    time.sleep(10)

    task_defs = register_task_definitions(image_uri, exec_arn, task_arn)
    machine_arn = create_state_machine(task_defs, sfn_arn, subnets, sg)

    print("\nsetup complete\n")
    print("next:")
    print(f"  1. put real API keys in {config.SECRET_NAME}")
    print(f"  2. bash build_and_push.sh              build and push the image")
    print(f"  3. python preflight.py                 check quotas before spending")
    print(f"  4. python upload_pdfs.py --dir ./pdfs")
    print(f"  5. python run.py")
    print(f"\nstate machine: {machine_arn}")


if __name__ == "__main__":
    main()
