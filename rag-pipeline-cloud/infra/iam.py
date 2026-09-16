"""The IAM roles this pipeline needs.

Four roles, not one, and the split is the part worth understanding.

    execution role      the ECS agent uses this. Pulls the image, writes logs,
                         resolves Secrets Manager and Parameter Store values
                         into the container's environment BEFORE your code
                         starts.
    task role            what the running code itself assumes. Needs S3 and
                         DynamoDB. Never needs secretsmanager:GetSecretValue —
                         the agent already placed those values in the
                         environment by the time your code runs.
    classifier role      the Lambda that reads page count and submits Batch
                         jobs.
    error handler role   the Lambda the DynamoDB Stream invokes to move a
                         failed document to error/.

Conflating execution and task is the single most common IAM mistake in ECS/
Batch setups, because both "run inside the same task" and it is tempting to
give one role everything. Keeping them separate means a compromised container
process — the task role — never has permission to read the secret directly,
only to use values already injected.
"""

import json

import boto3
from botocore.exceptions import ClientError

from . import config

iam = boto3.client("iam", region_name=config.REGION)


def exists(fn, *args, **kwargs) -> bool:
    """True if an IAM describe/get call succeeds. Idempotency guard, same
    pattern used throughout this project: setup fails part-way often enough
    that every creation step needs to check first."""
    try:
        fn(*args, **kwargs)
        return True
    except ClientError:
        return False


def upsert_role(name: str, service: str, policy: dict,
                managed: list[str] = ()) -> str:
    """Create a role if it does not exist, and (re)attach its inline policy
    either way — re-running deploy after a policy change should update it,
    not skip it because the role already exists.

    The inline PutRolePolicy call is skipped when `policy` has no
    statements. An empty `"Statement": []` is not a harmless no-op policy —
    IAM rejects it outright with MalformedPolicyDocumentException, because a
    policy document is required to have at least one statement to be valid
    at all. Three of this project's seven roles (the EC2 instance role, the
    Batch service role, the Spot fleet role) need nothing beyond their
    managed policy and were passed an empty inline policy for that reason;
    this is what makes that a correct call rather than a crash.
    """
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": service},
        "Action": "sts:AssumeRole"}]}

    if not exists(iam.get_role, RoleName=name):
        iam.create_role(RoleName=name,
                        AssumeRolePolicyDocument=json.dumps(trust))
        print(f"   created role {name}")
    else:
        print(f"   role {name} exists")

    if policy.get("Statement"):
        iam.put_role_policy(RoleName=name, PolicyName=f"{name}-inline",
                            PolicyDocument=json.dumps(policy))
    for arn in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=arn)

    # Role creation is eventually consistent — a resource created right
    # after (a Batch compute environment, a Lambda) can fail to assume it
    # for a few seconds if used immediately.
    return iam.get_role(RoleName=name)["Role"]["Arn"]


def create_roles(account_id: str) -> dict:
    """Create all four roles. Returns a dict of name -> ARN."""
    bucket = config.bucket_name(account_id)
    secret_arn = (f"arn:aws:secretsmanager:{config.REGION}:{account_id}:"
                 f"secret:{config.SECRET_NAME}-*")
    param_arn = (f"arn:aws:ssm:{config.REGION}:{account_id}:"
                f"parameter{config.PARAM_PREFIX}/*")
    # GetParametersByPath, called with Path="/rag-pipeline/config", is
    # evaluated by IAM against the BARE path as the requested resource —
    # confirmed directly from a real AccessDeniedException, which named
    # exactly this ARN (no trailing /*) as what was checked and denied
    # when only param_arn (the wildcard form) was granted. Several
    # published examples show the wildcard form alone working for this
    # same action, which does not match what this account's own IAM
    # evaluation actually did — granting both forms is the version that
    # cannot be wrong regardless of which behavior is correct in general.
    param_prefix_arn = (f"arn:aws:ssm:{config.REGION}:{account_id}:"
                       f"parameter{config.PARAM_PREFIX}")

    print("-- IAM: execution role (ECS agent) --")
    # No SSM permission here, deliberately — checked directly against
    # batch.py's register_job_definition: every `secrets` entry it builds
    # points at secret_arn (Secrets Manager) only, never at an SSM
    # parameter ARN. The ECS agent, using this role, never resolves a
    # Parameter Store value at container launch. Tuning parameters are
    # fetched by the task's OWN CODE at runtime instead — ingest.py and
    # build_graph.py's own bootstrap, using ssm:GetParametersByPath on the
    # TASK role below — a completely different mechanism at a different
    # point in the container's life. Granting SSM here would be unused,
    # and worse, misleading: a future reader debugging "why isn't my
    # parameter being injected as a secret" would reasonably assume this
    # role's SSM access meant that injection path existed, when it does
    # not.
    exec_arn = upsert_role(
        config.EXEC_ROLE, "ecs-tasks.amazonaws.com",
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow",
             "Action": ["secretsmanager:GetSecretValue"],
             "Resource": secret_arn},
        ]},
        managed=["arn:aws:iam::aws:policy/service-role/"
                "AmazonECSTaskExecutionRolePolicy"],
    )

    print("-- IAM: task role (the running container) --")
    task_arn = upsert_role(
        config.TASK_ROLE, "ecs-tasks.amazonaws.com",
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow",
             "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
             "Resource": [f"arn:aws:s3:::{bucket}",
                         f"arn:aws:s3:::{bucket}/*"]},
            {"Effect": "Allow",
             "Action": ["dynamodb:PutItem", "dynamodb:GetItem",
                       "dynamodb:Query", "dynamodb:UpdateItem"],
             "Resource": [
                 f"arn:aws:dynamodb:{config.REGION}:{account_id}:"
                 f"table/{config.AUDIT_TABLE}",
                 f"arn:aws:dynamodb:{config.REGION}:{account_id}:"
                 f"table/{config.AUDIT_TABLE}/index/*"]},
            {"Effect": "Allow",
             # Both ingest.py and build_graph.py call
             # ssm.get_paginator("get_parameters_by_path") as the literal
             # first thing they do, before importing rag or graph_rag —
             # that bootstrap is what makes every tuning setting actually
             # reach the container. GetParametersByPath is a DIFFERENT IAM
             # action from GetParameter (which the classifier role needs
             # for its own, unrelated per-name lookups) — granting one
             # does not grant the other, confirmed by the classifier
             # hitting exactly this AccessDeniedException shape for its
             # own missing action. Found by cross-referencing every real
             # boto3 call the worker and graph worker make against what
             # this policy actually granted, rather than waiting for a
             # second live failure to report it.
             "Action": ["ssm:GetParametersByPath"],
             "Resource": [param_prefix_arn, param_arn]},
        ]},
    )

    print("-- IAM: classifier Lambda role --")
    classifier_arn = upsert_role(
        config.CLASSIFIER_ROLE, "lambda.amazonaws.com",
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow",
             "Action": ["s3:GetObject"],
             "Resource": f"arn:aws:s3:::{bucket}/*"},
            {"Effect": "Allow",
             "Action": ["dynamodb:GetItem", "dynamodb:PutItem",
                       "dynamodb:Query"],
             "Resource": [
                 f"arn:aws:dynamodb:{config.REGION}:{account_id}:"
                 f"table/{config.AUDIT_TABLE}",
                 f"arn:aws:dynamodb:{config.REGION}:{account_id}:"
                 f"table/{config.AUDIT_TABLE}/index/*"]},
            {"Effect": "Allow",
             # The classifier's own _load_batch_resources() reads
             # JOBDEF_<TIER> and QUEUE_<TIER> from Parameter Store — the
             # same path batch.deploy_all() writes them to. Missing this
             # was a real, confirmed bug: the first live upload failed with
             # AccessDeniedException on ssm:GetParameter, since nothing
             # anywhere had granted this role permission to read Parameter
             # Store at all. param_arn is the same resource pattern the
             # execution role above already uses, reused rather than
             # redefined.
             "Action": ["ssm:GetParameter"],
             "Resource": param_arn},
            {"Effect": "Allow",
             "Action": ["batch:SubmitJob"], "Resource": "*"},
        ]},
        managed=["arn:aws:iam::aws:policy/service-role/"
                "AWSLambdaBasicExecutionRole"],
    )

    print("-- IAM: error-handler Lambda role --")
    error_arn = upsert_role(
        config.ERROR_HANDLER_ROLE, "lambda.amazonaws.com",
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow",
             "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
             "Resource": f"arn:aws:s3:::{bucket}/*"},
            {"Effect": "Allow",
             "Action": ["dynamodb:GetShardIterator", "dynamodb:GetRecords",
                       "dynamodb:DescribeStream", "dynamodb:ListStreams"],
             "Resource": "*"},
        ]},
        managed=["arn:aws:iam::aws:policy/service-role/"
                "AWSLambdaBasicExecutionRole"],
    )

    print("-- IAM: graph trigger Lambda role --")
    graph_trigger_arn = upsert_role(
        config.GRAPH_TRIGGER_ROLE, "lambda.amazonaws.com",
        {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow",
             "Action": ["dynamodb:GetShardIterator", "dynamodb:GetRecords",
                       "dynamodb:DescribeStream", "dynamodb:ListStreams"],
             "Resource": "*"},
            {"Effect": "Allow",
             # Looks up JOBDEF_GRAPH/QUEUE_GRAPH — same GetParameter usage
             # as the classifier's own tier lookups, same resource pattern.
             "Action": ["ssm:GetParameter"],
             "Resource": param_arn},
            {"Effect": "Allow",
             "Action": ["batch:SubmitJob"], "Resource": "*"},
        ]},
        managed=["arn:aws:iam::aws:policy/service-role/"
                "AWSLambdaBasicExecutionRole"],
    )

    print("-- IAM: EC2 instance role (lets an instance join ECS) --")
    upsert_role(
        config.INSTANCE_ROLE, "ec2.amazonaws.com", {"Version": "2012-10-17",
        "Statement": []},
        managed=["arn:aws:iam::aws:policy/service-role/"
                "AmazonEC2ContainerServiceforEC2Role"],
    )
    instance_profile_arn = create_instance_profile(
        config.INSTANCE_PROFILE, config.INSTANCE_ROLE)

    print("-- IAM: Batch service role --")
    batch_service_arn = upsert_role(
        config.BATCH_SERVICE_ROLE, "batch.amazonaws.com",
        {"Version": "2012-10-17", "Statement": []},
        managed=["arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"],
    )

    print("-- IAM: Spot Fleet role --")
    spot_fleet_arn = upsert_role(
        config.SPOT_FLEET_ROLE, "spotfleet.amazonaws.com",
        {"Version": "2012-10-17", "Statement": []},
        managed=["arn:aws:iam::aws:policy/service-role/"
                "AmazonEC2SpotFleetTaggingRole"],
    )

    return {"exec": exec_arn, "task": task_arn,
           "classifier": classifier_arn, "error_handler": error_arn,
           "graph_trigger": graph_trigger_arn,
           "instance_profile": instance_profile_arn,
           "batch_service": batch_service_arn, "spot_fleet": spot_fleet_arn}


def create_instance_profile(profile_name: str, role_name: str) -> str:
    """An instance profile is the wrapper a role needs to be attached to an
    EC2 instance — a role ARN alone cannot be used in a launch template or a
    Batch compute environment's instanceRole field, only a profile ARN can.
    This is a real, easy-to-miss AWS distinction: creating the role is not
    sufficient on its own."""
    if not exists(iam.get_instance_profile, InstanceProfileName=profile_name):
        iam.create_instance_profile(InstanceProfileName=profile_name)
        # Eventually consistent — attaching the role immediately after
        # creation can otherwise race.
        import time
        time.sleep(2)
    profile = iam.get_instance_profile(InstanceProfileName=profile_name)
    attached = [r["RoleName"] for r in profile["InstanceProfile"]["Roles"]]
    if role_name not in attached:
        iam.add_role_to_instance_profile(
            InstanceProfileName=profile_name, RoleName=role_name)
    return profile["InstanceProfile"]["Arn"]
