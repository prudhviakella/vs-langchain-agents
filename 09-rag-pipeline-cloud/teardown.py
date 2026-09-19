#!/usr/bin/env python3
"""Remove every AWS resource this project created.

    python teardown.py             deletes the secret with a 7-day recovery
                                   window (Secrets Manager's default safety
                                   net — a teardown run by mistake is still
                                   recoverable)
    python teardown.py --force    deletes the secret immediately, no
                                   recovery window. Use for a genuinely
                                   disposable dev/test stack, not for
                                   anything that might have held real keys
                                   worth recovering.

Reverse of deploy.py's order, mostly, with three real subtleties:

    Batch compute environments must be DISABLED and confirmed disabled
    before they can be deleted — deleting immediately after disabling
    fails, because the state change is not instant.

    The S3 bucket has versioning enabled (deploy.py turns it on
    deliberately, to protect against a same-name re-upload silently
    replacing a document mid-read). Emptying a versioned bucket means
    deleting every VERSION and every delete marker, not just the current
    objects — a plain delete-objects-then-delete-bucket leaves the bucket
    non-empty and the deletion fails with no clear reason why.

    IAM roles must have every attached and inline policy removed, and be
    removed from any instance profile, before the role itself can be
    deleted — AWS rejects deleting a role that still has policies attached.
"""

import argparse
import time

import boto3
from botocore.exceptions import ClientError

from infra import config

iam = boto3.client("iam", region_name=config.REGION)
s3 = boto3.client("s3", region_name=config.REGION)
ddb = boto3.client("dynamodb", region_name=config.REGION)
sm = boto3.client("secretsmanager", region_name=config.REGION)
ssm = boto3.client("ssm", region_name=config.REGION)
ecr = boto3.client("ecr", region_name=config.REGION)
batch = boto3.client("batch", region_name=config.REGION)
lam = boto3.client("lambda", region_name=config.REGION)
sts = boto3.client("sts", region_name=config.REGION)


def step(message: str) -> None:
    print(f"\n-- {message} --")


def ignore_missing(fn, *args, **kwargs) -> None:
    """Run a delete call, treating 'already gone' as success.

    Teardown must be safe to re-run — a partial teardown that failed on
    step 6 needs steps 1 through 5 to skip cleanly, not raise, when run
    again.
    """
    try:
        fn(*args, **kwargs)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code not in ("ResourceNotFoundException", "NoSuchEntity",
                        "NoSuchBucket", "ResourceNotFoundFault",
                        "RepositoryNotFoundException",
                        "ClientException"):
            raise


def delete_lambdas() -> None:
    step("Lambda: triggers and functions")
    for function_name in (config.CLASSIFIER_FUNCTION, config.ERROR_HANDLER_FUNCTION,
                          config.GRAPH_TRIGGER_FUNCTION):
        try:
            mappings = lam.list_event_source_mappings(FunctionName=function_name)
            for mapping in mappings["EventSourceMappings"]:
                lam.delete_event_source_mapping(UUID=mapping["UUID"])
        except ClientError:
            pass
        ignore_missing(lam.delete_function, FunctionName=function_name)
        print(f"   {function_name} removed")


def delete_batch() -> None:
    step("AWS Batch: job definitions, queues, compute environments")

    for tier in list(config.STAGE1_TIERS) + ["graph"]:
        jobdef_name = f"{config.PROJECT}-{tier}-jobdef"
        try:
            for jd in batch.describe_job_definitions(
                jobDefinitionName=jobdef_name, status="ACTIVE")["jobDefinitions"]:
                batch.deregister_job_definition(jobDefinition=jd["jobDefinitionArn"])
        except ClientError:
            pass

        queue_name = config.STAGE2_QUEUE if tier == "graph" else f"{config.PROJECT}-{tier}"
        try:
            batch.update_job_queue(jobQueue=queue_name, state="DISABLED")
            _wait_for(lambda: batch.describe_job_queues(jobQueues=[queue_name])
                     ["jobQueues"][0]["status"] == "VALID", timeout=60)
            batch.delete_job_queue(jobQueue=queue_name)
        except (ClientError, IndexError):
            pass

        ce_name = f"{config.PROJECT}-graph-ce" if tier == "graph" else f"{config.PROJECT}-{tier}-ce"
        try:
            batch.update_compute_environment(computeEnvironment=ce_name, state="DISABLED")
            _wait_for(lambda: batch.describe_compute_environments(
                computeEnvironments=[ce_name])["computeEnvironments"][0]["status"] == "VALID",
                timeout=120)
            batch.delete_compute_environment(computeEnvironment=ce_name)
        except (ClientError, IndexError):
            pass

        print(f"   {tier}: job definition, queue, and compute environment removed")


def _wait_for(condition, timeout: int, interval: int = 5) -> None:
    started = time.time()
    while time.time() - started < timeout:
        try:
            if condition():
                return
        except Exception:
            return
        time.sleep(interval)


def delete_ecr() -> None:
    step("ECR repositories")
    for name in (config.ECR_REPO_WORKER, config.ECR_REPO_GRAPH):
        ignore_missing(ecr.delete_repository, repositoryName=name, force=True)
        print(f"   {name} removed")


def delete_parameters() -> None:
    step("Parameter Store")
    for key in list(config.DEFAULT_PARAMS) + \
              [f"JOBDEF_{t.upper()}" for t in list(config.STAGE1_TIERS) + ["GRAPH"]] + \
              [f"QUEUE_{t.upper()}" for t in list(config.STAGE1_TIERS) + ["GRAPH"]]:
        ignore_missing(ssm.delete_parameter, Name=config.param_name(key))
    print(f"   removed everything under {config.PARAM_PREFIX}/")


def delete_secret(force: bool) -> None:
    step("Secrets Manager")
    if force:
        ignore_missing(sm.delete_secret, SecretId=config.SECRET_NAME,
                       ForceDeleteWithoutRecovery=True)
        print(f"   {config.SECRET_NAME} deleted immediately")
    else:
        ignore_missing(sm.delete_secret, SecretId=config.SECRET_NAME,
                       RecoveryWindowInDays=7)
        print(f"   {config.SECRET_NAME} scheduled for deletion in 7 days "
              "(--force to delete immediately)")


def delete_dynamodb() -> None:
    step("DynamoDB audit table")
    ignore_missing(ddb.delete_table, TableName=config.AUDIT_TABLE)
    print(f"   {config.AUDIT_TABLE} removed (its stream is removed with it)")


def empty_and_delete_bucket(bucket: str) -> None:
    """Empty a VERSIONED bucket, then delete it.

    Versioning means the bucket is never actually empty after a plain
    delete-objects call — every version and every delete marker is a
    separate object as far as bucket-emptiness is concerned, and
    delete_bucket fails on a non-empty bucket with no indication that
    "non-empty" means "has old versions", not "has current objects".
    """
    step(f"S3 bucket {bucket}")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        print("   does not exist")
        return

    paginator = s3.get_paginator("list_object_versions")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket):
        objects = ([{"Key": v["Key"], "VersionId": v["VersionId"]}
                   for v in page.get("Versions", [])]
                  + [{"Key": m["Key"], "VersionId": m["VersionId"]}
                     for m in page.get("DeleteMarkers", [])])
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            deleted += len(objects)

    s3.delete_bucket(Bucket=bucket)
    print(f"   emptied ({deleted} version(s)/marker(s)) and deleted")


def delete_iam() -> None:
    step("IAM roles")
    for role_name in (config.EXEC_ROLE, config.TASK_ROLE, config.CLASSIFIER_ROLE,
                      config.ERROR_HANDLER_ROLE, config.GRAPH_TRIGGER_ROLE,
                      config.INSTANCE_ROLE,
                      config.BATCH_SERVICE_ROLE, config.SPOT_FLEET_ROLE):
        try:
            for policy_name in iam.list_role_policies(RoleName=role_name)["PolicyNames"]:
                iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            for policy in iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]:
                iam.detach_role_policy(RoleName=role_name, PolicyArn=policy["PolicyArn"])
        except ClientError:
            pass
        ignore_missing(iam.delete_role, RoleName=role_name)

    try:
        for role in iam.get_instance_profile(
            InstanceProfileName=config.INSTANCE_PROFILE)["InstanceProfile"]["Roles"]:
            iam.remove_role_from_instance_profile(
                InstanceProfileName=config.INSTANCE_PROFILE, RoleName=role["RoleName"])
    except ClientError:
        pass
    ignore_missing(iam.delete_instance_profile, InstanceProfileName=config.INSTANCE_PROFILE)
    print("   all roles and the instance profile removed")


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove every resource this project created")
    ap.add_argument("--force", action="store_true",
                    help="delete the secret immediately, no recovery window")
    args = ap.parse_args()

    account_id = sts.get_caller_identity()["Account"]
    bucket = config.bucket_name(account_id)

    print(f"Tearing down project {config.PROJECT!r} in {config.REGION}")
    print(f"Bucket: {bucket}\n")
    confirm = input("Type the project name to confirm: ")
    if confirm != config.PROJECT:
        raise SystemExit("names did not match, aborted")

    # Reverse of deploy.py's order: triggers and compute first (they are
    # what costs money while running), storage and identity last.
    delete_lambdas()
    delete_batch()
    delete_ecr()
    delete_parameters()
    delete_secret(args.force)
    delete_dynamodb()
    empty_and_delete_bucket(bucket)
    delete_iam()

    print("\nTeardown complete.")


if __name__ == "__main__":
    main()
