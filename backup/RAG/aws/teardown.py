"""Delete everything setup.py created.

    python aws/teardown.py            prompts before deleting
    python aws/teardown.py --yes      no prompt
    python aws/teardown.py --keep-bucket

Order matters: the state machine references the task definitions, which reference
the roles. Deleting a role still in use leaves the dependent resource in a broken
state that is harder to clean up than the original.
"""

import argparse

import boto3
from botocore.exceptions import ClientError

import config

REGION = config.REGION
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def attempt(label: str, fn, *args, **kwargs) -> None:
    """Delete one resource, reporting rather than raising if it is already gone.

    Teardown has to be resumable. A partial setup, or a second teardown after an
    interrupted first, leaves some resources missing — and stopping on the first
    NotFound would strand everything after it, which is exactly the resources that
    cost money.
    """
    try:
        fn(*args, **kwargs)
        print(f"   deleted {label}")
    except ClientError as exc:
        print(f"   skipped {label} ({exc.response['Error']['Code']})")


def empty_bucket(bucket: str) -> None:
    """Remove every object so the bucket can be deleted.

    S3 refuses to delete a non-empty bucket. Versions are cleared as well as current
    objects: if versioning was ever enabled, deleting an object only writes a delete
    marker and the bucket stays non-empty — a common way for teardown to appear to
    succeed while leaving a billed bucket behind.
    """
    resource = boto3.resource("s3", region_name=REGION).Bucket(bucket)
    resource.object_versions.delete()
    resource.objects.all().delete()


def main() -> None:
    """Delete every project resource, in dependency order."""
    ap = argparse.ArgumentParser(description="Delete all pipeline resources")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--keep-bucket", action="store_true")
    args = ap.parse_args()

    # Typing the project name rather than "y": this deletes an entire stack, and the
    # name is what distinguishes one student's from another's.
    print(f"about to delete every {config.PROJECT} resource in {REGION}")
    if not args.yes and input("type the project name to confirm: ") != config.PROJECT:
        raise SystemExit("aborted")

    sfn = boto3.client("stepfunctions", region_name=REGION)
    ecs = boto3.client("ecs", region_name=REGION)
    iam = boto3.client("iam", region_name=REGION)

    # Dependents first, so nothing is deleted while still referenced.
    print("\n── Step Functions")
    attempt(config.STATE_MACHINE, sfn.delete_state_machine, stateMachineArn=(
        f"arn:aws:states:{REGION}:{ACCOUNT}:stateMachine:{config.STATE_MACHINE}"))

    print("\n── ECS")
    for tier in config.TIERS:
        family = f"{config.PROJECT}-{tier}"
        try:
            # Every setup run registers a new revision, so a family accumulates them.
            # All have to be deregistered, not just the latest.
            for arn in ecs.list_task_definitions(
                    familyPrefix=family)["taskDefinitionArns"]:
                ecs.deregister_task_definition(taskDefinition=arn)
            print(f"   deregistered {family}")
        except ClientError as exc:
            print(f"   skipped {family} ({exc.response['Error']['Code']})")
    attempt(config.CLUSTER, ecs.delete_cluster, cluster=config.CLUSTER)

    print("\n── IAM")
    for role in (config.TASK_ROLE, config.EXEC_ROLE, config.SFN_ROLE):
        try:
            # A role cannot be deleted while any policy is attached to it, and
            # managed and inline policies detach differently.
            for policy in iam.list_attached_role_policies(
                    RoleName=role)["AttachedPolicies"]:
                iam.detach_role_policy(RoleName=role, PolicyArn=policy["PolicyArn"])
            for name in iam.list_role_policies(RoleName=role)["PolicyNames"]:
                iam.delete_role_policy(RoleName=role, PolicyName=name)
        except ClientError:
            # The role is already gone; attempt() below reports it.
            pass
        attempt(role, iam.delete_role, RoleName=role)

    print("\n── Data stores")
    attempt(config.AUDIT_TABLE,
            boto3.client("dynamodb", region_name=REGION).delete_table,
            TableName=config.AUDIT_TABLE)
    # force=True because a repository with images in it cannot be deleted otherwise.
    attempt(config.ECR_REPO, boto3.client("ecr", region_name=REGION).delete_repository,
            repositoryName=config.ECR_REPO, force=True)
    attempt(config.LOG_GROUP, boto3.client("logs", region_name=REGION).delete_log_group,
            logGroupName=config.LOG_GROUP)
    # Secrets keep a 7-to-30-day recovery window unless forced, and the name stays
    # reserved for that whole period — which blocks re-creating the same stack
    # during a teach-and-retry cycle.
    attempt(config.SECRET_NAME,
            boto3.client("secretsmanager", region_name=REGION).delete_secret,
            SecretId=config.SECRET_NAME, ForceDeleteWithoutRecovery=True)

    if args.keep_bucket:
        # Worth keeping when the parse cache in it represents hours of extraction.
        print(f"\n   kept bucket {config.BUCKET}")
    else:
        print("\n── S3")
        try:
            empty_bucket(config.BUCKET)
        except ClientError:
            pass
        attempt(config.BUCKET, boto3.client("s3", region_name=REGION).delete_bucket,
                Bucket=config.BUCKET)

    print("\nteardown complete")


if __name__ == "__main__":
    main()
