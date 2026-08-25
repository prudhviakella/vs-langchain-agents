"""Check that this account can actually run the configured parallelism.

    python aws/preflight.py

Three things stop a first run, and all three fail in ways that point elsewhere:

  - The Fargate vCPU quota. Measured in vCPUs, not tasks, and defaulting far below
    what three lanes need. Exceeding it surfaces inside a Step Functions branch as
    a generic task failure with nothing about quotas in the message.
  - Placeholder API keys. Surfaces forty minutes into a parse as an auth error from
    inside a container.
  - A missing or stale image. Surfaces as either a pull failure or, worse, a
    successful run of last week's code.

Checking them takes two seconds and costs nothing.
"""

import json

import boto3
from botocore.exceptions import ClientError

import config

# Service Quotas code for "Fargate On-Demand vCPU resource count". Quota codes are
# stable identifiers; the human-readable names change.
FARGATE_VCPU_QUOTA = ("fargate", "L-3032A538")


def required_vcpu() -> tuple[int, dict]:
    """Peak vCPU demand if all three lanes run at full concurrency at once.

    Which they will: the Parallel state starts every branch simultaneously. Summing
    the lanes rather than taking the maximum is the whole point — a configuration
    that looks modest per lane can add up well past the account quota.
    """
    per_lane = {
        # Fargate CPU units are thousandths of a vCPU; 2048 means 2 vCPU.
        tier: int(spec["cpu"]) // 1024 * spec["concurrency"]
        for tier, spec in config.TIERS.items()
    }
    return sum(per_lane.values()), per_lane


def current_quota() -> float | None:
    """The account's applied Fargate vCPU quota, or None if it cannot be read.

    Falls back to the documented default when the applied value is unavailable,
    which happens when the caller lacks servicequotas permissions. Returning None
    rather than raising keeps preflight advisory: an unreadable quota should not
    block a run that might be fine.
    """
    quotas = boto3.client("service-quotas", region_name=config.REGION)
    service, code = FARGATE_VCPU_QUOTA
    try:
        return quotas.get_service_quota(
            ServiceCode=service, QuotaCode=code)["Quota"]["Value"]
    except ClientError:
        try:
            return quotas.get_aws_default_service_quota(
                ServiceCode=service, QuotaCode=code)["Quota"]["Value"]
        except ClientError:
            return None


def check_secret() -> bool:
    """Whether real API keys have replaced the placeholders setup.py wrote."""
    sm = boto3.client("secretsmanager", region_name=config.REGION)
    try:
        value = json.loads(sm.get_secret_value(
            SecretId=config.SECRET_NAME)["SecretString"])
    except ClientError:
        print("   secret not found — run aws/setup.py first")
        return False
    unset = [k for k, v in value.items() if not v or v == "replace-me"]
    if unset:
        print(f"   placeholder values still set for: {', '.join(unset)}")
        return False
    # The values themselves are never printed or logged.
    print("   API keys present")
    return True


def check_image() -> bool:
    """Whether a :latest image has been pushed, and how recently.

    The push date is printed because the most common cause of a run executing stale
    code is forgetting to rebuild after editing ingest.py — which produces no error,
    just old behaviour.
    """
    ecr = boto3.client("ecr", region_name=config.REGION)
    try:
        images = ecr.describe_images(repositoryName=config.ECR_REPO,
                                     imageIds=[{"imageTag": "latest"}])["imageDetails"]
    except ClientError:
        print("   no :latest image — run aws/build_and_push.sh")
        return False
    size_gb = images[0]["imageSizeInBytes"] / 1e9
    print(f"   image present, {size_gb:.1f} GB, pushed {images[0]['imagePushedAt']:%Y-%m-%d}")
    return True


def main() -> None:
    """Check quota, secrets and image; exit non-zero if anything blocks a run."""
    print(f"preflight for {config.PROJECT} in {config.REGION}\n")
    ok = True

    print("── Fargate vCPU quota")
    needed, per_lane = required_vcpu()
    for tier, vcpu in per_lane.items():
        spec = config.TIERS[tier]
        print(f"   {tier:<8} {int(spec['cpu']) // 1024} vCPU x {spec['concurrency']}"
              f" concurrent = {vcpu:>3} vCPU")
    print(f"   {'peak':<8} {needed:>21} vCPU")

    quota = current_quota()
    if quota is None:
        print("   could not read the quota; check it manually in Service Quotas")
    elif quota < needed:
        ok = False
        print(f"\n   QUOTA TOO LOW: {quota:.0f} available, {needed} required.")
        print("   Tasks beyond the limit will fail to launch. Either request an")
        print("   increase (takes minutes to hours), or lower the concurrency")
        print("   values in aws/config.py.\n")
        print("   aws service-quotas request-service-quota-increase \\")
        print("     --service-code fargate --quota-code L-3032A538 \\")
        print(f"     --desired-value {needed} --region {config.REGION}")
    else:
        print(f"   quota {quota:.0f} vCPU — sufficient")

    print("\n── Secrets")
    ok &= check_secret()

    print("\n── Container image")
    ok &= check_image()

    # Non-zero exit so this can gate a deployment script, not just inform a human.
    print("\n" + ("ready to run" if ok else "not ready — fix the items above"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
