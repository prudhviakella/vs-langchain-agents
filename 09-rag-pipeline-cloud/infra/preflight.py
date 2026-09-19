"""Check the account's EC2 Spot vCPU quota before spending anything on it.

    from infra import preflight
    preflight.check(config.STAGE1_TIERS, config.STAGE2_MAX_VCPUS)

WHY THIS EXISTS

AWS Batch on EC2 Spot draws from one account-and-region-wide quota:
"All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests",
quota code L-34B43A08 under the ec2 service — confirmed against AWS's own
published defaults, not assumed. It defaults to just 5 vCPUs on many
accounts, and it applies across every Batch compute environment in the
account and region at once, not per lane.

The failure this produces if the quota is too low is not an error. `deploy.py`
creates every compute environment, queue, and job definition successfully —
none of that touches the quota. The first job whose tier needs more vCPU than
the quota allows sits in RUNNABLE forever: Batch cannot launch a Spot
instance for it, and nothing in the Batch console or CloudWatch logs says
why, because from Batch's perspective nothing has failed — it is still
waiting for capacity that will never arrive. Diagnosing a stuck RUNNABLE job
from nothing is far worse than seeing this check fail before deploy even
starts.

WHAT "ENOUGH" MEANS HERE

The quota is checked against the LARGEST single tier's per-task vCPU
requirement, not the sum across all lanes. The sum (every lane running its
own maximum concurrency simultaneously) is the ceiling this project could
eventually use, not what it needs to run its first job. A quota that covers
one job in the largest tier is enough to prove the pipeline works; scaling
concurrency further is a quota increase requested when actual queue depth
calls for it, not before.
"""

import boto3
from botocore.exceptions import ClientError

from . import config

SPOT_STANDARD_QUOTA = ("ec2", "L-34B43A08")

sq = boto3.client("service-quotas", region_name=config.REGION)


def current_quota() -> float | None:
    """The account's applied Spot vCPU quota, or None if it cannot be read.

    Falls back to the account default when the applied value is unreadable
    (missing servicequotas permission, a very new account). Returning None
    rather than raising keeps this advisory — an unreadable quota should
    not block a deploy that might be perfectly fine.
    """
    service, code = SPOT_STANDARD_QUOTA
    try:
        return sq.get_service_quota(
            ServiceCode=service, QuotaCode=code)["Quota"]["Value"]
    except ClientError:
        try:
            return sq.get_aws_default_service_quota(
                ServiceCode=service, QuotaCode=code)["Quota"]["Value"]
        except ClientError:
            return None


def required_vcpu(tiers: dict, stage2_cpu: str) -> int:
    """The largest single per-task vCPU requirement across both stages —
    the minimum quota needed to run one job successfully, not the sum of
    every lane's ceiling.
    """
    stage1_max = max(int(spec["cpu"]) // 1024 for spec in tiers.values())
    stage2 = int(stage2_cpu) // 1024
    return max(stage1_max, stage2)


def check(tiers: dict, stage2_cpu: str) -> None:
    """Print the quota status. Raises only when the quota is confirmed too
    low — an unreadable quota is a warning, not a blocker, since a false
    positive here is worse than letting a genuinely fine account proceed.
    """
    needed = required_vcpu(tiers, stage2_cpu)
    quota = current_quota()

    print("-- EC2 Spot vCPU quota (L-34B43A08) --")
    if quota is None:
        print("   could not read the applied or default quota — proceeding "
              "without this check. If a Batch job later sits in RUNNABLE "
              "indefinitely, this is the first thing to check by hand:")
        print("   aws service-quotas get-service-quota --service-code ec2 "
              "--quota-code L-34B43A08")
        return

    print(f"   quota: {quota:.0f} vCPU   |   largest single job needs: "
          f"{needed} vCPU")
    if quota < needed:
        raise SystemExit(
            f"\nThe account's EC2 Spot vCPU quota ({quota:.0f}) is below "
            f"what the largest configured tier needs ({needed}). Deploy "
            "would succeed, and the first large job would then sit in "
            "RUNNABLE forever with no error anywhere explaining why — "
            "Batch cannot launch a Spot instance for it and never will "
            "at this quota.\n\n"
            "Request an increase (approval is often near-instant for a "
            "modest amount like this):\n\n"
            "  aws service-quotas request-service-quota-increase \\\n"
            "    --service-code ec2 --quota-code L-34B43A08 \\\n"
            f"    --desired-value {max(needed, 32)}\n\n"
            "Or lower STAGE1_TIERS' cpu values in infra/config.py to fit "
            "the current quota, and re-run.\n\n"
            "To proceed anyway (e.g. you know the request is already "
            "in flight): python deploy.py --skip-preflight")
    print("   sufficient for the largest configured tier")
