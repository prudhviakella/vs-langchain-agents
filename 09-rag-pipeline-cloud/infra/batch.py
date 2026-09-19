"""AWS Batch: compute environments, job queues, job definitions.

Four compute environments total — small, medium, large for Stage 1, one more
for Stage 2 — each EC2 Spot, each minvCpus=0. An empty queue costs nothing,
which is what makes four separate lanes affordable instead of one lane sized
for the worst case.

Job DEFINITIONS are registered here (static: which image, how much memory,
which roles). Job SUBMISSION happens at runtime in the classifier Lambda,
once per document — this module only provisions what submission needs to
exist.

RETRIES, HONESTLY

Batch's retryStrategy can distinguish a Spot interruption from an
application error via evaluateOnExit, keyed on exit code or status reason.
This project's worker (ingest.py) returns exit code 1 for every failure
uniformly — a corrupt PDF and a Spot reclaim look identical at the exit-code
level. Building real discrimination would mean changing the worker's exit
codes to encode failure class, which is a reasonable future improvement and
not done here.

So retries are unconditional, up to RETRY_ATTEMPTS, same as the Step
Functions version of this project used (`MaxAttempts: 1`, i.e. one retry).
The cost of this: a genuinely broken PDF is retried once before landing in
error/ instead of failing straight there. Given re-ingestion is idempotent —
chunk ids are content-addressed and sync is a set difference — the wasted
retry costs one extra parse, not correctness.
"""

import boto3
from botocore.exceptions import ClientError

from . import config

batch = boto3.client("batch", region_name=config.REGION)
logs = boto3.client("logs", region_name=config.REGION)
ssm = boto3.client("ssm", region_name=config.REGION)


def exists(fn, *args, **kwargs) -> bool:
    try:
        result = fn(*args, **kwargs)
        # Batch's describe_* calls return 200 with an empty list rather than
        # raising for a name that does not exist, unlike most other
        # services' guard pattern used elsewhere in this project.
        for key in ("computeEnvironments", "jobQueues", "jobDefinitions"):
            if key in result:
                return len(result[key]) > 0
        return True
    except ClientError:
        return False


def create_log_group() -> None:
    print(f"-- CloudWatch log group /ecs/{config.PROJECT} --")
    try:
        logs.create_log_group(logGroupName=f"/ecs/{config.PROJECT}")
        print("   created")
    except logs.exceptions.ResourceAlreadyExistsException:
        print("   exists")


def _disable_and_delete_queue(name: str) -> None:
    """Disable and delete a job queue, waiting for each step to actually
    finish — not just for the API call that starts it.

    THE BUG THIS FIXES

    The first version of this function called delete_job_queue() and
    returned immediately. Confirmed directly, on a real account: that call
    only STARTS the deletion — the queue exists in a DELETING state for a
    real amount of time afterward, not instantly gone. The caller
    (_disable_and_delete_ce) proceeded straight to deleting the compute
    environment, which failed with the exact same error this whole
    fallback exists to work around:

        ClientException: Cannot delete, found existing JobQueue
        relationship.

    because the queue — mid-deletion, not yet actually gone — still counted
    as an existing relationship from the compute environment's side. This
    is the identical class of mistake already caught once in this same
    file for the compute environment's own DISABLE step (state changes
    are not instant, so waiting after an API call has to mean waiting for
    the state to actually change, not waiting a fixed guess or not at
    all) — just not yet applied to this delete specifically.

    Waits for describe_job_queues to actually fail to find the queue
    (exists() returning False) as the one unambiguous "truly gone" signal,
    rather than polling a status field that may itself report DELETING for
    a while before disappearing.
    """
    if not exists(batch.describe_job_queues, jobQueues=[name]):
        return
    batch.update_job_queue(jobQueue=name, state="DISABLED")
    import time
    started = time.time()
    while time.time() - started < 60:
        queue = batch.describe_job_queues(jobQueues=[name])["jobQueues"][0]
        if queue["status"] == "VALID":
            break
        time.sleep(3)

    batch.delete_job_queue(jobQueue=name)
    started = time.time()
    while time.time() - started < 90:
        if not exists(batch.describe_job_queues, jobQueues=[name]):
            return
        time.sleep(3)

    # A timeout here must never look like success. The first version of
    # this function fell through silently when the wait ran out, which is
    # exactly what let a later run reach delete_compute_environment while
    # the queue was still finishing — confirmed directly: small and medium
    # deleted within the old 60s window in the same run that large did not,
    # pure timing variance rather than anything tier-specific. Raising here
    # gives the caller (and _disable_and_delete_ce's own retry, added for
    # the same reason) a real signal instead of a silent maybe.
    raise RuntimeError(
        f"job queue {name!r} did not finish deleting within 90s — "
        "AWS is taking longer than usual; re-running deploy.py should "
        "pick up from here once it settles.")


def _disable_and_delete_ce(name: str, queue_name: str) -> None:
    """Disable and delete a compute environment, waiting for each state
    transition before moving on.

    Removes the QUEUE first, deliberately — confirmed directly, on a real
    account, that Batch refuses to delete a compute environment still
    referenced by any job queue:

        ClientException: Cannot delete, found existing JobQueue
        relationship.

    This was missed on the first version of this fallback, which deleted
    only the compute environment and hit exactly that error immediately
    after the earlier fix (catching the service-linked-role requirement)
    correctly triggered. The queue is recreated afterward by the caller's
    own existing create_job_queue() call, which runs unconditionally after
    this function returns and will see no queue by this name and create a
    fresh one pointed at the newly recreated compute environment — no
    special-casing needed there.

    Same sequence as teardown.py's own compute-environment cleanup — not
    imported from there, since teardown.py depends on infra, not the other
    way around, and this is a small enough sequence that duplicating it is
    simpler than restructuring the import direction for one shared helper.

    Batch rejects deleting a compute environment that is not yet DISABLED,
    and disabling is not instant — hence waiting for VALID (Batch's signal
    that the disable itself has finished processing, not that it never
    existed) before attempting the delete.
    """
    _disable_and_delete_queue(queue_name)
    batch.update_compute_environment(computeEnvironment=name, state="DISABLED")
    _wait_for_ce(name, timeout=60)

    # Retried, not a single attempt — _disable_and_delete_queue above only
    # confirms describe_job_queues stops finding the queue, which is not
    # quite the same moment as the compute environment's OWN internal view
    # of that relationship clearing. A short retry absorbs whatever gap
    # exists between the two without needing to guess its size up front.
    import time
    for attempt in range(4):
        try:
            batch.delete_compute_environment(computeEnvironment=name)
            break
        except batch.exceptions.ClientException as exc:
            if "existing JobQueue relationship" not in str(exc) or attempt == 3:
                raise
            time.sleep(10)

    # Waited for actual completion, not a fixed guess — the identical
    # mistake was just made and caught on _disable_and_delete_queue above
    # (a fixed time.sleep(10) here would carry the same risk: too short on
    # a slower day, and the caller's subsequent create_compute_environment
    # call would race a still-deleting resource under the same name).
    started = time.time()
    while time.time() - started < 60:
        if not exists(batch.describe_compute_environments, computeEnvironments=[name]):
            return
        time.sleep(3)


def create_compute_environment(name: str, max_vcpus: int, instance_profile_arn: str,
                               batch_service_arn: str, spot_fleet_arn: str,
                               subnets: list[str], sg_id: str,
                               queue_name: str) -> str:
    """One EC2 Spot compute environment.

    SPOT_CAPACITY_OPTIMIZED rather than the cost-only strategy: it spreads
    across instance pools by interruption likelihood, not price alone, which
    matters more here than shaving a further few percent off an already-
    cheap Spot price — a document that gets interrupted twice costs more in
    wasted parse time than the price difference between pools.
    """
    print(f"-- Batch compute environment {name} --")
    if exists(batch.describe_compute_environments, computeEnvironments=[name]):
        # An in-place update of instanceTypes was the first thing tried
        # here, and it is a real, documented capability — but only for a
        # compute environment using AWS's own Batch service-linked role.
        # A compute environment using a CUSTOM service role (what
        # infra/iam.py creates) rejects it outright:
        #
        #   ClientException: Following fields ... instanceTypes ... can be
        #   updated only for Non-fargate Compute Environment having a
        #   Batch Service Linked Role.
        #
        # confirmed directly, on this exact call, against a real compute
        # environment — not found by reading docs first. The precondition
        # was documented by AWS in the context of updating AMIs specifically
        # and did not generalise, in what was checked beforehand, to cover
        # instanceTypes as well.
        #
        # Rather than switch every compute environment to the service-
        # linked role account-wide — a larger change than this warrants —
        # this falls back to disabling and recreating the compute
        # environment under the same name. Batch compute environment ARNs
        # are name-based, so a queue that already references this ARN
        # keeps working against the recreated environment with no change
        # of its own needed.
        try:
            batch.update_compute_environment(
                computeEnvironment=name,
                computeResources={"instanceTypes": config.INSTANCE_TYPES,
                                  "maxvCpus": max_vcpus},
            )
            _wait_for_ce(name)
            print(f"   exists — instanceTypes and maxvCpus updated to "
                  f"{config.INSTANCE_TYPES}")
            return _ce_arn(name)
        except batch.exceptions.ClientException as exc:
            if "Batch Service Linked Role" not in str(exc):
                raise
            print("   exists, but cannot update instanceTypes in place on "
                  "a custom service role — disabling and recreating")
            _disable_and_delete_ce(name, queue_name)
            # Falls through to the create path below.

    batch.create_compute_environment(
        computeEnvironmentName=name,
        type="MANAGED",
        state="ENABLED",
        computeResources={
            "type": "SPOT",
            "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
            "minvCpus": 0,
            "maxvCpus": max_vcpus,
            "instanceTypes": config.INSTANCE_TYPES,
            "subnets": subnets,
            "securityGroupIds": [sg_id],
            "instanceRole": instance_profile_arn,
            "spotIamFleetRole": spot_fleet_arn,
        },
        serviceRole=batch_service_arn,
    )
    _wait_for_ce(name)
    print("   created")
    return _ce_arn(name)


def _ce_arn(name: str) -> str:
    return batch.describe_compute_environments(
        computeEnvironments=[name])["computeEnvironments"][0]["computeEnvironmentArn"]


def _wait_for_ce(name: str, timeout: int = 120) -> None:
    """Batch compute environments take a short but real time to reach VALID
    after creation, before a job queue can be pointed at them. Polling
    rather than a fixed sleep, since the actual time varies."""
    import time
    started = time.time()
    while time.time() - started < timeout:
        ce = batch.describe_compute_environments(
            computeEnvironments=[name])["computeEnvironments"][0]
        if ce["status"] == "VALID":
            return
        if ce["status"] == "INVALID":
            raise RuntimeError(
                f"compute environment {name} became INVALID: "
                f"{ce.get('statusReason', 'no reason given')}")
        time.sleep(3)
    raise RuntimeError(f"compute environment {name} did not become VALID "
                       f"within {timeout}s")


def create_job_queue(name: str, ce_arn: str) -> str:
    print(f"-- Batch job queue {name} --")
    if exists(batch.describe_job_queues, jobQueues=[name]):
        print("   exists")
        return _queue_arn(name)

    # Retried, not a single attempt — confirmed directly, on a real
    # account: describe_job_queues can report a name as gone (which is
    # what let the exists() check above proceed to create rather than
    # skip) while the actual CreateJobQueue call, moments later, still
    # rejects it with "Object already exists". This is the identical
    # CLASS of AWS timing gap already caught once for
    # delete_compute_environment — a describe-based check and the
    # write operation it gates do not always agree on a resource's state
    # at the exact same instant — just showing up at a different layer
    # this time: describe-says-gone-but-create-still-rejects, rather than
    # delete-not-actually-finished. Same fix: a short retry rather than
    # trusting the describe check alone.
    import time
    for attempt in range(4):
        try:
            batch.create_job_queue(
                jobQueueName=name, state="ENABLED", priority=1,
                computeEnvironmentOrder=[{"order": 1, "computeEnvironment": ce_arn}],
            )
            break
        except batch.exceptions.ClientException as exc:
            if "already exists" not in str(exc) or attempt == 3:
                raise
            time.sleep(10)
    print("   created")
    return _queue_arn(name)


def _queue_arn(name: str) -> str:
    return batch.describe_job_queues(
        jobQueues=[name])["jobQueues"][0]["jobQueueArn"]


def _vcpu_value(cpu_units: str) -> str:
    """Batch CPU units (thousandths of a vCPU, e.g. "4096") to the string
    resourceRequirements actually wants.

    `str(int(cpu_units) / 1024)` — the first version of this — uses Python's
    true division, so every whole-number vCPU comes out "2.0", "4.0", "1.0".
    Checked against every documented AWS example (the CloudFormation
    ResourceRequirement reference, RegisterJobDefinition's own docs, and
    Pulumi's generated examples): all of them show either a clean integer
    ("512") or a genuine fraction ("0.25") — never a whole number with a
    trailing .0. Confirmed on this project's own configured tiers that the
    bug was live: 2048/4096/4096/1024 CPU units all produced "2.0", "4.0",
    "4.0", "1.0" before this fix.

    Whole numbers format as bare integers; genuine fractions (a future
    cpu="512" for half a vCPU) keep their decimal — plain integer division
    would silently turn 512 into "0", which is a worse bug than the one
    being fixed.
    """
    vcpu = int(cpu_units) / 1024
    return str(int(vcpu)) if vcpu == int(vcpu) else str(vcpu)


def register_job_definition(name: str, image_uri: str, vcpu: str, memory: str,
                            exec_role_arn: str, task_role_arn: str,
                            secret_arn: str, timeout_seconds: int) -> str:
    """One job definition. Registering with the same name creates a new
    REVISION rather than failing — Batch versions job definitions, so a
    re-deploy after an image update naturally produces job:N+1 without
    needing its own idempotency check.
    """
    print(f"-- Batch job definition {name} --")

    secrets = [{"name": key, "valueFrom": f"{secret_arn}:{key}::"}
              for key in config.SECRET_KEYS]

    batch.register_job_definition(
        jobDefinitionName=name,
        type="container",
        containerProperties={
            "image": image_uri,
            "resourceRequirements": [
                {"type": "VCPU", "value": _vcpu_value(vcpu)},
                {"type": "MEMORY", "value": memory},
            ],
            "jobRoleArn": task_role_arn,
            "executionRoleArn": exec_role_arn,
            "secrets": secrets,
            # Confirmed missing entirely, not just AUDIT_TABLE: this
            # field did not exist at all before, so nothing plain (as
            # opposed to secret) ever reached the container. Both
            # ingest.py and build_graph.py read AUDIT_TABLE via
            # os.getenv as their --audit-table default, and PARAM_PREFIX
            # is what makes their Parameter Store bootstrap run at all
            # rather than silently no-op (by design, for local
            # development with no AWS resources at hand — the same
            # designed behavior was firing here for the wrong reason).
            # Real evidence this was broken: three real runs of the same
            # document, all stuck at the classifier's own SUBMITTED
            # write with no STARTED, stage, COMPLETED, or FAILED record
            # ever appearing — exactly what a no-op Audit object
            # produces.
            "environment": [
                {"name": "AUDIT_TABLE", "value": config.AUDIT_TABLE},
                {"name": "PARAM_PREFIX", "value": config.PARAM_PREFIX},
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {"awslogs-group": f"/ecs/{config.PROJECT}",
                           "awslogs-stream-prefix": name},
            },
        },
        retryStrategy={"attempts": config.RETRY_ATTEMPTS},
        timeout={"attemptDurationSeconds": timeout_seconds},
    )
    revision = batch.describe_job_definitions(
        jobDefinitionName=name, status="ACTIVE")["jobDefinitions"][0]
    print(f"   registered, revision {revision['revision']}")
    return revision["jobDefinitionArn"]


def deploy_all(account_id: str, worker_image: str, graph_image: str,
              roles: dict) -> dict:
    """Provision every Batch resource for both stages.

    Returns the resource names/ARNs the classifier and the CLI helpers need
    at runtime, and also writes the job-definition ARNs to Parameter Store —
    the classifier Lambda reads them from there rather than hardcoding names
    or calling describe_job_definitions on every invocation.
    """
    from . import network
    subnets, sg_id = network.default_network()
    secret_arn = (f"arn:aws:secretsmanager:{config.REGION}:{account_id}:"
                 f"secret:{config.SECRET_NAME}")

    create_log_group()

    result = {"queues": {}, "job_defs": {}}

    for tier, spec in config.STAGE1_TIERS.items():
        ce_name = f"{config.PROJECT}-{tier}-ce"
        queue_name = f"{config.PROJECT}-{tier}"
        jobdef_name = f"{config.PROJECT}-{tier}-jobdef"

        ce_arn = create_compute_environment(
            ce_name, spec["max_vcpus"], roles["instance_profile"],
            roles["batch_service"], roles["spot_fleet"], subnets, sg_id,
            queue_name)
        create_job_queue(queue_name, ce_arn)
        jobdef_arn = register_job_definition(
            jobdef_name, worker_image, spec["cpu"], spec["memory"],
            roles["exec"], roles["task"], secret_arn,
            config.STAGE1_TIMEOUT_SECONDS)

        result["queues"][tier] = queue_name
        result["job_defs"][tier] = jobdef_arn
        ssm.put_parameter(Name=config.param_name(f"JOBDEF_{tier.upper()}"),
                          Value=jobdef_arn, Type="String", Overwrite=True)
        ssm.put_parameter(Name=config.param_name(f"QUEUE_{tier.upper()}"),
                          Value=queue_name, Type="String", Overwrite=True)

    print("-- Stage 2: graph build queue --")
    graph_ce_arn = create_compute_environment(
        f"{config.PROJECT}-graph-ce", config.STAGE2_MAX_VCPUS,
        roles["instance_profile"], roles["batch_service"],
        roles["spot_fleet"], subnets, sg_id, config.STAGE2_QUEUE)
    create_job_queue(config.STAGE2_QUEUE, graph_ce_arn)
    graph_jobdef_arn = register_job_definition(
        f"{config.PROJECT}-graph-jobdef", graph_image,
        config.STAGE2_CPU, config.STAGE2_MEMORY,
        roles["exec"], roles["task"], secret_arn,
        config.STAGE2_TIMEOUT_SECONDS)

    result["queues"]["graph"] = config.STAGE2_QUEUE
    result["job_defs"]["graph"] = graph_jobdef_arn
    ssm.put_parameter(Name=config.param_name("JOBDEF_GRAPH"),
                      Value=graph_jobdef_arn, Type="String", Overwrite=True)
    ssm.put_parameter(Name=config.param_name("QUEUE_GRAPH"),
                      Value=config.STAGE2_QUEUE, Type="String", Overwrite=True)

    return result
