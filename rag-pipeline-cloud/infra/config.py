"""Names, sizes, and settings for every AWS resource this pipeline creates.

One module rather than constants scattered across scripts, because deploy,
status, and teardown all have to agree on what things are called. A name that
drifts between deploy and teardown leaves an orphaned resource billing
quietly — the same reasoning as the earlier Fargate version of this project,
carried forward.

Change PROJECT to run a second, isolated copy of the whole stack.
"""

import os

PROJECT = os.getenv("PROJECT", "rag-pipeline")
REGION = os.getenv("AWS_REGION", "us-east-1")

# S3 bucket names are globally unique across ALL AWS accounts, not per-account,
# so the account id is appended to avoid colliding with anyone else's
# rag-pipeline. Resolved lazily in deploy.py once the account id is known,
# not at import time — importing this module must never require live AWS
# credentials, since checks and tests import it too.
BUCKET_NAME_TEMPLATE = "{project}-{account_id}"

# ─────────────────────────────────────────────────────────────────────────────
# S3 layout — one prefix per document, everything a run produces under it.
#
#     raw/<tier>/<doc>.pdf              uploaded PDFs, tier-sorted at upload time
#     docs/<doc_id>/
#         extract.md, extract.json       what the parse found
#         chunks.md, chunks.json         every chunk as it will be embedded
#         figures/fig_0001.png
#     cache/
#         figures/<sha256>.txt           figure descriptions, content-addressed
#         embeddings/<sha256>.npy        embeddings, content-addressed
#     registry/<nct_id>.json             ClinicalTrials.gov responses, cached
#     error/<doc>.pdf                    documents that failed after retries
#
# The cache prefix is what makes a Spot-reclaimed retry cheap: parsing a
# 250-page document again costs real time and money, but the figure
# descriptions and embeddings it already computed are looked up, not redone.
# Figure descriptions are also non-deterministic — measured earlier in this
# project, 12 of 17 changed across two identical runs — and since chunk ids
# are content-addressed, an uncached description changes the id and churns
# the index on every retry. The cache is not an optimisation here; without it
# a Spot-heavy run would silently duplicate work AND corrupt incremental sync.
# ─────────────────────────────────────────────────────────────────────────────
RAW_PREFIX = "raw"
DOCS_PREFIX = "docs"
CACHE_PREFIX = "cache"
REGISTRY_PREFIX = "registry"
ERROR_PREFIX = "error"

AUDIT_TABLE = f"{PROJECT}-audit"
ECR_REPO_WORKER = f"{PROJECT}-worker"
ECR_REPO_GRAPH = f"{PROJECT}-graph-worker"
SECRET_NAME = f"{PROJECT}/api-keys"
PARAM_PREFIX = f"/{PROJECT}/config"

TASK_ROLE = f"{PROJECT}-task-role"
EXEC_ROLE = f"{PROJECT}-exec-role"
CLASSIFIER_ROLE = f"{PROJECT}-classifier-role"
ERROR_HANDLER_ROLE = f"{PROJECT}-error-handler-role"
GRAPH_TRIGGER_ROLE = f"{PROJECT}-graph-trigger-role"

# EC2-on-Batch needs three roles Fargate never did, because Fargate has no EC2
# instance to register with ECS and no Spot fleet to manage:
#
#   instance role     what the EC2 INSTANCE itself assumes to register with
#                     ECS and pull the container image. This is not the task
#                     role — the task role is what the CONTAINER assumes once
#                     it is running; the instance role is what lets the
#                     instance exist as an ECS node at all.
#   batch service role   what the Batch SERVICE assumes to launch and
#                        terminate EC2 instances on your behalf as queue
#                        depth changes.
#   spot fleet role      what Batch assumes to request Spot capacity
#                        specifically. Required for SPOT compute
#                        environments; not needed for ON_DEMAND.
#
# Missing any of the three means a compute environment fails to create with
# an error that names the missing role, but only after everything else in
# deploy has already run — worth getting right up front rather than
# discovering it mid-deploy.
INSTANCE_ROLE = f"{PROJECT}-instance-role"
INSTANCE_PROFILE = f"{PROJECT}-instance-profile"
BATCH_SERVICE_ROLE = f"{PROJECT}-batch-service-role"
SPOT_FLEET_ROLE = f"{PROJECT}-spot-fleet-role"

CLASSIFIER_FUNCTION = f"{PROJECT}-classifier"
ERROR_HANDLER_FUNCTION = f"{PROJECT}-error-handler"
GRAPH_TRIGGER_FUNCTION = f"{PROJECT}-graph-trigger"

# ─────────────────────────────────────────────────────────────────────────────
# Batch — Stage 1 (parse and index)
#
# THREE TIERS FOR THE ARCHITECTURE, SIZED FOR TODAY'S CORPUS
#
# The three-lane structure exists for the scaling story this project was
# designed toward — a million-document corpus genuinely needs three
# differently-sized lanes running independently. The CURRENT corpus is 20
# documents, and sizing each lane as if it already had to serve that future
# would mean paying for capacity nothing here uses.
#
# minvCpus=0 on every lane regardless of size: an empty queue costs nothing,
# which is what makes three separate lanes affordable at all rather than one
# lane sized for the worst case.
#
# Grounded in what was actually measured on this exact corpus, not guessed:
#
#     19 documents, run sequentially           942s total  (~50s/doc average)
#     NCT03164772 (113 pages)                  60-124s to parse, depending on config
#     NCT02951156 (192 pages)                  552s to parse (3s/page)
#     NCT03155620 (114 pages)                  19.6 HOURS — a measured outlier,
#                                               traced to an accelerator/MPS
#                                               detection issue on the machine
#                                               it was parsed on, not to page
#                                               count or a genuine resource
#                                               shortage. More vCPU would not
#                                               have fixed that specific case;
#                                               it is flagged here so a similar
#                                               anomaly on Batch is investigated
#                                               rather than "fixed" by resizing.
#
# Page count decides the tier at classification time, but predicts cost
# badly on its own (the 192-page and 114-page cases above land nowhere near
# each other in actual duration) — the classifier consults measured history
# in DynamoDB first and falls back to page count only for a document with
# none yet.
#
# vCPU/memory below are roughly half the earlier diagram's figures — that
# diagram's 4/8/16 vCPU sizing was illustrative of the eventual scale
# target, not a measurement of this corpus's actual needs.
#
# max_vcpus is the real change. The diagram-era values (256/512/1024) size
# each LANE as if it might need hundreds of concurrent large-instance tasks
# — appropriate at a million documents, absurd at twenty. Sized here for a
# handful of concurrent jobs per lane, with real headroom over what 20
# documents actually produces, not over what a much larger corpus eventually
# will.
#
# SCALING BACK UP LATER
#
# Nothing else in the architecture changes. When the corpus genuinely grows,
# raise max_vcpus (and, if warranted, cpu/memory) here — the queues, the
# compute environments, the classifier's tier logic, all already built for
# it. This is the one place that should change, and the only reason to
# change it is measured queue depth or measured wall-clock time, not
# anticipation.
# ─────────────────────────────────────────────────────────────────────────────

# TEMPORARY — this account is under AWS's post-July-2025 Free Tier
# restriction, which hard-blocks launching any EC2 instance type outside a
# short allow-list until the account graduates out of the trial period.
# Confirmed directly: every "optimal" launch attempt failed with
# `InvalidParameterCombination: not eligible for Free Tier`, for every
# instance type tried, in every AZ — an account-level gate, not anything in
# this project's own configuration.
#
# m7i-flex.large is the best-fitting option on that allow-list: 2 vCPU,
# 8 GiB (confirmed against AWS's own published specs) — real memory headroom
# for a document-parsing workload, unlike the other eligible option
# (c7i-flex.large, compute-optimized with less memory per vCPU).
#
# This is the ONE reason `medium` and `large` below are sized down to 2 vCPU
# to match `small` — there is no free-tier-eligible instance type anywhere
# with 4+ vCPU, so the tiers cannot keep their real sizes and also fit this
# instance. This is a real, temporary loss of capacity, accepted to unblock
# testing rather than wait on an AWS Support case.
#
# REVERTING ONCE THE ACCOUNT GRADUATES
#
# Change this back to ["optimal"], and restore `medium` and `large` below to
# 8192/16384 (their pre-workaround values). Nothing else about the
# architecture depends on this value — it is read in exactly one place, in
# batch.py's create_compute_environment.
INSTANCE_TYPES = ["m7i-flex.large"]

STAGE1_TIERS = {
    # Memory raised to 7168 (7 GiB) from 4096, not the full 8192 the
    # instance has — real evidence this needed raising: NCT03961204 (84
    # pages, medium tier) was killed mid-parse with no Python traceback,
    # "Essential container in task exited" and exit code 137 in the Batch
    # console, the exact signature of an OOM kill from outside the
    # process. Docling running OCR, TableFormer in ACCURATE mode, picture
    # classification, formula enrichment, and code enrichment all at once
    # on that document exceeded the previous 4096 ceiling.
    #
    # NOT raised to the literal 8192: EC2-backed ECS/Batch always reserves
    # real memory on the host for the OS, Docker daemon, and ECS agent —
    # requesting the full physical amount for the container makes the job
    # unschedulable on any m7i-flex.large instance, since none of them
    # ever has that much genuinely free. That failure mode looks
    # identical to the earlier Free Tier quota problem (stuck in
    # RUNNABLE forever, no clear error), just from a different cause.
    # 1024 MiB of headroom is a conservative, round buffer against that.
    "small":  {"cpu": "2048", "memory": "7168",  "max_vcpus": 8,
              "page_ceiling": 50},
    "medium": {"cpu": "2048", "memory": "7168",  "max_vcpus": 16,
              "page_ceiling": 150},
    "large":  {"cpu": "2048", "memory": "7168",  "max_vcpus": 16,
              "page_ceiling": None},   # no ceiling — anything larger lands here
}

# `large` is deliberately the SAME size as `medium` here, not bigger — worth
# explaining, since two differently-named tiers with identical per-task
# specs looks like an oversight if you don't know why.
#
# AWS's EC2 Spot vCPU quota ("All Standard Spot Instance Requests",
# L-34B43A08 — see infra/preflight.py) defaults to 5 on many accounts, and
# it is account-and-region-wide, not per lane. `large` genuinely warrants
# more than `medium` — 8 vCPU is a defensible size for this corpus's
# biggest documents — but 8 exceeds that default quota outright, and a job
# whose OWN requirement exceeds the account's quota can never run at any
# quota utilisation; it sits in RUNNABLE forever, which is exactly the
# silent failure infra/preflight.py exists to catch before deploy even
# starts.
#
# This matters differently for one deploy than for a class of 220 students
# each deploying to their own AWS account. A single quota-increase request
# is a two-minute, usually-automatic approval — verified directly on a real
# account: PENDING to CASE_CLOSED in about three minutes for a modest
# increase to 32. But 220 individual requests hitting 220 different
# accounts is 220 chances for one of them to land on an account that gets
# flagged for manual review, with no way to predict which student that will
# be or how long they wait. For a class, the point is the tiered-queue
# PATTERN — small/medium/large routing, three independent Spot lanes — not
# that `large` specifically needs 8 vCPU rather than 4. Keeping every tier
# at or under the default quota means nobody's first day is spent filing an
# AWS support case.
#
# A single deployer with a genuinely large corpus, or anyone who has
# already requested the increase (see the README), can safely restore
# `large` to 8192/16384 — nothing else about the architecture depends on
# this specific value.

# Stage 2 (graph build) needs none of Docling's models — it reads chunks.json
# and talks to Neo4j and ClinicalTrials.gov. One small queue is enough; it is
# a separate queue rather than reusing Stage 1's, deliberately, so the graph
# backlog can never compete with parse jobs for the same Spot capacity.
STAGE2_QUEUE = f"{PROJECT}-graph"
STAGE2_CPU = "1024"
STAGE2_MEMORY = "2048"
# 128 was sized the same way Stage 1's original max_vcpus was — for a future
# corpus, not today's. 20 documents' worth of graph jobs, each a handful of
# Cypher writes and one cached HTTP call, never needs more than a few
# concurrent. Raise this the same way as Stage 1's tiers, when queue depth
# actually says to.
STAGE2_MAX_VCPUS = 8

# Wall-clock ceiling per Stage 1 job. A large protocol with many figures can
# run well over an hour; this exists so a hung task fails and requeues rather
# than holding a lane's capacity indefinitely. Generous on purpose — this is
# not where a 19.6-hour outlier should be caught; that needs investigation,
# not a timeout that silently discards 19 hours of GPU-adjacent compute on
# every attempt.
STAGE1_TIMEOUT_SECONDS = 14400   # 4 hours
STAGE2_TIMEOUT_SECONDS = 1800    # 30 minutes

# Retries: Spot interruption and transient AWS errors are worth one retry.
# An application error (a corrupt PDF, a bug) retrying on the same input
# fails the same way twice and just doubles the cost of learning that.
RETRY_ATTEMPTS = 2

# ─────────────────────────────────────────────────────────────────────────────
# Configuration that reaches the containers
#
# Split deliberately: secrets in Secrets Manager, tuning knobs in Parameter
# Store. Conflating them means every chunking-parameter change needs the same
# review process as an API key rotation, which is friction nobody wants, and
# it means secrets end up in a place set up for casual editing.
# ─────────────────────────────────────────────────────────────────────────────
SECRET_KEYS = ["OPENAI_API_KEY", "PINECONE_API_KEY", "NEO4J_PASSWORD"]

# name -> default. Written to Parameter Store at deploy time if not already
# present, so a re-deploy never clobbers a value someone tuned by hand.
#
# Every key here must match a real os.getenv(...) call in the worker or
# graph worker — confirmed directly, this project's own history did not
# hold for three of them. INDEX_NAME and EMBED_MODEL were bare hardcoded
# literals in worker/rag/config.py with no os.getenv() at all, so a
# Parameter Store change had zero effect regardless of how correctly the
# bootstrap loaded it. This key was worse: it used to read "CHUNK_TOKENS",
# but the worker has only ever read CHUNK_TOKEN_TARGET (see
# worker/rag/config.py's own comment showing local usage as
# `CHUNK_TOKEN_TARGET=512 python ingest.py ...`) — setting "CHUNK_TOKENS"
# here did not merely get ignored the same way, it silently configured a
# key that never existed anywhere in the code being read.
DEFAULT_PARAMS = {
    "USE_OUTLINE_HEADINGS": "1",
    "RENUMBER_HEADINGS": "1",
    "MIN_CHUNK_TOKENS": "150",
    "FIX_HEADING_HIERARCHY": "1",
    "CHUNK_TOKEN_TARGET": "1024",
    "EMBED_MODEL": "text-embedding-3-small",
    "INDEX_NAME": "rag-docs",
    # NOT an empty string. SSM's PutParameter rejects a zero-length Value
    # outright — confirmed directly: deploy crashed on exactly this line with
    # ValidationException "Value must have length greater than or equal to
    # 1" the first time this was "" as a deliberate not-yet-set placeholder.
    # There is no way to store "unset" as an empty value in Parameter Store;
    # it has to be a real, non-empty sentinel instead. "replace-me" matches
    # the same convention already used for the Secrets Manager placeholders,
    # and graph_worker/graph_rag/store.py's driver() rejects it explicitly
    # with a clear message rather than letting it reach a real connection
    # attempt and fail as a cryptic low-level error.
    #
    # Any future parameter here that needs a "not yet configured" default
    # must use a non-empty sentinel for the same reason — never "".
    "NEO4J_URI": "replace-me",
    "NEO4J_USER": "neo4j",
}


def bucket_name(account_id: str) -> str:
    """The bucket name, which needs the account id and so cannot be a
    module-level constant — importing this module must not require a live
    AWS session."""
    return BUCKET_NAME_TEMPLATE.format(project=PROJECT, account_id=account_id)


def param_name(key: str) -> str:
    """Full Parameter Store path for one tuning key."""
    return f"{PARAM_PREFIX}/{key}"
