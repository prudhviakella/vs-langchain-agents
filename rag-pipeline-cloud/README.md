# RAG pipeline at scale — architecture and deployment

Ingests clinical trial PDFs into a vector index (Pinecone) and a knowledge
graph (Neo4j), running on AWS Batch with EC2 Spot instances, scaling from a
handful of documents to a very large corpus without a code change.

One command provisions everything. Uploading a PDF to `raw/` is the only
action needed to start processing it — nothing else to trigger by hand.

```
python deploy.py
python scripts/upload_pdf.py my-document.pdf
python scripts/status.py <doc-id>
```

---

## 1. Architecture

```
                    ┌─────────────┐
  PDF  ──upload──▶  │  S3  raw/   │
                    └──────┬──────┘
                           │ S3 event
                           ▼
                  ┌──────────────────┐
                  │ Lambda Classifier │  reads page count,
                  │                  │  consults history in
                  └────────┬─────────┘  DynamoDB, submits a
                           │            Batch job
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
     Batch queue     Batch queue     Batch queue
       small           medium           large
     2vCPU/7GB       2vCPU/7GB       2vCPU/7GB
     EC2 Spot         EC2 Spot         EC2 Spot
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                  ┌──────────────────┐
                  │  Stage 1 worker  │  parse → inspect → figures
                  │  (ingest.py)     │  → chunk → embed → upsert
                  └──┬────────┬───┬──┘
                     │        │   │
              ┌──────┘        │   └──────┐
              ▼               ▼          ▼
          Pinecone      S3 docs/    S3 cache/
          (vectors)   (reports,   (figure descriptions,
                       figures)    embeddings — survives
                                   a Spot reclaim)
                           │
                           │ doc marked COMPLETED
                           ▼
                  ┌──────────────────┐
                  │  Stage 2 worker  │  structure → registry → link
                  │ (build_graph.py) │
                  └──┬───────────┬───┘
                     ▼           ▼
                  Neo4j     ClinicalTrials.gov
                (graph)      (cached in S3 registry/)

  DynamoDB audit table records every stage of every run.
  Its Stream triggers a Lambda that moves failed documents to error/.
```

### Why two stages, not one

Stage 1 is the expensive part — Docling's models, real CPU time, minutes
per document. Stage 2 is a handful of Cypher writes and one cached HTTP
call — seconds. Coupling them means a transient Neo4j connection failure
throws away a nine-minute parse. Separated, the entire graph can be rebuilt
— a schema fix, a new node type — without re-parsing a single PDF.

### Why three size lanes, and why EC2 Spot instead of Fargate

Page count predicts processing cost badly on this kind of document: a
192-page protocol took 9 minutes; a 114-page one took 19.6 hours. Rather
than guess once and live with it, the classifier consults each document's
own processing **history** in DynamoDB and self-corrects — a document
mis-sized once lands correctly from then on. Page count is only the
fallback for a document with no history yet.

EC2 rather than Fargate because Stage 1's container image is
multi-gigabyte — Docling's model weights are baked in at build time. On
Fargate, every task pulls that image fresh. On EC2, an instance pulls it
once and reuses it across every job that instance runs for the rest of its
life. At scale, that difference dominates.

Every compute environment has `minvCpus=0` — an idle queue costs nothing,
which is what makes three separate lanes (plus the graph queue) affordable
instead of one lane sized for the worst case.

**The sizing above is for today's corpus (20 documents), not the eventual
scale target.** The three-lane *structure* is what a much larger corpus
needs; the *sizes* are grounded in what this corpus actually measured —
see `infra/config.py`'s `STAGE1_TIERS` comment for the real numbers behind
each figure, including the one genuine outlier (a 114-page document that
took 19.6 hours, traced to an accelerator-detection issue on the machine
that parsed it, not to page count). Scaling up later is a config change in
one place, not an architecture change.

---

## 2. Components

| Component | What it does |
|---|---|
| `deploy.py` | One-click provisioning of everything below |
| `teardown.py` | Reverses it, safely, with a confirmation prompt |
| `infra/` | The provisioning code deploy.py and teardown.py both call — one module per AWS service |
| `lambda_classifier/` | Reads page count + history, submits the Stage 1 Batch job |
| `lambda_error_handler/` | Moves a document's PDF to `error/` when its run genuinely fails |
| `lambda_graph_trigger/` | Submits the Stage 2 (graph build) job when a document's Stage 1 run completes |
| `worker/` | Stage 1 container — the full `rag` package plus `ingest.py` |
| `graph_worker/` | Stage 2 container — the `graph_rag` package plus `build_graph.py` |
| `scripts/upload_pdf.py` | Upload a PDF and print how to check on it |
| `scripts/status.py` | Read a document's full history from the audit table |

### `infra/` — one file per concern

```
config.py       every resource name, every Batch tier's sizing — the
               single source of truth every other module reads from
network.py      default VPC subnets and a security group
iam.py          seven roles: execution, task, classifier, error-handler,
               EC2 instance, Batch service, Spot fleet
storage.py      the S3 bucket (created empty) and the DynamoDB audit table
secrets.py      Secrets Manager placeholder + Parameter Store defaults
ecr.py          two repositories — see "why two images" below
batch.py        compute environments, job queues, job definitions,
               for both stages
lambdas.py      packages and deploys all three functions, wires all three triggers
```

### Why two container images, not one

Stage 1 needs Docling's models — multi-gigabyte. Stage 2 reads
`chunks.json` and talks to Neo4j and ClinicalTrials.gov; checked directly
against `graph_rag`'s actual imports, it needs neither Docling, nor
OpenAI, nor Pinecone. Sharing one image would mean every graph job pays the
same per-instance pull cost as a parse job, for weight it never uses —
reintroducing, for Stage 2, the exact problem separating EC2 from Fargate
was meant to solve for Stage 1.

---

## 3. The workflow, in order

1. **Upload.** `python scripts/upload_pdf.py document.pdf` puts the file at
  `s3://bucket/raw/document.pdf`.

2. **S3 event → Classifier Lambda.** Fires automatically — this is the
  self-starting part. The classifier:
   - computes `doc_id` from the filename (must match `rag.config.slugify`
     exactly, or history lookups silently never find a match — verified
     directly against the real implementation, not re-derived from memory)
   - reads the page count
   - checks DynamoDB for a previous **completed** run of this exact
     document; if one exists, uses whatever tier its measured duration
     implies (a run over an hour escalates to `large` regardless of what
     tier it ran in last time)
   - otherwise falls back to a page-count table
   - submits the job to the matching Batch queue

3. **Stage 1 worker runs the same five stages as local development:**
  parse → inspect → figures → chunk → index. Before starting, it syncs
  `s3://bucket/cache/` down to its local cache directory; after finishing
  (success or failure), it syncs back up. Reports (`extract.json`,
  `chunks.json`, and their Markdown equivalents) are uploaded to
  `s3://bucket/docs/<doc_id>/` regardless of outcome, because the case
  that matters most is a parse that died partway — the report it managed
  to write before dying is the one thing that explains why.

4. **On success**, the run is recorded `COMPLETED` in DynamoDB with the
  page count, chunk count, and how many chunks were added/removed from
  the index.

5. **On failure**, the run is recorded `FAILED`. The DynamoDB Stream
  fires the error handler, which — checking that this is a genuine
  transition into failure, not a status that was already `FAILED` from a
  previous attempt Batch is about to retry — moves the source PDF from
  `raw/` to `error/`.

6. **Stage 2 (graph build) starts automatically** the moment Stage 1's
  audit record transitions to `COMPLETED` — a third Lambda
  (`lambda_graph_trigger`), subscribed to the same DynamoDB Stream the
  error handler uses, submits the job. It reads `chunks.json` back from
  S3 (not Pinecone; that avoids the SDK-version-sensitive
  `list()`/`fetch()` API a Pinecone-based read would need). It writes
  `Document → Section → Chunk` structure for free, pulls the trial's
  registry facts from ClinicalTrials.gov (cached in S3, so a repeat run
  costs nothing), and links the two layers with a second pass —
  deliberately: the `ABOUT` edge from a Document to its Trial only forms
  once the Trial node exists, which it does not on the first pass.

---

## 4. Configuration

Two stores, deliberately not one — see `infra/secrets.py`'s module
docstring for the full reasoning.

**Secrets Manager** — one secret, three keys, created with placeholders:

```
OPENAI_API_KEY
PINECONE_API_KEY
NEO4J_PASSWORD
```

Set the real values after deploying:

```bash
aws secretsmanager put-secret-value --secret-id rag-pipeline/api-keys \
  --secret-string '{"OPENAI_API_KEY":"sk-...","PINECONE_API_KEY":"pc-...","NEO4J_PASSWORD":"..."}'
```

**Parameter Store** — tuning configuration, under `/rag-pipeline/config/`,
written with sensible defaults on first deploy and **never overwritten on
a re-deploy** if a value has been changed by hand:

```
USE_OUTLINE_HEADINGS     1
RENUMBER_HEADINGS        1
MIN_CHUNK_TOKENS         150
FIX_HEADING_HIERARCHY    1
CHUNK_TOKEN_TARGET       1024
EMBED_MODEL              text-embedding-3-small
INDEX_NAME               rag-docs
NEO4J_URI                replace-me (set this before Stage 2 will work)
NEO4J_USER               neo4j
```

Both stores are injected into the containers by the ECS agent as `secrets`
in the task definition, not as `environment` values — the values never
appear in the task definition, the console, or CloudTrail.

**This is not optional plumbing.** `rag.config` and `graph_rag.config`
both read every one of these settings with `os.getenv(...)` at import
time. Both `ingest.py` and `build_graph.py` load Parameter Store into the
process environment as the literal first thing they do, before importing
either package — get that ordering wrong and the container runs with
every default silently, which is exactly the bug this project hit twice
already during local development before it ever reached AWS.

---

## 5. Deploying

### Prerequisites

- **AWS credentials configured** (`aws configure`), with permission to create
 IAM roles, S3 buckets, DynamoDB tables, ECR repositories, Batch resources,
 and Lambda functions.

- **Docker Desktop installed AND running** — not just installed. Step 7
 shells out to `docker build` and `docker push`; if the daemon isn't up,
 that step fails immediately with a connection error. Open Docker Desktop
 and wait until it reports running, then confirm from a terminal before
 starting deploy:

 ```bash
 docker ps
 ```

 If that returns a (possibly empty) table rather than an error, Docker is
 ready.

- **A default VPC in the target region** — every account has one unless it
 has been deliberately deleted (see `infra/network.py` if not).

- **Python 3.11+**, with `pip install -r requirements.txt` — this installs
 `boto3` for the deploy tooling itself, separate from the two containers'
 own dependencies.

- **Free disk space and about 15 minutes for step 7 specifically.** The
 worker image bakes in Docling's model weights, and building it is by far
 the longest step in the whole deploy. Timed on a real run: roughly 4
 minutes to install Python dependencies, ~2 minutes to download the model
 weights, ~2 minutes to export the finished image — call it 10-15 minutes
 total for that one image, plus whatever your connection takes to push it
 to ECR. The graph worker image is unrelated and fast (a few seconds,
 mostly from Docker's layer cache). If step 7 looks stuck for several
 minutes with no new output, that is normal for the worker image
 specifically — the pip install and model download steps are genuinely
 that slow, not hung.

### Deploy

```bash
python deploy.py
```

Ten steps plus a quota check, each idempotent — safe to re-run if it fails
partway:

```
0/10  EC2 Spot vCPU quota check
1/10  IAM roles
2/10  S3 bucket
3/10  DynamoDB audit table
4/10  Secrets Manager
5/10  Parameter Store
6/10  ECR repositories
7/10  Build and push both container images
8/10  AWS Batch: compute environments, queues, job definitions
9/10  Lambda functions
10/10 Wiring the upload trigger
```

**Step 0 exists because of an AWS default that is easy to miss.** AWS Batch
on EC2 Spot draws from one account-and-region-wide quota — "All Standard
Spot Instance Requests" — that defaults to just **5 vCPUs** on many
accounts. The tiers as shipped (2/4/4 vCPU — see the note on `infra/
config.py`'s `STAGE1_TIERS` for why `large` is deliberately not bigger than
`medium` by default) fit under that, so step 0 passes with no action needed
out of the box. It still runs every time because raising any tier's `cpu`
value — restoring `large` to a genuinely bigger size for a larger corpus,
for instance — can push past the default again, and the failure that
produces is not an error: deploy would succeed completely, and the first
document routed to the resized tier would sit in `RUNNABLE` forever with
nothing anywhere — not the Batch console, not CloudWatch — explaining why.
Step 0 catches that before anything is created and prints the exact command
to request an increase. Pass `--skip-preflight` if you already know your
quota is sufficient.

Step 10 is deliberately last: everything the classifier needs — its own
role, the Batch queues, the Parameter Store values it reads on invocation
— must already exist before a real upload can arrive and fire it.

After it finishes, set the real secret values and your Neo4j URI (both
printed at the end of the run), then upload a PDF.

### Tearing down

```bash
python teardown.py           # secret recoverable for 7 days
python teardown.py --force   # secret deleted immediately — only for a
                             # genuinely disposable dev/test stack
```

Asks you to type the project name before doing anything, since this
deletes real data. Handles two easy-to-miss subtleties: the S3 bucket has
versioning enabled (a plain delete-objects call leaves old versions behind
and the bucket won't actually delete), and Batch compute environments must
be disabled and confirmed disabled before they can be removed.

### Running a second, isolated stack

```bash
PROJECT=rag-pipeline-staging python deploy.py
```

Every resource name derives from `PROJECT`, so this creates a completely
separate stack rather than colliding with an existing one.

---

## 6. Checking on things

```bash
python scripts/status.py nct03164772-heart-failure
```

Reads the audit table directly — the one place both stages agree to write
their state — and prints both stages' history for one document: every run,
every stage within the most recent run, and whether the graph has been
built yet.

For the corpus as a whole, the AWS Batch console shows queue depth per
lane; CloudWatch has alarms wired on queue depth and failed job count.

---

## 7. What this deliberately does not do, and why

**No dedicated VPC.** Uses the account's default VPC. A production
deployment with real network isolation requirements should replace
`infra/network.py`; building that here would be several hundred lines this
project does not need to prove the architecture.

**No exit-code-based retry discrimination.** Batch can distinguish a Spot
interruption from an application error via `evaluateOnExit`, keyed on exit
code. `ingest.py` returns exit code 1 uniformly for every failure — a
corrupt PDF and a Spot reclaim look identical at that level. Retries are
unconditional, up to `RETRY_ATTEMPTS`. The cost: a genuinely broken PDF is
retried once before landing in `error/`, instead of failing straight
there. Since re-ingestion is idempotent (content-addressed chunk ids,
sync as a set difference), the wasted retry costs one extra parse, not
correctness.

**Cache sync via `aws s3 sync`, not a real lookup service.** Round-tripping
the whole local cache directory through S3 is simple and correct at the
scale this project has actually measured — a fast diff of mostly-unchanged
small files. At a genuinely enormous cache (many millions of entries), a
full sync becomes its own bottleneck, and a real key-value lookup
(DynamoDB holding hash → S3 key, fetched lazily per miss) is the next
evolution. Not built here because the measured corpus does not need it
yet, and building it before it is needed is exactly the kind of
unmeasured complexity this project has otherwise avoided.

**No mid-document checkpointing.** A Spot-reclaimed document re-parses
from the start rather than resuming mid-parse. Measured directly: parsing
is roughly 95% of a document's total processing time and is a single
`converter.convert()` call with nothing to checkpoint inside it. What
makes a retry cheap instead is the S3-backed cache above — the retry
re-parses, but pays nothing for figure descriptions or embeddings it
already computed.

---

## 8. Local development

Every container's actual logic lives in `worker/rag/` and
`graph_worker/graph_rag/` — the same packages used for local, non-AWS
development (see the parent project's own README for that workflow). This
project's containers are thin wrappers: a Parameter Store bootstrap, S3
cache sync, and report upload, around exactly the same `build_records()` /
`load_structure()` calls a local notebook makes.

Running a worker locally against a real PDF, without any AWS resources
provisioned at all:

```bash
cd worker
python ingest.py --pdf ../path/to/document.pdf
```

`PARAM_PREFIX` is unset in that case, so the Parameter Store bootstrap is a
clean no-op — local `.env` supplies settings the normal way.
