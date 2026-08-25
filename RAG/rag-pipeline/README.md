# RAG Ingestion Pipeline

Turns PDFs into a searchable vector index. Runs on a laptop for one document and on
AWS for a corpus, using the same `ingest.py` in both cases.

```
PDF ─▶ extract ─▶ chunk ─▶ embed ─▶ Pinecone
```

## Layout

| File | Role |
|---|---|
| `01_ingestion.ipynb` | Guided run of one document, with a check after each stage. |
| `02_retrieval.ipynb` | Retrieval, compression, generation, evaluation. |
| `ingest.py` | Processes one PDF. The only file that touches documents. |
| `rag_common.py` | Config, embedding with cache, index access, manifest. |
| `Dockerfile` | Packages `ingest.py` for Fargate. |
| `aws/config.py` | Every resource name and tier size, in one place. |
| `aws/setup.py` | Creates all AWS resources. Safe to re-run. |
| `aws/sort_and_upload.py` | Sorts PDFs into size tiers, uploads, writes a manifest. |
| `aws/build_and_push.sh` | Builds the image and pushes it to ECR. |
| `aws/run.py` | Starts a run. |
| `aws/status.py` | Reads the audit trail. |
| `aws/teardown.py` | Deletes everything. |

## Part 1 — Local, one document

Two notebooks. They import from `ingest.py` and `rag_common.py` rather than restating
them, so what you verify in the notebook is exactly what runs on AWS — there is no
second copy to drift.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=pc-...

jupyter lab 01_ingestion.ipynb     # parse, inspect, chunk, index
jupyter lab 02_retrieval.ipynb     # retrieve, rerank, compress, evaluate
```

The same ingestion runs headless:

```bash
python ingest.py --pdf ./pdfs/AI-Enablers-Adopters-research-report.pdf
```

First run downloads Docling model weights (~500 MB). Every run then makes one
vision call per figure — seconds on a short report, minutes on a long one.

There is no parse cache. Its output depends on the settings as much as on the file,
so a cache keyed on the filename returns work done under different settings and
makes a changed setting look like it did nothing. Embeddings and table summaries
*are* cached, keyed on the content that produced them, where that failure cannot
happen.

### Validate the extraction before anything else

Ingestion writes two reports per document to `reports/`, and the first one lands
before a single chunk exists:

| File | Answers |
|---|---|
| `<doc>.extract.md` | Did the parse work? Every element in reading order, with table markdown and figure descriptions inline. |
| `<doc>.extract.json` | The same as data, for checking a corpus at once. |
| `<doc>.chunks.md` | Is the string being embedded the right string? Every chunk, with table summaries beside their fragments. |
| `<doc>.chunks.json` | Chunk metadata without the text. |

Failures are marked inline with a **MISSING** banner and collected at the top of the
report, because a missing figure description is an absence — and an absence is what
you fail to notice. Open `<doc>.extract.md` next to the PDF and read them together.

Check three things in the output:

- `pictures_described` equals `pictures` — otherwise figure descriptions silently
  did nothing and every chart is invisible to search.
- `formulas_undecoded` is 0 for any document containing equations.
- `0 over budget` — a chunk above the token limit is truncated by the embedding API
  without an error.

## Part 2 — AWS

### Sizing

Page count, not file size, predicts cost. A 33 MB scanned report can be shorter than
a 4 MB text one. Each tier maps to a task definition:

| Tier | Pages | CPU / memory | Concurrency |
|---|---|---|---|
| small | ≤ 50 | 2 vCPU / 4 GB | 4 |
| medium | 51–150 | 4 vCPU / 8 GB | 3 |
| large | > 150 | 8 vCPU / 16 GB | 3 |

Concurrency comes from measured lane load. On the 20-document reference corpus, 11
documents are large, so that lane decides total wall clock: at concurrency 1 it runs
about 52 minutes against 70 sequential, which is not worth building. At 3 it runs
about 17. Fargate bills per task-second, so the higher setting costs the same and
finishes sooner. Re-check this against your own corpus with `--dry-run`.

### Steps

```bash
python aws/setup.py

aws secretsmanager put-secret-value --secret-id rag-pipeline/api-keys \
  --secret-string '{"OPENAI_API_KEY":"sk-…","PINECONE_API_KEY":"pc-…"}'

bash aws/build_and_push.sh

python aws/preflight.py                                # quotas, keys, image
python aws/sort_and_upload.py --dir ./pdfs --dry-run   # check the split
python aws/sort_and_upload.py --dir ./pdfs

python aws/run.py
python aws/status.py
```

### The Fargate vCPU quota

This is the first thing that will stop a new account, and the error appears inside a
Step Functions branch as a generic task failure, so it is worth checking first.

Peak demand is `sum(cpu × concurrency)` across all three lanes running at once:

| Lane | vCPU each | Concurrency | vCPU |
|---|---|---|---|
| small | 2 | 4 | 8 |
| medium | 4 | 3 | 12 |
| large | 8 | 3 | 24 |
| | | | **44** |

New accounts default to **6 vCPU** — less than one large task. `preflight.py` compares
the configured demand against the actual quota and prints the increase request. Either
raise the quota or lower the concurrency values in `aws/config.py`.

### Why there is no ECS auto scaling here

ECS Service Auto Scaling scales an ECS **Service**, by adjusting `DesiredCount`. This
pipeline launches **standalone tasks** through `ecs:runTask.sync`, so there is no
service and nothing for Application Auto Scaling to attach to.

Parallelism is set by `MaxConcurrency` in each Map state and capped by the Fargate
vCPU quota. That is the whole model.

Auto scaling becomes the right tool when documents arrive continuously rather than as
a known batch: an SQS queue, an ECS Service consuming it, and a target-tracking policy
on backlog per task. That design trades the visible Step Functions graph and
declarative retries for queue semantics — visibility timeouts, heartbeats, dead-letter
queues — which is a larger jump in concepts than it looks.

`setup.py` checks for each resource before creating it, so re-running after a partial
failure is safe.

### Watching a run

The Step Functions console shows a live graph: three lanes side by side, one box per
document, green as they finish. `run.py` prints the link.

```bash
python aws/status.py                    # one line per document
python aws/status.py --doc nct04368728  # every stage of one document
python aws/status.py --failed
```

### Cleaning up

```bash
python aws/teardown.py
```

## Tables and figures

A markdown grid of numbers shares almost no vocabulary with a question like "how did
enrolment change" — the words enrolment and change appear in none of the cells. A
table embedded as raw markdown is close to unsearchable.

Every table above 400 characters therefore gets an extra chunk: a natural-language
summary naming what the table reports, its units, and its notable values. Retrieval
matches the summary; the raw rows remain as separate chunks carrying exact values,
linked to it by `table_id` and reachable through `prev_id`/`next_id`.

That keeps both properties without a second store: the description is findable, the
numbers are exact, and nothing has to be fetched from S3 at query time.

Figures are described by a vision model, and their PNGs are written to
`s3://BUCKET/figures/DOC_ID/` with the URI on the chunk, so an answer can show the
chart rather than only quote it.

### Chunk size

`CHUNK_TOKEN_TARGET` defaults to 1024. Table summaries carry the semantic weight for
tabular content, so fragments no longer have to stand alone and can stay small enough
to keep similarity scores sharp. Above roughly 2000 tokens a single vector averages
too many concepts and blunts both retrieval and reranking.

Treat it as a hyperparameter and sweep it against the labelled set:

```bash
CHUNK_TOKEN_TARGET=512 python ingest.py --pdf x.pdf
```

## How parallelism works here

`ingest.py` contains no distributed code — no queue polling, no locking, no
heartbeats. It processes one PDF and exits. Parallelism lives entirely in one field
of the state machine, `MaxConcurrency`, and the three lanes run at the same time.

Retries are safe because chunk IDs are derived from chunk content. Re-processing a
document upserts identical vectors and deletes nothing, so a duplicate or retried run
wastes money but cannot corrupt the index.

## Cost

For the 20-document, 2,785-page reference corpus:

| Item | Approx. |
|---|---|
| Vision calls | $0.20 |
| Embeddings | $0.03 |
| Fargate | $1–2 |
| DynamoDB, S3, Step Functions | cents |

Under $3 per full run, and near zero on re-runs that hit the parse cache.

