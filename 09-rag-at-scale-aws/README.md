# Running the pipeline on AWS

The local notebooks handle one document at a time. This runs a whole corpus in
parallel on Fargate.

Same code — `ingest.py` here is the container entrypoint and it imports the same
`rag` package the notebooks use, so what you tested locally is what runs.

**This builds the vector index, not the graph.** The graph module reads chunks back
out of that index afterwards and writes to Neo4j Aura, which it can do from
anywhere.

## Steps

```bash
python setup.py                            # create every resource

aws secretsmanager put-secret-value --secret-id rag-pipeline/api-keys \
  --secret-string '{"OPENAI_API_KEY":"sk-…","PINECONE_API_KEY":"pc-…"}'

bash build_and_push.sh                     # build and push the image
python preflight.py                        # check quotas before spending anything

python upload_pdfs.py --dir ./pdfs --dry-run   # see the split
python upload_pdfs.py --dir ./pdfs             # upload and write the manifest

python run.py                              # start it
python status.py                           # see what happened
```

`setup.py` checks for each resource before creating it, so re-running after a failure
is safe. `teardown.py` removes everything.

## API keys

Created by `setup.py` as a Secrets Manager secret, and declared in the task
definition as `secrets` rather than `environment`.

The difference matters: the ECS agent fetches the values and injects them into the
container at launch, so the keys never appear in the task definition, in the console,
or in CloudTrail. Nothing in this repository ever holds a key.

## How the parallelism works

Documents are sorted by page count into three tiers, and each tier gets its own lane
with its own machine size and concurrency. All three lanes run at once.

| Lane | Pages | Machine | At a time |
|---|---|---|---|
| small | ≤ 50 | 2 vCPU / 4 GB | 4 |
| medium | 51–150 | 4 vCPU / 8 GB | 3 |
| large | > 150 | 8 vCPU / 16 GB | 3 |

`ingest.py` contains no distributed code — no queues, no locking, no heartbeats. It
processes one PDF and exits. The parallelism is one field in the state machine.

Run `upload_pdfs.py --dry-run` first and look at the split. If most of your documents
land in one lane, that lane alone decides the total time and the other two are
decoration — the concurrency numbers in `config.py` are set from a measured corpus,
not guessed, and yours may differ.

## Read this before your first run

**The Fargate vCPU quota will stop you.** Ten containers at these sizes is 44 vCPU.
New accounts default to **6** — less than one large task.

The failure appears inside a Step Functions branch as a generic task error, with
nothing about quotas in the message. `preflight.py` checks it first and prints the
increase request.
