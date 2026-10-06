# Deploying the clinical-trials agents — step by step

```
  STEP 0  prerequisites (once)         AWS user · Python packages · your keys ready
     │
  STEP 1  trial_graph    deploy.py     the graph agent (Neo4j)
  STEP 2  secrets                      OpenAI + Neo4j keys into AWS
  STEP 3  trial_graph    setup_neo4j   the name index both agents search
     │
  STEP 4  trial_search   deploy.py     the document agent (Pinecone)
  STEP 5  secrets                      Pinecone + Cohere keys into AWS
     │
  STEP 6  supervisor     deploy.py     routes between the two — always AFTER them
     │
  STEP 7  webapp/deploy  deploy.py     the web app: frontend + backend
     │
  STEP 8  check it works
```

Every command is run in the **PyCharm Terminal** (it opens with the project's
virtual environment already active). Each step starts at the project folder
`11-trial-agents-project`, goes into one sub-folder, runs one command, and comes
back. Every step has a **Mac** block and a **Windows** block — use the one for
your machine.

If a step fails, fix what it says and run the **same** step again — every
deploy finds what already exists and reuses it.

You need **no Docker and no Node**: the agents and the web app come as prebuilt
images from Docker Hub (`--image`), copied into your AWS account over HTTPS.
The instructor's Docker Hub user in these commands is `prudhviakella898`.

---

## STEP 0 — Prerequisites (once)

### 0.1 Go to the project folder

Every step below starts here.

**Mac**
```bash
cd ~/PycharmProjects/vs-langchain-agents/11-trial-agents-project
```

**Windows**
```powershell
cd $HOME\PycharmProjects\vs-langchain-agents\11-trial-agents-project
```

### 0.2 Check the virtual environment is active

The prompt must start with `(.venv)`. If it does not:

**Mac**
```bash
source ../.venv/bin/activate
```

**Windows**
```powershell
..\.venv\Scripts\activate
```

(If your `.venv` folder is somewhere else, use that path.)

### 0.3 An AWS user that is not root

Root keys cannot be limited by any policy. Use an IAM user (or SSO role) with
administrator access. Region: `us-east-1`.

**Mac**
```bash
aws configure
aws sts get-caller-identity
```

**Windows**
```powershell
aws configure
aws sts get-caller-identity
```

The `Arn` printed must **not** end in `:root`.

### 0.4 Python packages

**Mac**
```bash
pip install boto3 neo4j httpx
```

**Windows**
```powershell
pip install boto3 neo4j httpx
```

### 0.5 Keys you will paste in STEP 2 and STEP 5 — have them ready

| Key | Where it comes from |
|---|---|
| OpenAI API key | platform.openai.com → API keys |
| Neo4j URI + password | your Neo4j Aura instance (`neo4j+s://<id>.databases.neo4j.io`) |
| Pinecone API key | app.pinecone.io → API keys |
| Cohere API key | dashboard.cohere.com → API keys |

Never paste a key into a file you commit, or into a chat.

---

## STEP 1 — Deploy trial_graph

**Mac**
```bash
cd trial_graph
python deploy.py --image prudhviakella898/trial-graph-agent:1.0
cd ..
```

**Windows**
```powershell
cd trial_graph
python deploy.py --image prudhviakella898/trial-graph-agent:1.0
cd ..
```

**What it does:** checks your AWS access, creates the tools Lambda and its
Gateway, the guardrail, the prompt, the settings, copies the image into your
ECR, and starts the agent runtime. Takes 3–5 minutes.

**It worked when you see:**
```
done. trial_graph runtime: arn:aws:bedrock-agentcore:us-east-1:<account>:runtime/trial_graph-XXXXXXXXXX
```
followed by reminders that `trial-graph/neo4j` and `trial-agents/openai` still
hold placeholders. That is expected — STEP 2 fills them.

---

## STEP 2 — Put the OpenAI and Neo4j keys into AWS

The deploy created both secrets with placeholder values. Replace them.
`model` is the API model id exactly as shown (`gpt-6-sol`).

**Mac** — the JSON goes straight on the command line:
```bash
aws secretsmanager put-secret-value --secret-id trial-agents/openai --secret-string '{"api_key":"sk-...","model":"gpt-6-sol"}'
aws secretsmanager put-secret-value --secret-id trial-graph/neo4j --secret-string '{"uri":"neo4j+s://<id>.databases.neo4j.io","user":"neo4j","password":"..."}'
```

**Windows** — PowerShell mangles the quotes inside JSON, so write each secret
to a file, store it, then **delete the file**:
```powershell
notepad openai.json
#   paste:  {"api_key":"sk-...","model":"gpt-6-sol"}      then save and close Notepad
aws secretsmanager put-secret-value --secret-id trial-agents/openai --secret-string file://openai.json
del openai.json

notepad neo4j.json
#   paste:  {"uri":"neo4j+s://<id>.databases.neo4j.io","user":"neo4j","password":"..."}
aws secretsmanager put-secret-value --secret-id trial-graph/neo4j --secret-string file://neo4j.json
del neo4j.json
```

**It worked when** each command prints an `ARN` and a `VersionId`.
No redeploy is needed: the agents read secrets when they start.

---

## STEP 3 — Create the Neo4j name index

Both agents look trial names up through this index (`IMbrave150` →
`NCT03434379` → its protocol). Without it every name lookup fails.

**Mac**
```bash
cd trial_graph
python setup_neo4j.py
cd ..
```

**Windows**
```powershell
cd trial_graph
python setup_neo4j.py
cd ..
```

**It worked when you see:**
```
  index 'trial_entity_names': ONLINE, analyzer 'english'
```
Safe to run again at any time.

---

## STEP 4 — Deploy trial_search

`--pinecone-index` is the Pinecone index your RAG pipeline wrote the protocol
chunks to. In the course it is `rag-docs` — change it if yours has another name.

**Mac**
```bash
cd trial_search
python deploy.py --pinecone-index rag-docs --image prudhviakella898/trial-search-agent:1.0
cd ..
```

**Windows**
```powershell
cd trial_search
python deploy.py --pinecone-index rag-docs --image prudhviakella898/trial-search-agent:1.0
cd ..
```

**What it does:** same pattern as STEP 1, for the document agent. It reuses
the Neo4j secret from STEP 2 — if it says `trial-graph/neo4j does not exist`,
STEP 1 did not finish.

**It worked when you see:**
```
done. trial_search runtime: arn:aws:bedrock-agentcore:us-east-1:<account>:runtime/trial_search-XXXXXXXXXX
```
with reminders just above it that `trial-search/pinecone` and
`trial-search/cohere` hold placeholders — STEP 5 fills them.

---

## STEP 5 — Put the Pinecone and Cohere keys into AWS

**Mac**
```bash
aws secretsmanager put-secret-value --secret-id trial-search/pinecone --secret-string '{"api_key":"pcsk_..."}'
aws secretsmanager put-secret-value --secret-id trial-search/cohere --secret-string '{"api_key":"...","model":"rerank-v3.5"}'
```

**Windows**
```powershell
notepad pinecone.json
#   paste:  {"api_key":"pcsk_..."}      then save and close Notepad
aws secretsmanager put-secret-value --secret-id trial-search/pinecone --secret-string file://pinecone.json
del pinecone.json

notepad cohere.json
#   paste:  {"api_key":"...","model":"rerank-v3.5"}
aws secretsmanager put-secret-value --secret-id trial-search/cohere --secret-string file://cohere.json
del cohere.json
```

### Check the name lookup against your real graph (recommended)

**Mac**
```bash
cd trial_search
python check_resolve.py
cd ..
```

**Windows**
```powershell
cd trial_search
python check_resolve.py
cd ..
```

Every real name should list the right trial first; the made-up name
`a name that matches nothing xyz` should show `no candidates`.

---

## STEP 6 — Deploy the supervisor

Always after STEP 1 and STEP 4: the supervisor is allowed to call exactly the
agents that are registered when it deploys. It also creates the analyst
memory store, using the Pinecone key from STEP 5.

**Mac**
```bash
cd supervisor
python deploy.py --image prudhviakella898/trial-supervisor-agent:1.0
cd ..
```

**Windows**
```powershell
cd supervisor
python deploy.py --image prudhviakella898/trial-supervisor-agent:1.0
cd ..
```

**It worked when** its first step lists both agents:
```
=== STEP 1: registry ===
  trial_graph    arn:aws:bedrock-agentcore:...:runtime/trial_graph-...
  trial_search   arn:aws:bedrock-agentcore:...:runtime/trial_search-...
```
and it ends with `done. supervisor runtime: arn:...`.

If an agent is missing from that list, deploy it, then run STEP 6 again.

---

## STEP 7 — Deploy the web app

The web app deploy is two folders down (`webapp/deploy`), so you come back up
two levels.

**Mac**
```bash
cd webapp/deploy
python deploy.py --image prudhviakella898/trial-webapp-backend:1.0 --frontend-image prudhviakella898/trial-webapp-frontend:1.0
cd ../..
```

**Windows**
```powershell
cd webapp\deploy
python deploy.py --image prudhviakella898/trial-webapp-backend:1.0 --frontend-image prudhviakella898/trial-webapp-frontend:1.0
cd ..\..
```

**What it does:** reads the supervisor's ARN from `supervisor/deployment.json`,
creates the network, the backend on ECS Fargate behind a private load
balancer, an S3 bucket for the frontend, and a CloudFront address in front of
both. The first run takes **10–15 minutes** (CloudFront is slow to create).

**It worked when it ends with:**
```
done.  https://dxxxxxxxxxxxx.cloudfront.net
```
Open that address. A new username asks for your first and last name once.

### Cost, and deleting it after class

About USD 35 a month while it runs. To delete the web app:

**Mac**
```bash
cd webapp/deploy
python deploy.py --destroy
cd ../..
```

**Windows**
```powershell
cd webapp\deploy
python deploy.py --destroy
cd ..\..
```

---

## STEP 8 — Check it works

### 8.1 All three agents are READY

**Mac**
```bash
aws bedrock-agentcore-control list-agent-runtimes --query "agentRuntimes[].[agentRuntimeName,status]" --output table
```

**Windows**
```powershell
aws bedrock-agentcore-control list-agent-runtimes --query "agentRuntimes[].[agentRuntimeName,status]" --output table
```

Expect `trial_graph`, `trial_search` and `supervisor`, each `READY`.

### 8.2 Ask in the web app — one question per agent path

| Ask | It should |
|---|---|
| Which trials does Novo Nordisk sponsor? | use trial_graph only — a table |
| What are the exclusion criteria of the IMbrave150 trial? | use trial_search, scoped to one protocol |
| What does Novo Nordisk's obesity trial exclude? | trial_graph first, then trial_search |
| Remember that I prefer answers as tables. | call `remember_fact` (see "How it was answered") |

Each answer's **Details** shows the agents called, the queries that ran,
citations, and the memory tools used.

---

## When something goes wrong

| You see | Do this |
|---|---|
| `AWS credentials are not usable` | run `aws configure` again (STEP 0.3) |
| `... :root` in `aws sts get-caller-identity` | create an IAM user; do not deploy as root |
| `No module named 'boto3'` (or `neo4j`, `httpx`) | the virtual environment is not active — STEP 0.2, then 0.4 |
| `still holds placeholder` when the agent answers | STEP 2 / STEP 5 not done for that secret |
| Windows: `Error parsing parameter '--secret-string'` | use the Notepad + `file://` way in STEP 2 / STEP 5 |
| `index 'trial_entity_names' is not ONLINE` | STEP 2's Neo4j secret is wrong — fix it, re-run STEP 3 |
| `nothing registered under /trial-agents/registry` | STEP 1 and STEP 4 first, then STEP 6 |
| `supervisor/deployment.json missing` (STEP 7) | STEP 6 did not finish |
| `image ... not found on Docker Hub` | check the image name and tag letter by letter |
| `Role validation failed` / `cannot be assumed` | AWS is still creating the role — wait a minute, run the same step again |

Any step can be re-run safely. Logs for a running agent are in CloudWatch:
`/aws/bedrock-agentcore/runtimes/`.
