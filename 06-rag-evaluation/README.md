# RAG Evaluation Pipeline — MLflow 3
**Applied GenAI Course · Vidya Sankalp · Prudhvi**

End-to-end RAG evaluation pipeline for HotpotQA using OpenAI embeddings, Pinecone vector store, and MLflow 3 for experiment tracking.

- **Part A** → Hand-coded metrics → MLflow **Model Training** tab
- **Part B** → `mlflow.genai.evaluate()` + scorers → MLflow **GenAI** tab

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1 — Start MLflow Server](#step-1--start-the-mlflow-tracking-server)
3. [Step 2 — Prepare Dataset](#step-2--prepare-the-golden-dataset)
4. [Step 3 — Ingest into Pinecone](#step-3--ingest-documents-into-pinecone)
5. [Step 4 — Run Evaluation](#step-4--run-the-evaluation)
6. [Step 5 — Register a Model](#step-5--register-a-model-optional)
7. [Understanding the Metrics](#understanding-the-metrics)
   - [Family 1 — Retrieval Metrics](#family-1--retrieval-metrics)
   - [Family 2 — Lexical Metrics](#family-2--lexical-metrics)
   - [Family 3 — LLM-as-Judge Metrics](#family-3--llm-as-judge-metrics)
   - [The RAG Triad](#the-rag-triad--the-most-important-composite-score)
   - [Metric Quick Reference](#metric-quick-reference-card)
   - [Diagnostic Guide](#diagnostic-guide--what-to-do-when-scores-are-low)
8. [Project Structure](#project-structure)
9. [MLflow UI Quick Reference](#mlflow-ui-quick-reference)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Python environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install all dependencies
pip install \
  mlflow>=3.0 \
  openai \
  pinecone-client \
  pandas \
  psycopg2-binary \
  boto3 \
  datasets \
  ragas
```

### 2. PostgreSQL database

```bash
# Connect to postgres and create the database + user
psql -U postgres

# Inside psql shell:
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
\q
```

> **Note:** If your password contains special characters, URL-encode them:
> `@` → `%40`  |  `#` → `%23`  |  `$` → `%24`  |  `%` → `%25`

### 3. AWS S3 bucket for artifacts

```bash
aws s3 mb s3://mlflow-prudhvi --region ap-south-1
aws sts get-caller-identity   # confirm credentials are configured
```

> Running locally without S3? Use `--default-artifact-root ./mlflow-artifacts` instead.

### 4. Environment variables

Create a `.env` file in the project root (never commit this to git):

```bash
# .env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcn-...
PINECONE_INDEX=hotpotqa-ragbench-mini
PINECONE_NS=hotpotqa
EMBEDDING_MODEL=text-embedding-3-large
GENERATOR_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
TOP_K=5
CHUNK_SIZE=400

# MLflow
MLFLOW_URI=http://localhost:5001
MLFLOW_EXPERIMENT_A=RAG-HotpotQA-Evaluation
MLFLOW_EXPERIMENT_B=RAG-HotpotQA-GenAI-Eval
```

Load before running scripts:

```bash
export $(cat .env | xargs)
```

---

## Step 1 — Start the MLflow Tracking Server

Open a **dedicated terminal** and keep it running throughout your session.

### Option A: PostgreSQL + S3 (recommended)

```bash
mlflow server \
  --backend-store-uri postgresql://mlflow_user:your_password%40here@localhost:5432/mlflow_db \
  --default-artifact-root s3://mlflow-prudhvi/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5001
```

### Option B: SQLite + local filesystem (dev / course use)

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5001
```

### Option C: Background mode (keeps terminal free)

```bash
nohup mlflow server \
  --backend-store-uri postgresql://mlflow_user:your_password%40here@localhost:5432/mlflow_db \
  --default-artifact-root s3://mlflow-prudhvi/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5001 > mlflow_server.log 2>&1 &

# Verify it started
curl -s http://localhost:5001/health   # should return {"status": "OK"}

# Stop the server later
kill $(lsof -t -i:5001)
```

Open **http://localhost:5001** — you should see the MLflow UI with two tabs:
- **GenAI** — Part B traces and scorer results
- **Model Training** — Part A metrics and model registry

---

## Step 2 — Prepare the Golden Dataset

```bash
python 1_prepare_dataset.py
```

Downloads RAGBench HotpotQA from HuggingFace and produces:

```
data/
  golden_hotpotqa_30.jsonl    # 30 question / answer / context triples
  corpus_hotpotqa.jsonl       # document corpus for ingestion
```

**Common error:**
```
ImportError: cannot import name 'load_dataset' from 'datasets'
```
Fix: rename any local file or folder named `datasets` to something else.

---

## Step 3 — Ingest Documents into Pinecone

```bash
python 2_ingest.py
```

Chunks documents → embeds with `text-embedding-3-large` → upserts to Pinecone.

**Check the index exists first:**
```bash
python - <<'EOF'
from pinecone import Pinecone
import os
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
print([i.name for i in pc.list_indexes()])
EOF
```

**Create the index if it doesn't exist:**
```bash
python - <<'EOF'
from pinecone import Pinecone, ServerlessSpec
import os
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc.create_index(
    name="hotpotqa-ragbench-mini",
    dimension=3072,     # must match text-embedding-3-large
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
print("Index created")
EOF
```

---

## Step 4 — Run the Evaluation

### Part A — Model Training mode

```bash
python 3_eval.py --mode a
```

Logs to MLflow Model Training tab:
- **Params:** embedding_model, judge_model, top_k, chunk_size
- **Aggregate metrics:** avg_precision, avg_recall, avg_ndcg, avg_hit_rate, avg_faithfulness, avg_context_relevance, avg_answer_relevance, avg_rag_triad, avg_completeness, avg_conciseness, avg_exact_match, avg_token_f1
- **Step metrics:** per-example time-series charts
- **Artifacts:** `eval_results.jsonl`, `judge_reasoning.json`

### Part B — GenAI mode

```bash
python 3_eval.py --mode b
```

Logs to MLflow GenAI tab:
- **Traces:** CHAIN → RETRIEVER → LLM waterfall per example
- **Scorer results:** 11 scorers × 30 examples in a table

### Run both

```bash
python 3_eval.py --mode both
```

---

## Step 5 — Register a Model (optional)

After finding a good Part A run in the UI:

```bash
python - <<'EOF'
from 3_eval import register_best_model
register_best_model(
    run_id="paste-run-id-from-ui-here",
    description="chunk=400, K=5, rag_triad=0.823, faithfulness=0.833"
)
EOF
```

Then promote in the UI: **Model Training → Models → RAG-HotpotQA-Pipeline → version N → Production**

---

## Understanding the Metrics

Our pipeline computes **three families of metrics**. Each family answers a different question about your RAG system.

```
┌─────────────────────────────────────────────────────────────────┐
│  Family 1 — Retrieval Metrics   Did Pinecone return the right   │
│                                 chunks? (pure Python, free)     │
├─────────────────────────────────────────────────────────────────┤
│  Family 2 — Lexical Metrics     Does the answer match the       │
│                                 reference word-for-word?        │
│                                 (pure Python, free)             │
├─────────────────────────────────────────────────────────────────┤
│  Family 3 — LLM-as-Judge        Does the answer make sense,     │
│                                 avoid hallucination, and        │
│                                 cover all key points?           │
│                                 (OpenAI API calls, costs money) │
└─────────────────────────────────────────────────────────────────┘
```

---

## Family 1 — Retrieval Metrics

These metrics evaluate the **retriever step only** — they tell you how well Pinecone is finding the right chunks before the LLM even sees them.

> **If these come back 0.0, do not shrug it off.** It means the vector IDs in Pinecone
> do not match the `context_ids` in the golden dataset, so nothing can ever count as a
> hit. Check three things in order: that `2_ingest.py` finished, that `PINECONE_INDEX`
> and `PINECONE_NS` are the same in both scripts, and that a retrieved ID looks like
> `5a732ab25542991f29ee2d25_d0`. All 30 golden questions have all 4 of their context
> IDs present in the corpus, so a zero here is a wiring problem, not a data problem.

---

### Precision@K

**What it asks:** Of the K chunks I retrieved, how many were actually relevant?

**Formula:**
```
Precision@K = (Number of relevant chunks retrieved) / K

Example:
  K = 5 (we retrieved 5 chunks)
  2 of those 5 chunks were actually relevant
  Precision@5 = 2/5 = 0.40
```

**Think of it this way:** Imagine you searched Google and got 10 results. If 7 of those 10 results were useful, your Precision = 7/10 = 0.70. If only 2 were useful, Precision = 2/10 = 0.20. **Precision measures how clean your results are.**

| Score | What it means |
|-------|--------------|
| 1.0 | Every single retrieved chunk was relevant — perfect retrieval |
| 0.5 | Half the chunks were relevant, half were noise |
| 0.0 | None of the retrieved chunks were relevant |

**What to do when Precision is LOW:**
- Your retriever is returning noisy, off-topic documents
- Try reducing `TOP_K` (retrieve fewer but better chunks)
- Try a better embedding model
- Try adding metadata filters in Pinecone to narrow the search space
- Check if your chunks are too large — big chunks mix relevant and irrelevant content

---

### Recall@K

**What it asks:** Of ALL the relevant chunks that exist, how many did I manage to retrieve?

**Formula:**
```
Recall@K = (Number of relevant chunks retrieved) / (Total relevant chunks that exist)

Example:
  There are 4 relevant chunks for this question in the database
  We retrieved 5 chunks total, and 3 of them were relevant
  Recall@5 = 3/4 = 0.75
```

**Think of it this way:** Imagine there are 10 doctors in your city. You're trying to find all of them. If you found 8 out of 10, your Recall = 8/10 = 0.80. **Recall measures how complete your search was.**

| Score | What it means |
|-------|--------------|
| 1.0 | You found every relevant chunk that exists — nothing missed |
| 0.5 | You found half the relevant chunks, missed the other half |
| 0.0 | You missed all relevant chunks completely |

**What to do when Recall is LOW:**
- Increase `TOP_K` to retrieve more chunks (retrieves more, catches more relevant ones)
- Your chunks may be too small — key information split across multiple chunks that all need to be retrieved
- Your embedding model may not be capturing the semantic meaning well enough
- Consider hybrid search (keyword + semantic) to catch exact-match cases

---

### The Precision vs Recall Tradeoff

**This is one of the most important concepts in information retrieval.**

Precision and Recall are in constant tension with each other:

```
If you increase TOP_K (retrieve more chunks):
  ✅ Recall goes UP   — you catch more relevant chunks
  ❌ Precision goes DOWN — you also pick up more irrelevant noise

If you decrease TOP_K (retrieve fewer chunks):
  ✅ Precision goes UP  — fewer but cleaner results
  ❌ Recall goes DOWN   — you might miss some relevant chunks
```

**Real-world example:**
```
Question: "Who invented the telephone?"

TOP_K = 2:
  Retrieved: ["Alexander Graham Bell invented the telephone in 1876",
              "Bell was born in Scotland"]
  Precision = 1.0 (both relevant), Recall = 0.5 (missed other relevant chunks)

TOP_K = 10:
  Retrieved: ["Alexander Graham Bell...", "Bell was born...",
              "Thomas Edison invented the phonograph",   ← noise
              "The telephone changed communication",
              "Elisha Gray also claimed invention",
              ...5 more mostly irrelevant chunks]
  Precision = 0.4 (4 of 10 relevant), Recall = 1.0 (found everything)
```

**The sweet spot for RAG:** You want high enough Recall that the LLM has all the information it needs, but not so many chunks that the LLM gets confused by noise. `TOP_K = 5` is a good starting point for most use cases.

---

### F1 Score (Harmonic Mean of Precision and Recall)

**What it asks:** Can I get a single number that balances both Precision and Recall?

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Example:
  Precision = 0.8, Recall = 0.6
  F1 = 2 × (0.8 × 0.6) / (0.8 + 0.6)
     = 2 × 0.48 / 1.4
     = 0.96 / 1.4
     = 0.686
```

**Why harmonic mean instead of regular average?**

The regular average (arithmetic mean) of 0.8 and 0.6 = 0.70.
The harmonic mean = 0.686.

They seem close, but the difference matters at extremes:

```
Extreme example:
  Precision = 1.0, Recall = 0.0
  Arithmetic mean = (1.0 + 0.0) / 2 = 0.50  ← misleadingly high!
  Harmonic mean   = 2 × (1.0 × 0.0) / (1.0 + 0.0) = 0.0  ← correctly zero

Why 0.0 is correct here:
  Recall = 0.0 means we found NONE of the relevant documents.
  A retriever that finds nothing useful should score 0, not 0.5.
  The harmonic mean punishes extreme imbalance — which is exactly what we want.
```

**Think of it this way:** F1 is like judging a student who is brilliant at one thing but completely fails another. You shouldn't give them a B just because they averaged out. The harmonic mean gives a fairer, harsher penalty for weakness in either dimension.

| Score | What it means |
|-------|--------------|
| 1.0 | Perfect Precision AND perfect Recall |
| 0.7+ | Good balance — solid retrieval |
| 0.4–0.7 | Moderate — room for improvement |
| < 0.4 | Poor — either missing relevant docs or returning too much noise |

---

### Hit Rate

**What it asks:** Did we retrieve at least ONE relevant chunk? (Yes or No)

**Formula:**
```
Hit Rate = 1.0 if any retrieved chunk is relevant, else 0.0

Example:
  Retrieved 5 chunks. 1 of them is relevant.
  Hit Rate = 1.0  ← at least one hit is enough

  Retrieved 5 chunks. 0 of them are relevant.
  Hit Rate = 0.0
```

**Why this matters:** Even if Precision and Recall are imperfect, if the LLM has at least one relevant chunk in the context window, it has a chance of producing a good answer. Hit Rate measures whether the pipeline is capable of being useful at all.

**Think of it like a search engine minimum bar:** Did the search engine find even ONE useful result? If Hit Rate is consistently below 0.5, your retriever is fundamentally broken for more than half your queries.

| Score | What it means |
|-------|--------------|
| 1.0 | At least one relevant chunk was found — pipeline has a chance |
| 0.0 | No relevant chunks found — answer will likely be hallucinated or "I don't know" |

**What to do when Hit Rate is LOW:**
- This is more serious than low Precision or Recall
- Your embedding model may be poorly aligned with the document domain
- Check if documents were chunked correctly during ingestion
- Verify the Pinecone namespace is correct
- Try a higher `TOP_K` as a quick fix

---

### nDCG (Normalised Discounted Cumulative Gain)

**What it asks:** Did we retrieve relevant chunks AND did we rank them highly (position 1 > position 5)?

**Why ranking matters:** The LLM reads your chunks in order. If the most relevant chunk is at position 5 but less relevant chunks are at positions 1-4, the LLM may not weight it as strongly. Good retrieval means putting the best chunks at the top.

**Building up to the formula step by step:**

**Step 1 — Cumulative Gain (CG):** Just count how many relevant chunks you retrieved (ignores position).
```
CG = 1 + 1 + 0 + 0 + 1 = 3   (relevant, relevant, not, not, relevant)
```

**Step 2 — Discounted Cumulative Gain (DCG):** Give less credit to relevant chunks found at lower positions.
```
DCG = Σ relevance(i) / log₂(position + 1)

Position 1 is worth: 1 / log₂(2) = 1 / 1.0 = 1.00
Position 2 is worth: 1 / log₂(3) = 1 / 1.58 = 0.63
Position 3 is worth: 1 / log₂(4) = 1 / 2.0  = 0.50
Position 4 is worth: 1 / log₂(5) = 1 / 2.32 = 0.43
Position 5 is worth: 1 / log₂(6) = 1 / 2.58 = 0.39

So a relevant doc at position 1 earns 1.00, but at position 5 earns only 0.39.
The discount is logarithmic — the penalty drops sharply from pos 1 to 2,
then more slowly from pos 3 onward.
```

**Step 3 — Ideal DCG (IDCG):** What would DCG be if all relevant docs were at the top?
```
If you have 2 relevant docs, ideal ranking puts them at positions 1 and 2:
IDCG = 1.00 + 0.63 = 1.63
```

**Step 4 — nDCG:** Normalise by the ideal so score is always between 0 and 1.
```
nDCG = DCG / IDCG

Example:
  Retrieved: [relevant, not-relevant, relevant, not-relevant, not-relevant]
  DCG  = 1.00 + 0 + 0.50 + 0 + 0 = 1.50
  IDCG = 1.00 + 0.63 = 1.63  (ideal: both relevant docs at top 2 positions)
  nDCG = 1.50 / 1.63 = 0.92  ← good, but not perfect because pos 3 > pos 2
```

| Score | What it means |
|-------|--------------|
| 1.0 | All relevant chunks are at the very top positions — perfect ranking |
| 0.7+ | Relevant chunks are near the top — good ranking |
| 0.4–0.7 | Relevant chunks are scattered through the results |
| < 0.4 | Relevant chunks are buried at the bottom — poor ranking |

**What to do when nDCG is LOW but Hit Rate is HIGH:**
- You're finding relevant chunks but ranking them poorly
- Try reranking: run a cross-encoder reranker after Pinecone retrieval to re-order results
- Experiment with different similarity metrics (cosine vs dot product)
- Review your chunking strategy — smaller, more focused chunks tend to rank better

---

## Family 2 — Lexical Metrics

These metrics compare the generated answer directly against the gold reference answer using **word matching only**. No model calls needed — they are fast and free.

> **Important limitation:** Lexical metrics penalise correct paraphrases.
> 
> Reference: *"Alexander Graham Bell invented the telephone"*  
> Generated: *"The telephone was invented by Bell"*  
> 
> Both are correct, but exact match = 0.0 because the words aren't in the same order.
> This is why we always use lexical metrics **alongside** LLM-judge metrics.

---

### Exact Match (EM)

**What it asks:** Is the generated answer **word-for-word identical** to the reference answer (after cleaning)?

**Formula:**
```
Exact Match = 1.0 if clean(generated) == clean(reference) else 0.0

Cleaning means: lowercase + remove punctuation + collapse whitespace

Example:
  Reference:  "Daniel Ricciardo won the race."
  Generated:  "daniel ricciardo won the race"
  After clean: "daniel ricciardo won the race" == "daniel ricciardo won the race"
  Exact Match = 1.0  ✅

  Reference:  "Daniel Ricciardo won the race."
  Generated:  "The race was won by Daniel Ricciardo."
  After clean: "the race was won by daniel ricciardo" ≠ "daniel ricciardo won the race"
  Exact Match = 0.0  ❌ (even though it's correct!)
```

**When is Exact Match useful?**
- For short factual answers (names, dates, numbers) — e.g. "Who won?" → "Daniel Ricciardo"
- As a lower bound: if EM is high, the pipeline is producing very precise answers
- Not useful for paragraph-length answers where paraphrase is expected

**What to do when Exact Match is LOW:**
- Don't panic — this is almost always expected for longer answers
- Check if Token F1 is high — if EM=0 but F1=0.8, the answer is correct but paraphrased
- If both EM and F1 are low, check LLM-judge scores to understand why

---

### Token F1

**What it asks:** How much word overlap is there between the generated answer and the reference answer?

**Formula:**
```
1. Tokenise both answers (split into words after cleaning)
2. Count overlapping tokens (intersection of word counts)
3. Precision = overlap / generated_word_count
4. Recall    = overlap / reference_word_count
5. Token F1  = 2 × (Precision × Recall) / (Precision + Recall)

Example:
  Reference:  "Daniel Ricciardo won the 44-lap race for Red Bull Racing"
  Generated:  "The 44-lap race was won by Daniel Ricciardo for Red Bull"

  Reference tokens:  {daniel:1, ricciardo:1, won:1, the:1, 44:1, lap:1,
                       race:1, for:1, red:1, bull:1, racing:1}  → 11 tokens
  Generated tokens:  {the:1, 44:1, lap:1, race:1, was:1, won:1,
                       by:1, daniel:1, ricciardo:1, for:1, red:1, bull:1} → 12 tokens

  Overlap: {daniel, ricciardo, won, the, 44, lap, race, for, red, bull} = 10 tokens

  Precision = 10/12 = 0.833   (10 of 12 generated words matched the reference)
  Recall    = 10/11 = 0.909   (10 of 11 reference words appeared in generated)
  Token F1  = 2 × (0.833 × 0.909) / (0.833 + 0.909) = 0.869
```

**Why Token F1 is better than Exact Match for longer answers:**
- Gives partial credit — an answer that contains most of the right words scores well
- Order-independent — "Bell invented telephone" and "telephone invented Bell" score equally
- Standard benchmark metric used in SQuAD and HotpotQA academic papers

| Score | What it means |
|-------|--------------|
| 0.9+ | Near-perfect word overlap — answer is almost identical to reference |
| 0.7–0.9 | Good overlap — answer captures most key words |
| 0.5–0.7 | Moderate overlap — answer partially correct or paraphrased heavily |
| < 0.5 | Low overlap — answer may be wrong or very differently worded |

**What to do when Token F1 is LOW:**
- First check if LLM-judge scores are high — if yes, the answer is correct but heavily paraphrased (Token F1 limitation, not a pipeline problem)
- If LLM-judge scores are also low, the answer itself is wrong
- Review `judge_reasoning.json` artifacts to see what the judge thinks is missing

---

## Family 3 — LLM-as-Judge Metrics

These metrics use GPT (`JUDGE_MODEL`) to evaluate quality dimensions that pure word-matching cannot capture. Each judge reads the question, answer, and/or context, then returns a score from 0.0 to 1.0 with reasoning.

**How every judge works:**
```
1. We write a carefully crafted prompt describing what to evaluate
2. We ask the judge: "Return JSON with {score: 0.0-1.0, reasoning: ...}"
3. We parse the score and store the reasoning in judge_reasoning.json

Why JSON output? → response_format={"type": "json_object"} forces valid
JSON every time. Without this, the model might wrap output in markdown
fences which breaks json.loads().
```

---

### Context Relevance

**What it asks:** Are the chunks retrieved from Pinecone actually relevant to the question being asked?

**Why it matters:** This is the very first thing that can go wrong in a RAG pipeline. If Pinecone returns the wrong documents, the LLM has nothing useful to work with — no matter how good the generator is.

```
Score = average relevance of each individual chunk to the question

Example:
  Question: "Who invented the telephone?"
  Chunk 1: "Alexander Graham Bell invented the telephone in 1876"  → score 1.0
  Chunk 2: "Bell was born in Edinburgh, Scotland in 1847"          → score 0.7
  Chunk 3: "Thomas Edison invented the phonograph, not telephone"  → score 0.3
  Chunk 4: "The history of electricity spans centuries"            → score 0.1
  Chunk 5: "Graham crackers were invented by Sylvester Graham"     → score 0.0

  Average = (1.0 + 0.7 + 0.3 + 0.1 + 0.0) / 5 = 0.42
```

| Score | What it means | What to do |
|-------|--------------|------------|
| 0.8+ | Pinecone is returning highly relevant chunks | ✅ No action needed |
| 0.5–0.8 | Mix of relevant and noisy chunks | Consider reducing TOP_K or improving chunking |
| < 0.5 | Pinecone is mostly returning off-topic documents | 🔴 Fix retrieval first before tuning anything else |

**This is the first leg of the RAG Triad.**

---

### Faithfulness

**What it asks:** Does the generated answer only contain facts that are present in the retrieved context? Or is the model making things up?

**Why it matters:** This is your **hallucination detector**. In a RAG pipeline, the model should only answer from what it retrieved. If it uses knowledge it learned during training instead of the context, you lose source traceability — you can no longer say "the answer came from document X."

```
Score = supported_claims / total_claims_in_answer

Example:
  Context: ["Bell invented the telephone in 1876",
            "Bell was born in Scotland"]

  Answer A: "Alexander Graham Bell, who was born in Scotland, invented
             the telephone in 1876."
  → All 3 claims supported by context
  → Faithfulness = 3/3 = 1.0  ✅ No hallucination

  Answer B: "Alexander Graham Bell invented the telephone in 1876.
             He held over 300 patents during his lifetime."
  → Claim 1 supported ✅, Claim 2 NOT in context ❌
  → Faithfulness = 1/2 = 0.50  ⚠️ Partial hallucination

  Answer C: "Bell invented the telephone. He was American and attended
             MIT before his famous experiment."
  → Claim 1 supported ✅, Claims 2-3 hallucinated ❌❌
  → Faithfulness = 1/3 = 0.33  🔴 Significant hallucination
```

| Score | What it means | What to do |
|-------|--------------|------------|
| 0.9+ | Model is staying grounded — almost no hallucination | ✅ Healthy |
| 0.7–0.9 | Occasional unsupported claims | Review examples in judge_reasoning.json |
| < 0.7 | Frequent hallucination | 🔴 Strengthen system prompt — "ONLY use the provided context" |

**This is the second leg of the RAG Triad.**

---

### Answer Relevance

**What it asks:** Does the answer actually address the question that was asked?

**Why it matters:** A model can be perfectly faithful (never hallucinate) but still give an irrelevant answer. For example, if you ask "Who invented the telephone?" and the model responds "The telephone was invented during the industrial revolution" — that's technically not wrong, but it doesn't answer the question.

```
Score interpretation:
  1.0 = answer directly and completely addresses the question
  0.5 = answer is related but only partially answers the question
  0.0 = answer is off-topic, refuses to answer, or addresses a different question

Examples:
  Q: "Who won the 44-lap race for Red Bull Racing?"
  A: "Daniel Ricciardo won the 44-lap race for Red Bull."  → 1.0  ✅
  A: "Red Bull Racing is a Formula 1 team."               → 0.2  ❌ (related but wrong)
  A: "I cannot find that information in the context."     → 0.0  ❌ (refuses)
```

| Score | What it means | What to do |
|-------|--------------|------------|
| 0.8+ | Answer directly addresses the question | ✅ Good |
| 0.5–0.8 | Partial answer or slight topic drift | Improve generator prompt to be more direct |
| < 0.5 | Answer is off-topic or refusing | 🔴 Check if context relevance is also low (root cause = bad retrieval) |

**This is the third leg of the RAG Triad.**

---

### Completeness

**What it asks:** Does the answer cover ALL the key points from the gold reference answer, or is it missing important information?

**Why it matters:** A short answer can be both faithful AND relevant, but still miss crucial details. Completeness catches answers that are technically correct but incomplete.

```
Score = fraction of reference key points present in the generated answer

Example:
  Reference: "Governor John R. Rogers High School is located in the
              Puyallup School District of Washington, United States."

  Generated A: "It is in the Puyallup School District of Washington."
  → Missing: "Governor John R. Rogers High School" (the subject)
  → Completeness ≈ 0.6  (partial)

  Generated B: "Governor John R. Rogers High School is in the
                Puyallup School District, Washington, USA."
  → All key points covered: school name, district, state
  → Completeness ≈ 0.95
```

| Score | What it means | What to do |
|-------|--------------|------------|
| 0.9+ | Answer covers everything in the reference | ✅ |
| 0.6–0.9 | Mostly complete, minor omissions | Check if retrieved chunks contain the missing facts |
| < 0.6 | Significant gaps | Increase TOP_K, or check if relevant chunks were retrieved at all |

**Key insight:** If Completeness is low but Context Relevance is high, the LLM is not extracting all the information from the context. If Context Relevance is also low, the problem is in retrieval — the chunks with the missing information were never retrieved.

---

### Conciseness

**What it asks:** Is the answer short and focused, or is it padded with unnecessary words?

**Why it matters:** In production RAG systems (chatbots, voice assistants, document Q&A), verbose answers are a poor user experience. Also, models that pad their answers with filler text ("That's a great question! Let me explain this in detail...") are often uncertain and hiding behind verbosity.

```
Score interpretation:
  1.0 = perfectly concise — every sentence adds value
  0.5 = somewhat padded — a few unnecessary phrases
  0.0 = excessively verbose — the answer could be 80% shorter

Examples:
  Q: "Who invented the telephone?"

  Concise (score ~1.0):
    "Alexander Graham Bell invented the telephone in 1876."

  Moderate (score ~0.6):
    "Based on the provided context, it appears that the telephone was
     invented by Alexander Graham Bell. Bell is credited with this
     invention, which occurred in the year 1876."

  Verbose (score ~0.2):
    "That's a wonderful question about the history of communication
     technology. According to the information I have been provided,
     and based on careful analysis of the available context, I can
     inform you that the telephone, which is a device used for voice
     communication over distances, was invented by Alexander Graham
     Bell in the year 1876. Bell, whose full name was Alexander Graham
     Bell, is widely credited with this important invention..."
```

**What to do when Conciseness is LOW:**
- Add a length instruction to your system prompt: *"Answer in 1-3 sentences maximum."*
- Use a lower temperature (0 is already set in our code — good)
- Check if context chunks are too large — models tend to ramble when given too much context

---

## The RAG Triad — The Most Important Composite Score

The RAG Triad combines three metrics into one headline score that tells you the overall health of your pipeline.

```
RAG Triad = f(Context Relevance, Faithfulness, Answer Relevance)
```

### Why these three?

```
Step 1 — Did we RETRIEVE the right information?
         ↓ measured by Context Relevance

Step 2 — Did we USE that information without hallucinating?
         ↓ measured by Faithfulness

Step 3 — Did the final answer ADDRESS the question?
         ↓ measured by Answer Relevance
```

If any one of these three fails, the pipeline has a fundamental problem regardless of the other two.

### The Formula — Harmonic Mean

```
RAG Triad = 3 × (CR × Faith × AR) / (CR×Faith + Faith×AR + AR×CR)

Where:
  CR    = Context Relevance
  Faith = Faithfulness
  AR    = Answer Relevance
```

### Why harmonic mean instead of simple average?

```
Simple average example (misleading):
  CR = 1.0, Faith = 1.0, AR = 0.0
  Simple average = (1.0 + 1.0 + 0.0) / 3 = 0.67  ← sounds ok, but...

  AR = 0.0 means the answer doesn't address the question AT ALL.
  A pipeline that gives useless answers should NOT score 0.67!

Harmonic mean:
  Triad = 3 × (1.0 × 1.0 × 0.0) / (1.0 + 0.0 + 0.0) = 0.0  ✅ Correctly zero

The harmonic mean COLLAPSES to near zero if ANY single leg is near zero.
This is by design — it forces you to fix every weak link, not just average them away.
```

### Real examples:

```
Scenario 1: Good pipeline
  CR = 0.85, Faith = 0.90, AR = 0.88  →  Triad ≈ 0.877

Scenario 2: Bad retrieval, good generation
  CR = 0.20, Faith = 0.95, AR = 0.90  →  Triad ≈ 0.337
  (Retrieval is the bottleneck — fix Pinecone first)

Scenario 3: Good retrieval, hallucinating generator
  CR = 0.90, Faith = 0.25, AR = 0.85  →  Triad ≈ 0.413
  (Fix the system prompt to ground the model in context)

Scenario 4: Everything balanced but mediocre
  CR = 0.60, Faith = 0.60, AR = 0.60  →  Triad = 0.600
  (Uniform improvement needed across all three)
```

### How to use the RAG Triad in practice:

```
1. Start with the RAG Triad as your headline metric
2. If Triad is LOW, look at the three individual legs:
   - Context Relevance is lowest → fix retrieval (Pinecone, chunking, TOP_K)
   - Faithfulness is lowest     → fix generator prompt (stronger grounding instruction)
   - Answer Relevance is lowest → fix generator prompt (be more direct)
3. Never try to fix all three at once — change one thing at a time
4. Re-run the evaluation after each change and compare runs in MLflow
```

---

## Metric Quick Reference Card

| Metric | Family | What it measures | Range | Good score |
|--------|--------|-----------------|-------|-----------|
| Precision@K | Retrieval | % of retrieved chunks that are relevant | 0–1 | > 0.7 |
| Recall@K | Retrieval | % of relevant chunks that were retrieved | 0–1 | > 0.7 |
| F1@K | Retrieval | Harmonic mean of Precision and Recall | 0–1 | > 0.7 |
| Hit Rate | Retrieval | Did we find at least 1 relevant chunk? | 0 or 1 | = 1.0 |
| nDCG@K | Retrieval | Are relevant chunks ranked highly? | 0–1 | > 0.7 |
| Exact Match | Lexical | Word-for-word match with reference | 0 or 1 | — (supplementary) |
| Token F1 | Lexical | Word overlap with reference | 0–1 | > 0.6 |
| Context Relevance | LLM Judge | Are retrieved chunks relevant to the query? | 0–1 | > 0.75 |
| Faithfulness | LLM Judge | No hallucination — grounded in context | 0–1 | > 0.85 |
| Answer Relevance | LLM Judge | Answer addresses the question | 0–1 | > 0.80 |
| **RAG Triad** | **Composite** | **Overall pipeline health** | **0–1** | **> 0.75** |
| Completeness | LLM Judge | Covers all reference key points | 0–1 | > 0.75 |
| Conciseness | LLM Judge | Answer is short and focused | 0–1 | > 0.70 |

---

## Diagnostic Guide — What to Do When Scores Are Low

Use this guide to find the root cause and fix it:

```
┌─────────────────────────────────────────────────────────────────────┐
│  RAG Triad is LOW                                                   │
│  → Look at which individual leg is lowest                           │
└──────────────────────┬──────────────────────┬───────────────────────┘
                       │                      │                      │
            Context Relevance           Faithfulness          Answer Relevance
              is lowest                  is lowest               is lowest
                  │                          │                       │
    ┌─────────────▼──────────┐  ┌────────────▼───────────┐  ┌───────▼──────────────┐
    │  RETRIEVAL PROBLEM     │  │  GENERATION PROBLEM    │  │  RELEVANCE PROBLEM   │
    │                        │  │                        │  │                      │
    │  • Increase TOP_K      │  │  • Strengthen system   │  │  • Be more direct    │
    │  • Try better embedding│  │    prompt grounding    │  │    in system prompt  │
    │  • Smaller chunks      │  │  • "ONLY use context"  │  │  • Check if context  │
    │  • Check namespace     │  │  • Add citation        │  │    relevance is OK   │
    │  • Hybrid search       │  │    requirement         │  │    (chain reaction)  │
    └────────────────────────┘  └────────────────────────┘  └──────────────────────┘
```

### Pattern: Low Precision, High Recall
```
Pinecone is finding relevant docs but also returning a lot of noise.
Fix: Reduce TOP_K, improve chunking to be more focused, add metadata filters.
```

### Pattern: High Precision, Low Recall
```
Pinecone returns clean results but misses relevant docs.
Fix: Increase TOP_K, use larger chunk size so key facts aren't split.
```

### Pattern: High Context Relevance, Low Faithfulness
```
Relevant chunks retrieved but model is still hallucinating.
Fix: The system prompt is not strict enough. Add: "Do NOT use any knowledge
outside of the provided context. If the context doesn't contain the answer,
say 'I cannot find this in the provided documents'."
```

### Pattern: High Faithfulness, Low Completeness
```
Model is staying grounded but giving incomplete answers.
Fix: Increase TOP_K to retrieve more chunks. The missing information
exists in the database but wasn't retrieved.
```

### Pattern: High everything except Conciseness
```
Pipeline is accurate but answers are too verbose.
Fix: Add "Answer in 1-3 sentences." to your system prompt.
```

### Pattern: Low Token F1, High LLM Judge scores
```
This is NORMAL — not a problem. The answer is correct but phrased
differently from the reference. Trust the LLM judge over Token F1
for meaning-based evaluation.
```

---

## Project Structure

```
.
├── 1_prepare_dataset.py          # Download + build golden dataset
├── 2_ingest.py                   # Chunk + embed + upsert to Pinecone
├── 3_eval.py                     # Evaluate — Part A + Part B
├── .env                          # API keys and config (never commit)
├── .gitignore
├── README.md
├── data/    # Generated by 1_prepare_dataset.py
│   ├── golden_hotpotqa_30.jsonl
│   └── corpus_hotpotqa.jsonl
├── mlflow-artifacts/             # Local artifact store (Option B only)
├── mlflow.db                     # SQLite DB (Option B only)
└── mlflow_server.log             # Server log (background mode)
```

---

## MLflow UI Quick Reference

| What you want | Where to find it |
|---|---|
| All experiments | http://localhost:5001 → Model Training tab |
| Part A run metrics | Model Training → RAG-HotpotQA-Evaluation → click run |
| Compare two runs | Select runs with checkboxes → Compare button |
| Step charts (per-example) | Any run → Metrics tab → click a metric name |
| Part B traces | GenAI tab → Traces → click any trace → waterfall view |
| Part B scorer results | GenAI tab → Evaluation Results → per-row table |
| Download artifacts | Any run → Artifacts tab → click file |
| Model Registry | Model Training tab → Models |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `403 Forbidden` with empty body | `MLFLOW_TRACKING_URI` set to `databricks` from old session | `unset MLFLOW_TRACKING_URI` then re-run |
| `could not translate host name "123@localhost"` | `@` in password not URL-encoded | Replace `@` with `%40` in the connection URI |
| `ImportError: cannot import name 'load_dataset'` | Local file named `datasets` shadows the package | Rename conflicting local file/folder |
| `Connection refused at localhost:5001` | MLflow server not running | Start server in a separate terminal (Step 1) |
| `PINECONE_API_KEY not set` | Env vars not loaded | Run `export $(cat .env \| xargs)` |
| `Index not found` | Pinecone index doesn't exist | Create index (see Step 3) |
| `openai.AuthenticationError` | Invalid or missing API key | Check `OPENAI_API_KEY` env var |
| `Registry tab not visible` | Using file:// backend | Switch to server mode (`http://localhost:5001`) |
| `Metrics show as NaN` | Non-numeric value passed to log_metrics | Wrap metric values in `float()` |
| Port 5001 already in use | Another process on that port | `kill $(lsof -t -i:5001)` then restart |
| Retrieval metrics all 0.0 | Pinecone IDs ≠ golden JSONL IDs | Expected — see note in Retrieval Metrics section |

---

## .gitignore

```
.env
*.db
mlflow-artifacts/
mlflow_server.log
data/
__pycache__/
.venv/
*.pyc
eval_results*.jsonl
judge_reasoning*.json
```