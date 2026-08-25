"""
╔══════════════════════════════════════════════════════════════════════════╗
║          RAG Evaluation Pipeline — MLflow 3 (GenAI + Model Training)    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  DATASET  RAGBench HotpotQA (30 golden examples)                        ║
║    input       → question / query                                        ║
║    reference   → gold answer                                             ║
║    contexts    → list[str]  gold context chunks (4 per example)          ║
║    context_ids → list[str]  corpus-level IDs for Recall/Precision/F1    ║
║    id          → record identifier                                       ║
║                                                                          ║
║  TWO EVALUATION MODES  (select via --mode flag)                          ║
║                                                                          ║
║  Part A  →  Model Training mode                                          ║
║             Hand-coded metrics computed in pure Python + LLM judges.    ║
║             Logged via mlflow.log_params() + mlflow.log_metrics().       ║
║             Visible in MLflow UI: Model Training tab                     ║
║                                                                          ║
║  Part B  →  GenAI mode                                                   ║
║             @mlflow.trace decorates the pipeline → every call creates   ║
║             a waterfall trace (CHAIN → RETRIEVER → LLM) in the UI.      ║
║             mlflow.genai.evaluate() runs scorers over a DataFrame.      ║
║             Visible in MLflow UI: GenAI tab → Traces + Eval Results     ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PART A — LLM CALL OPTIMISATIONS                                        ║
║                                                                          ║
║  BEFORE (original)                                                       ║
║    score_context_relevance  → 1 call per chunk × TOP_K = 5 calls        ║
║    score_faithfulness       → 1 call                                    ║
║    score_answer_relevance   → 1 call                                    ║
║    score_completeness       → 1 call                                    ║
║    score_conciseness        → 1 call                                    ║
║    Total judge calls        → ~9 per example × 30 = ~270 calls          ║
║                                                                          ║
║  AFTER (optimised)                                                       ║
║    score_context_relevance_batched → 1 call for ALL chunks at once      ║
║    score_all_judges_combined       → 1 call for faith + AR + comp + con ║
║    Total judge calls               → 2 per example × 30 = 60 calls     ║
║    + ThreadPoolExecutor            → 30 examples run in parallel        ║
║                                                                          ║
║  Result: ~77% fewer LLM calls, wall-clock time cut by parallelism       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PREREQUISITES                                                           ║
║                                                                          ║
║  1. MLflow server running:                                               ║
║       mlflow server                                                      ║
║         --backend-store-uri postgresql://user:pass%40w@host:5432/db     ║
║         --default-artifact-root s3://your-bucket/mlflow-artifacts       ║
║         --host 0.0.0.0 --port 5001                                       ║
║                                                                          ║
║  2. Environment variables:                                               ║
║       export OPENAI_API_KEY="sk-..."                                     ║
║       export PINECONE_API_KEY="pcn-..."                                  ║
║                                                                          ║
║  3. golden_hotpotqa_30.jsonl  →  run 1_prepare_dataset.py first         ║
║  4. Pinecone index populated  →  run 2_ingest.py first                  ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  RUN COMMANDS                                                            ║
║    python 3_eval.py              # Part A only (default)                 ║
║    python 3_eval.py --mode a     # Part A explicitly                     ║
║    python 3_eval.py --mode b     # Part B (GenAI mode)                   ║
║    python 3_eval.py --mode both  # Part A then Part B sequentially       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone

# ── MLflow core ───────────────────────────────────────────────────────────
import mlflow
import mlflow.genai

# ── MLflow GenAI scorers ──────────────────────────────────────────────────
from mlflow.genai.scorers import (
    scorer,
    Correctness,
    RelevanceToQuery,
    Safety,
)


# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

MLFLOW_URL        = "http://localhost:5001"
EXPERIMENT_A_NAME = "RAG-HotpotQA-Evaluation-v5"
EXPERIMENT_B_NAME = "RAG-HotpotQA-GenAI-Eval-v5"

PINECONE_INDEX    = "hotpotqa-ragbench-mini"
PINECONE_NS       = "hotpotqa"
EMBEDDING_MODEL   = "text-embedding-3-large"
GENERATOR_MODEL   = "gpt-4o"
JUDGE_MODEL       = "gpt-4o"
TOP_K             = 5
CHUNK_SIZE        = 400

# Part A parallelism — how many examples to evaluate concurrently.
# Each worker makes ~4 LLM calls so stay within OpenAI rate limits.
# Lower this if you hit 429 errors; raise it on a paid tier.
PART_A_WORKERS    = 5

GOLDEN_FILE       = Path("data/golden_hotpotqa_30.jsonl")

mlflow.set_tracking_uri(MLFLOW_URL)


# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CLIENTS
# ══════════════════════════════════════════════════════════════════════════

oai_client = None   # set in main()
pine_index = None   # set in main()


# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════

def load_golden_data(filepath):
    """Load the 30 golden examples and normalise field names."""
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            raw = json.loads(line)
            examples.append({
                "id":          raw["id"],
                "question":    raw["input"],
                "answer":      raw["reference"],
                "contexts":    raw["contexts"],
                "context_ids": raw["context_ids"],
            })
    log.info(f"Loaded {len(examples)} examples")
    return examples


# ══════════════════════════════════════════════════════════════════════════
# RAG PIPELINE — Retrieve + Generate
# ══════════════════════════════════════════════════════════════════════════

@mlflow.trace(span_type="RETRIEVER")
def retrieve_chunks(question):
    embed_response  = oai_client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    question_vector = embed_response.data[0].embedding
    search_result   = pine_index.query(
        vector=question_vector,
        top_k=TOP_K,
        namespace=PINECONE_NS,
        include_metadata=True
    )
    doc_ids = [match.id for match in search_result.matches]
    chunks  = [match.metadata.get("text", "") for match in search_result.matches]
    return doc_ids, chunks


@mlflow.trace(span_type="LLM")
def generate_answer(question, chunks):
    context_text = "".join(f"[{i+1}] {chunk}\n\n" for i, chunk in enumerate(chunks))
    response = oai_client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY using the context provided. "
                    "If the context lacks the answer, say so explicitly. "
                    "Be concise — 1 to 3 sentences."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content.strip()


@mlflow.trace(span_type="CHAIN")
def run_rag_pipeline(question):
    doc_ids, chunks = retrieve_chunks(question)
    answer          = generate_answer(question, chunks)
    return {"answer": answer, "doc_ids": doc_ids, "chunks": chunks}


# ══════════════════════════════════════════════════════════════════════════
# METRIC FAMILY 1 — RETRIEVAL METRICS (pure Python, no API calls)
# ══════════════════════════════════════════════════════════════════════════

def compute_retrieval_metrics(retrieved_ids, relevant_ids):
    """
    Precision@K, Recall@K, F1@K, Hit Rate, nDCG@K.

    Match logic: exact equality.
    No sub-chunking is applied during ingestion, so every Pinecone vector ID
    is exactly the corpus record ID ("{example_id}_d{doc_index}"), which is
    also what context_ids stores in the golden dataset.
    """
    relevant_set = set(relevant_ids)
    k            = max(len(retrieved_ids), 1)
    n_relevant   = max(len(relevant_set), 1)

    hit_flags = [rid in relevant_set for rid in retrieved_ids]
    n_hits    = sum(hit_flags)

    precision = n_hits / k
    recall    = n_hits / n_relevant
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    hit_rate  = 1.0 if n_hits > 0 else 0.0

    dcg  = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hit_flags) if h)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "ndcg":      round(ndcg,      4),
        "hit_rate":  round(hit_rate,  4),
    }


# ══════════════════════════════════════════════════════════════════════════
# METRIC FAMILY 2 — LEXICAL METRICS (pure Python, no API calls)
# ══════════════════════════════════════════════════════════════════════════

def clean_text(text):
    result = text.lower()
    return "".join(ch for ch in result if ch.isalnum() or ch.isspace()).strip()


def compute_exact_match(generated, reference):
    return 1.0 if clean_text(generated) == clean_text(reference) else 0.0


def compute_token_f1(generated, reference):
    gen_tokens = Counter(clean_text(generated).split())
    ref_tokens = Counter(clean_text(reference).split())
    overlap    = sum((gen_tokens & ref_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(gen_tokens.values())
    recall    = overlap / sum(ref_tokens.values())
    return round(2 * precision * recall / (precision + recall), 4)


# ══════════════════════════════════════════════════════════════════════════
# METRIC FAMILY 3 — LLM-AS-JUDGE METRICS  ★ OPTIMISED ★
#
# BEFORE: 9 separate LLM calls per example
#   • score_context_relevance → 1 call PER chunk  (TOP_K=5 → 5 calls)
#   • score_faithfulness      → 1 call
#   • score_answer_relevance  → 1 call
#   • score_completeness      → 1 call
#   • score_conciseness       → 1 call
#
# AFTER: 2 LLM calls per example
#   • score_context_relevance_batched  → 1 call scores ALL chunks at once
#   • score_all_judges_combined        → 1 call for all 4 remaining judges
#
# HOW THE BATCH CALLS WORK:
#   We ask the judge to return a JSON object containing ALL scores we need.
#   response_format={"type": "json_object"} guarantees valid JSON every time.
#   A single well-structured prompt is cheaper and faster than N round-trips.
# ══════════════════════════════════════════════════════════════════════════

def ask_judge(prompt):
    """
    Core helper — sends a scoring prompt to JUDGE_MODEL.
    Returns (score: float, reasoning: str).
    Falls back to (0.0, raw_text) on parse failure so evaluation never crashes.
    """
    response = oai_client.chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content
    try:
        result    = json.loads(raw)
        score     = float(result.get("score", 0.0))
        score     = max(0.0, min(1.0, score))
        reasoning = result.get("reasoning", "")
        return score, reasoning
    except Exception:
        return 0.0, raw


# ── OPTIMISED: Batch Context Relevance ───────────────────────────────────────
#
# OLD approach — 1 call per chunk:
#   for chunk in chunks:
#       ask_judge(f"How relevant is this chunk: {chunk}")   ← 5 API calls
#
# NEW approach — 1 call for all chunks:
#   Ask the judge to score ALL chunks in one prompt and return a JSON array.
#   This saves 4 round-trips and halves the total judge call count.

def score_context_relevance_batched(question, chunks):
    """
    Score relevance of ALL retrieved chunks in a single LLM call.

    Returns (avg_score: float, reasoning: str).

    The judge returns:
      {
        "chunk_scores": [0.9, 0.2, 0.8, 0.1, 0.7],
        "reasoning": "Chunks 1,3,5 directly address the question; 2,4 are tangential"
      }
    We average chunk_scores for a single context_relevance float.
    """
    if not chunks:
        return 0.0, "no chunks retrieved"

    # Build a numbered list of truncated chunks for the prompt
    chunks_text = "\n\n".join(
        f"[Chunk {i+1}]: {chunk[:600]}" for i, chunk in enumerate(chunks)
    )

    score_schema = ", ".join(f'"chunk_{i+1}": 0.0-1.0' for i in range(len(chunks)))
    prompt = (
        f"Score each context chunk for relevance to the question below.\n\n"
        f"Question: {question}\n\n"
        f"Chunks:\n{chunks_text}\n\n"
        f"For each chunk: 1.0 = perfectly relevant | 0.5 = somewhat relevant | 0.0 = not relevant\n"
        f"Reply with JSON only:\n"
        f'{{"chunk_scores": [list of {len(chunks)} floats in order], '
        f'"reasoning": "one sentence summary"}}'
    )

    response = oai_client.chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content
    try:
        result       = json.loads(raw)
        chunk_scores = result.get("chunk_scores", [])
        reasoning    = result.get("reasoning", "")

        # Validate: must be a non-empty list of numbers
        if not chunk_scores or not isinstance(chunk_scores, list):
            raise ValueError("chunk_scores missing or not a list")

        scores    = [max(0.0, min(1.0, float(s))) for s in chunk_scores]
        avg_score = round(sum(scores) / len(scores), 4)
        return avg_score, reasoning

    except Exception as e:
        log.warning(f"Batch context relevance parse failed ({e}) — falling back to 0.0")
        return 0.0, raw


# ── OPTIMISED: Combined Judge (4 metrics in 1 call) ──────────────────────────
#
# OLD approach — 4 separate calls:
#   score_faithfulness()      → 1 call
#   score_answer_relevance()  → 1 call
#   score_completeness()      → 1 call
#   score_conciseness()       → 1 call
#
# NEW approach — 1 call returning all 4 scores:
#   We ask the judge to evaluate ALL four dimensions in one structured prompt.
#   The judge returns a JSON object with four score+reasoning pairs.
#   This reduces 4 round-trips to 1, cutting latency and token overhead.
#
# TRADE-OFF:
#   Combining many tasks in one prompt can reduce per-task accuracy vs a
#   dedicated focused prompt. In practice the difference is small for these
#   four dimensions since each has a clearly scoped definition and rubric.
#   The ~75% cost saving makes this worthwhile for a 30-example eval run.

def score_all_judges_combined(question, generated_answer, chunks, reference_answer):
    """
    Score faithfulness, answer relevance, completeness, and conciseness
    in a SINGLE LLM call.

    Returns a dict with keys:
        faithfulness_score, faithfulness_reason
        answer_relevance_score, answer_relevance_reason
        completeness_score, completeness_reason
        conciseness_score, conciseness_reason

    Falls back to 0.0 for all scores on any parse error.
    """
    context_block = "\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(chunks))

    prompt = f"""You are an expert RAG evaluation judge. Score the following answer on FOUR dimensions.
Return ONLY a JSON object — no markdown, no extra text.

---
QUESTION:
{question}

RETRIEVED CONTEXT:
{context_block}

GENERATED ANSWER:
{generated_answer}

REFERENCE ANSWER:
{reference_answer}
---

Score each dimension on a 0.0–1.0 scale using these rubrics:

1. faithfulness
   Does the answer ONLY contain facts present in the retrieved context?
   1.0 = every claim is supported | 0.5 = partially grounded | 0.0 = entirely hallucinated

2. answer_relevance
   Does the answer properly address the question?
   1.0 = fully answers | 0.5 = partial or tangential | 0.0 = off-topic or refuses

3. completeness
   Does the answer cover all key points from the REFERENCE answer?
   1.0 = covers everything | 0.5 = covers roughly half | 0.0 = misses all key points

4. conciseness
   Is the answer concise and focused (not padded or unnecessarily verbose)?
   1.0 = perfectly concise | 0.5 = slightly wordy | 0.0 = far too long or heavily padded

Required JSON format:
{{
  "faithfulness":      {{"score": 0.0, "reasoning": "one sentence"}},
  "answer_relevance":  {{"score": 0.0, "reasoning": "one sentence"}},
  "completeness":      {{"score": 0.0, "reasoning": "one sentence"}},
  "conciseness":       {{"score": 0.0, "reasoning": "one sentence"}}
}}"""

    response = oai_client.chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content

    # Defaults — returned as-is if parsing fails so evaluation never crashes
    defaults = {
        "faithfulness_score":      0.0, "faithfulness_reason":      raw,
        "answer_relevance_score":  0.0, "answer_relevance_reason":  raw,
        "completeness_score":      0.0, "completeness_reason":      raw,
        "conciseness_score":       0.0, "conciseness_reason":       raw,
    }

    try:
        result = json.loads(raw)

        def _safe(key):
            """Extract and clamp score from a sub-dict; return (score, reason)."""
            sub = result.get(key, {})
            s   = max(0.0, min(1.0, float(sub.get("score", 0.0))))
            r   = sub.get("reasoning", "")
            return s, r

        fa_score,   fa_reason   = _safe("faithfulness")
        ar_score,   ar_reason   = _safe("answer_relevance")
        comp_score, comp_reason = _safe("completeness")
        conc_score, conc_reason = _safe("conciseness")

        return {
            "faithfulness_score":      fa_score,   "faithfulness_reason":      fa_reason,
            "answer_relevance_score":  ar_score,   "answer_relevance_reason":  ar_reason,
            "completeness_score":      comp_score, "completeness_reason":      comp_reason,
            "conciseness_score":       conc_score, "conciseness_reason":       conc_reason,
        }

    except Exception as e:
        log.warning(f"Combined judge parse failed ({e}) — returning all zeros")
        return defaults


# ── Single-purpose helpers — used by Part B scorers (unchanged interface) ─────

def score_context_relevance(question, chunks):
    """Thin wrapper that delegates to the batched implementation."""
    return score_context_relevance_batched(question, chunks)


def score_faithfulness(generated_answer, chunks):
    """Standalone faithfulness judge — used by Part B @scorer."""
    if not chunks:
        return 0.0, "no chunks retrieved"
    context_block = "\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(chunks))
    return ask_judge(
        f"Does the answer only contain facts that are found in the context?\n\n"
        f"Context:\n{context_block}\n\n"
        f"Answer: {generated_answer}\n\n"
        f"1.0 = fully supported | 0.5 = partially | 0.0 = entirely hallucinated\n"
        f'Reply with JSON only: {{"score": 0.0 to 1.0, "reasoning": "list any unsupported claims"}}'
    )


def score_answer_relevance(question, generated_answer):
    """Standalone answer relevance judge — used by Part B @scorer."""
    return ask_judge(
        f"Does this answer properly address the question?\n\n"
        f"Question: {question}\n"
        f"Answer: {generated_answer}\n\n"
        f"1.0 = fully answers | 0.5 = partial answer | 0.0 = off-topic or refuses\n"
        f'Reply with JSON only: {{"score": 0.0 to 1.0, "reasoning": "one sentence"}}'
    )


def score_completeness(generated_answer, reference_answer):
    """Standalone completeness judge — used by Part B @scorer."""
    return ask_judge(
        f"Does the generated answer cover all key points from the reference answer?\n\n"
        f"Reference: {reference_answer}\n"
        f"Generated: {generated_answer}\n\n"
        f"1.0 = covers everything | 0.5 = covers half | 0.0 = misses everything\n"
        f'Reply with JSON only: {{"score": 0.0 to 1.0, "reasoning": "list any missing points"}}'
    )


def score_conciseness(question, generated_answer):
    """Standalone conciseness judge — used by Part B @scorer."""
    return ask_judge(
        f"Is this answer concise and to the point, or too long and padded?\n\n"
        f"Question: {question}\n"
        f"Answer: {generated_answer}\n\n"
        f"1.0 = perfectly concise | 0.5 = slightly wordy | 0.0 = far too long\n"
        f'Reply with JSON only: {{"score": 0.0 to 1.0, "reasoning": "one sentence"}}'
    )


# ── RAG Triad ─────────────────────────────────────────────────────────────────

def compute_rag_triad(context_relevance, faithfulness, answer_relevance):
    """
    Harmonic mean of the RAG Triad.
    Collapses to near zero if ANY single leg is weak — intentional design.
    """
    cr, fa, ar = context_relevance, faithfulness, answer_relevance
    denom = cr * fa + fa * ar + ar * cr
    if denom == 0:
        return 0.0
    return round(3 * cr * fa * ar / denom, 4)


# ══════════════════════════════════════════════════════════════════════════
# PART A — MODEL TRAINING MODE  ★ OPTIMISED ★
#
# KEY CHANGES vs original:
#
# 1. evaluate_one_example() now makes 2 judge calls instead of 9:
#      score_context_relevance_batched()  — 1 call scores all TOP_K chunks
#      score_all_judges_combined()        — 1 call for faith/AR/comp/conc
#
# 2. run_part_a() uses ThreadPoolExecutor(max_workers=PART_A_WORKERS):
#      All 30 examples are evaluated concurrently rather than sequentially.
#      Each thread handles its own embed + generate + 2 judge calls.
#      Wall-clock time drops from ~(30 × serial_latency) to ~(6 × serial_latency)
#      with 5 workers.
#
# 3. MLflow logging is unchanged — aggregate metrics, step charts, artifacts.
# ══════════════════════════════════════════════════════════════════════════

def evaluate_one_example(example):
    """
    Run the full pipeline + all three metric families on ONE example.

    LLM calls per example (optimised):
      1  embeddings call               (retrieve_chunks)
      1  generation call               (generate_answer)
      1  batched context relevance     (score_context_relevance_batched)
      1  combined judge                (score_all_judges_combined)
      ─────────────────────────────────
      4  total   (was ~9 in original)
    """
    question  = example["question"]
    reference = example["answer"]

    # Retrieve + generate
    doc_ids, chunks = retrieve_chunks(question)
    generated       = generate_answer(question, chunks)

    # Family 1 — Retrieval metrics (pure Python)
    retrieval = compute_retrieval_metrics(doc_ids, example["context_ids"])

    # Family 2 — Lexical metrics (pure Python)
    em  = compute_exact_match(generated, reference)
    tf1 = compute_token_f1(generated, reference)

    # Family 3 — LLM judges  ★ 2 calls instead of 9 ★

    # Call 1: batch-score all retrieved chunks for context relevance
    cr_score, cr_reason = score_context_relevance_batched(question, chunks)

    # Call 2: score faithfulness, answer relevance, completeness, conciseness
    # in a single combined prompt
    combined = score_all_judges_combined(question, generated, chunks, reference)

    fa_score   = combined["faithfulness_score"]
    fa_reason  = combined["faithfulness_reason"]
    ar_score   = combined["answer_relevance_score"]
    ar_reason  = combined["answer_relevance_reason"]
    comp_score = combined["completeness_score"]
    comp_reason= combined["completeness_reason"]
    conc_score = combined["conciseness_score"]
    conc_reason= combined["conciseness_reason"]

    triad = compute_rag_triad(cr_score, fa_score, ar_score)

    return {
        # Inputs and outputs
        "question":         question,
        "reference":        reference,
        "generated_answer": generated,
        "retrieved_ids":    doc_ids,

        # Retrieval scores
        "precision": retrieval["precision"],
        "recall":    retrieval["recall"],
        "f1":        retrieval["f1"],
        "ndcg":      retrieval["ndcg"],
        "hit_rate":  retrieval["hit_rate"],

        # Lexical scores
        "exact_match": em,
        "token_f1":    tf1,

        # LLM-judge scores
        "context_relevance": cr_score,
        "faithfulness":      fa_score,
        "answer_relevance":  ar_score,
        "rag_triad":         triad,
        "completeness":      comp_score,
        "conciseness":       conc_score,

        # Reasoning for debugging low scores
        "cr_reasoning":   cr_reason,
        "fa_reasoning":   fa_reason,
        "ar_reasoning":   ar_reason,
        "comp_reasoning": comp_reason,
        "conc_reasoning": conc_reason,
    }


def average_scores(all_results):
    """Average each numeric metric across all examples."""
    metric_names = [
        "precision", "recall", "f1", "ndcg", "hit_rate",
        "exact_match", "token_f1",
        "context_relevance", "faithfulness", "answer_relevance",
        "rag_triad", "completeness", "conciseness",
    ]
    n = len(all_results)
    return {
        f"avg_{name}": round(sum(r[name] for r in all_results) / n, 4)
        for name in metric_names
    }


def _get_or_restore_experiment(name):
    """
    Get or restore a (possibly soft-deleted) MLflow experiment.
    Handles the 'Cannot set a deleted experiment' edge case.
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    exp    = client.get_experiment_by_name(name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        log.info(f"Restored deleted experiment '{name}'  id={exp.experiment_id}")
    mlflow.set_experiment(name)


def run_part_a(golden_data):
    """
    Part A — evaluate all 30 examples and log results to MLflow.

    ★ PARALLELISM:
      ThreadPoolExecutor runs up to PART_A_WORKERS examples concurrently.
      Each thread: embed → generate → 2 judge calls (4 LLM calls total).
      Results are collected in submission order via a future→index map so
      MLflow step=i metrics remain aligned with the original example order.

    ★ TOTAL LLM CALLS (30 examples, TOP_K=5, PART_A_WORKERS=5):
      Before optimisation: ~270 judge calls + 60 embed/gen = ~330 total
      After optimisation:   60 judge calls + 60 embed/gen =  ~120 total
      Wall-clock speedup:  ~5× with 5 workers
    """
    mlflow.set_tracking_uri(MLFLOW_URL)
    _get_or_restore_experiment(EXPERIMENT_A_NAME)

    with mlflow.start_run(run_name=f"part_a_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        mlflow.log_params({
            "embedding_model": EMBEDDING_MODEL,
            "generator_model": GENERATOR_MODEL,
            "judge_model":     JUDGE_MODEL,
            "top_k":           TOP_K,
            "chunk_size":      CHUNK_SIZE,
            "num_examples":    len(golden_data),
            "part_a_workers":  PART_A_WORKERS,
            "judge_strategy":  "batched_context_relevance + combined_4judges",
        })

        # ── Parallel evaluation ──────────────────────────────────────────
        all_results = [None] * len(golden_data)   # pre-sized so index order is preserved

        with ThreadPoolExecutor(max_workers=PART_A_WORKERS) as executor:
            # Submit all examples, keep a mapping future → original index
            future_to_idx = {
                executor.submit(evaluate_one_example, example): i
                for i, example in enumerate(golden_data)
            }

            completed = 0
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    all_results[i] = future.result()
                    completed += 1
                    log.info(
                        f"  [{completed}/{len(golden_data)}] "
                        f"example {i+1} done — "
                        f"triad={all_results[i]['rag_triad']:.3f}"
                    )
                except Exception as exc:
                    log.error(f"  Example {i+1} failed: {exc}")
                    # Insert a zero-scored placeholder so the run still completes
                    all_results[i] = {
                        "question":         golden_data[i]["question"],
                        "reference":        golden_data[i]["answer"],
                        "generated_answer": "",
                        "retrieved_ids":    [],
                        "precision": 0.0, "recall": 0.0, "f1": 0.0,
                        "ndcg": 0.0, "hit_rate": 0.0,
                        "exact_match": 0.0, "token_f1": 0.0,
                        "context_relevance": 0.0, "faithfulness": 0.0,
                        "answer_relevance": 0.0, "rag_triad": 0.0,
                        "completeness": 0.0, "conciseness": 0.0,
                        "cr_reasoning": str(exc), "fa_reasoning": "",
                        "ar_reasoning": "", "comp_reasoning": "",
                        "conc_reasoning": "",
                    }

        # ── Aggregate metrics ────────────────────────────────────────────
        averages = average_scores(all_results)
        mlflow.log_metrics(averages)

        # Per-example step metrics — shows as line charts in MLflow UI
        for step, result in enumerate(all_results):
            mlflow.log_metrics({
                "faithfulness":      result["faithfulness"],
                "context_relevance": result["context_relevance"],
                "answer_relevance":  result["answer_relevance"],
                "rag_triad":         result["rag_triad"],
            }, step=step)

        # Artifact 1: clean scores JSONL (no reasoning — good for pandas)
        results_file = "eval_results.jsonl"
        with open(results_file, "w") as f:
            for r in all_results:
                clean = {k: v for k, v in r.items() if "reasoning" not in k}
                f.write(json.dumps(clean) + "\n")
        mlflow.log_artifact(results_file)

        # Artifact 2: verbose reasoning JSON (for debugging low-scoring examples)
        reasoning_file = "judge_reasoning.json"
        reasoning_data = [
            {
                "question":         r["question"],
                "generated_answer": r["generated_answer"],
                "context_relevance": {"score": r["context_relevance"], "why": r["cr_reasoning"]},
                "faithfulness":      {"score": r["faithfulness"],      "why": r["fa_reasoning"]},
                "answer_relevance":  {"score": r["answer_relevance"],  "why": r["ar_reasoning"]},
                "completeness":      {"score": r["completeness"],      "why": r["comp_reasoning"]},
                "conciseness":       {"score": r["conciseness"],       "why": r["conc_reasoning"]},
            }
            for r in all_results
        ]
        with open(reasoning_file, "w") as f:
            json.dump(reasoning_data, f, indent=2)
        mlflow.log_artifact(reasoning_file)

        # Terminal summary
        log.info("\n" + "=" * 60)
        log.info("  PART A RESULTS — Model Training Mode  (optimised)")
        log.info("=" * 60)
        log.info(f"  {'Metric':<35} {'Score':>8}")
        log.info("-" * 60)
        for name, value in averages.items():
            log.info(f"  {name:<35} {value:>8.4f}")
        log.info("=" * 60)
        log.info(f"  Judge calls: ~{len(golden_data) * 2} total  "
                 f"(was ~{len(golden_data) * 9})")
        log.info(f"  View → {MLFLOW_URL}  (Model Training tab)")
        log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# PART B — GENAI MODE  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════

def _decode_outputs(outputs):
    try:
        data = json.loads(outputs)
        return data.get("answer", str(outputs)), data.get("retrieved_context", [])
    except Exception:
        return str(outputs), []


@scorer
def context_relevance_scorer(inputs, outputs, expectations, **kwargs):
    query                = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
    _, retrieved_context = _decode_outputs(outputs)
    if not retrieved_context:
        return 0.0
    score, _ = score_context_relevance(query, list(retrieved_context))
    return score


@scorer
def faithfulness_scorer(inputs, outputs, expectations, **kwargs):
    answer, retrieved_context = _decode_outputs(outputs)
    if not retrieved_context:
        return 0.0
    score, _ = score_faithfulness(answer, list(retrieved_context))
    return score


@scorer
def answer_relevance_scorer(inputs, outputs, **kwargs):
    query  = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
    answer, _ = _decode_outputs(outputs)
    score, _  = score_answer_relevance(query, answer)
    return score


@scorer
def completeness_scorer(inputs, outputs, expectations, **kwargs):
    answer, _       = _decode_outputs(outputs)
    expected_output = expectations.get("expected_output", "") if isinstance(expectations, dict) else ""
    if not expected_output:
        return 0.0
    score, _ = score_completeness(answer, str(expected_output))
    return score


@scorer
def conciseness_scorer(inputs, outputs, **kwargs):
    query  = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
    answer, _ = _decode_outputs(outputs)
    score, _  = score_conciseness(query, answer)
    return score


def _compute_rag_triad_from_results(results_df):
    triad_scores = []
    for _, row in results_df.iterrows():
        cr = row.get("context_relevance_scorer/score", 0.0) or 0.0
        fa = row.get("faithfulness_scorer/score",       0.0) or 0.0
        ar = row.get("answer_relevance_scorer/score",   0.0) or 0.0
        triad_scores.append(compute_rag_triad(float(cr), float(fa), float(ar)))
    return triad_scores


def run_part_b(golden_data):
    """Part B — trace the pipeline and evaluate with mlflow.genai.evaluate()."""
    mlflow.set_tracking_uri(MLFLOW_URL)
    _get_or_restore_experiment(EXPERIMENT_B_NAME)

    with mlflow.start_run(run_name=f"part_b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        mlflow.log_params({
            "embedding_model": EMBEDDING_MODEL,
            "generator_model": GENERATOR_MODEL,
            "judge_model":     JUDGE_MODEL,
            "top_k":           TOP_K,
            "num_examples":    len(golden_data),
        })

        def predict_fn(query):
            result = run_rag_pipeline(query)
            return json.dumps({
                "answer": result["answer"],
                "retrieved_context": result["chunks"],
            })

        eval_df = pd.DataFrame([
            {
                "inputs": {"query": example["question"]},
                "expectations": {
                    "expected_response": example["answer"],
                    "expected_output":   example["answer"],
                },
            }
            for example in golden_data
        ])

        log.info(f"  Built eval DataFrame with {len(eval_df)} rows")
        log.info("  Running mlflow.genai.evaluate() with 8 scorers...")

        results = mlflow.genai.evaluate(
            data=eval_df,
            predict_fn=predict_fn,
            scorers=[
                Correctness(),
                RelevanceToQuery(),
                Safety(),
                context_relevance_scorer,
                faithfulness_scorer,
                answer_relevance_scorer,
                completeness_scorer,
                conciseness_scorer,
            ]
        )

        try:
            triad_scores = _compute_rag_triad_from_results(results.tables["eval_results"])
            avg_triad    = round(sum(triad_scores) / len(triad_scores), 4)
            mlflow.log_metric("avg_rag_triad", avg_triad)
            log.info(f"  RAG Triad (post-eval) = {avg_triad:.4f}")
        except Exception as e:
            log.warning(f"  Could not compute RAG Triad: {e}")

        log.info("\n" + "=" * 55)
        log.info("  PART B RESULTS — GenAI Mode")
        log.info("=" * 55)
        for name, value in results.metrics.items():
            if isinstance(value, (int, float)):
                log.info(f"  {name:<32} {float(value):>8.4f}")
        log.info("=" * 55)
        log.info(f"  Traces  → {MLFLOW_URL}  (GenAI tab → Traces)")
        log.info(f"  Results → {MLFLOW_URL}  (GenAI tab → Eval Results)")
        log.info("=" * 55)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline — MLflow 3")
    parser.add_argument(
        "--mode",
        choices=["a", "b", "both"],
        default="a",
        help="a = Part A (hand-coded metrics)  |  b = Part B (GenAI mode)  |  both = run both"
    )
    args = parser.parse_args()

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {GOLDEN_FILE}\n"
            "Please run 1_prepare_dataset.py first."
        )

    golden_data = load_golden_data(GOLDEN_FILE)
    log.info(f"Mode: {args.mode}")

    global oai_client, pine_index
    oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc         = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pine_index = pc.Index(PINECONE_INDEX)

    mlflow.set_tracking_uri(MLFLOW_URL)

    if args.mode in ("a", "both"):
        log.info("\n── PART A ─────────────────────────────────────────────")
        run_part_a(golden_data)

    if args.mode in ("b", "both"):
        log.info("\n── PART B ─────────────────────────────────────────────")
        run_part_b(golden_data)

    log.info(f"\nAll done! Open {MLFLOW_URL} to see your results.")
    if args.mode in ("a", "both"):
        log.info("  Model Training tab → params, metrics, step charts, artifacts")
    if args.mode in ("b", "both"):
        log.info("  GenAI tab          → traces waterfall + scorer eval table")


if __name__ == "__main__":
    main()