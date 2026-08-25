"""
RAGBench (HotpotQA) Dataset Loader for RAG Evaluation
-----------------------------------------------------
Produces two files:
  1. golden_hotpotqa_30.jsonl      — 30 examples for evaluation
                                     (questions + gold answers + contexts + context_ids)
  2. rag_corpus_hotpotqa_500.jsonl — 500 document records to populate Pinecone

ID contract
-----------
Corpus record ID : {example_id}_d{doc_index}    e.g. "5ae7473f5542991bbc9761d2_d0"
Pinecone ID      : same — docs are embedded as-is, no sub-chunking applied.

context_ids in the golden dataset are therefore exact Pinecone IDs.
Evaluator match logic is simple equality:
    retrieved_id in set(context_ids)

Run:
  pip install datasets
  python 1_prepare_dataset.py
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

# pip install datasets
from datasets import load_dataset

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ragbench_loader")

# ── Config ───────────────────────────────────────────────────────────────────
DS_NAME    = "rungalileo/ragbench"
CONFIG     = "hotpotqa"
SPLIT      = "train"
GOLDEN_N   = 30
CORPUS_N   = 500
MAX_CHARS  = 8_000

OUT_DIR    = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLDEN_PATH = OUT_DIR / "golden_hotpotqa_30.jsonl"
CORPUS_PATH = OUT_DIR / "rag_corpus_hotpotqa_500.jsonl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def as_text_list(documents: Optional[List], documents_sentences: Optional[List] = None) -> List[str]:
    """
    Normalise the 'documents' field into a plain list[str].

    RAGBench shapes we handle:
      • list[str]           — already plain text
      • list[dict]          — dicts with 'text'/'content'/… keys or a 'sentences' list
      • list[list[str]]     — sentences grouped by document
      • fallback to documents_sentences if nothing else works
    """
    if not documents:
        return []

    first = documents[0]

    # Case A: already plain strings
    if isinstance(first, str):
        return [d for d in documents if isinstance(d, str) and d.strip()]

    # Case B: list of lists of strings (e.g. sentences per doc)
    if isinstance(first, list):
        return [
            " ".join(s for s in doc if isinstance(s, str) and s.strip())
            for doc in documents
            if isinstance(doc, list)
        ]

    # Case C: list of dicts
    if isinstance(first, dict):
        candidate_keys = ("text", "content", "page_content", "body")
        out = []
        for d in documents:
            if not isinstance(d, dict):
                continue
            txt = next(
                (d[k].strip() for k in candidate_keys if k in d and isinstance(d[k], str) and d[k].strip()),
                None,
            )
            if txt is None and isinstance(d.get("sentences"), list):
                txt = " ".join(s for s in d["sentences"] if isinstance(s, str) and s.strip())
            if txt:
                out.append(txt)
        if out:
            return out

    # Case D: documents_sentences as last resort
    if documents_sentences and isinstance(documents_sentences, list):
        return [
            " ".join(s for s in sents if isinstance(s, str) and s.strip())
            for sents in documents_sentences
            if isinstance(sents, list)
        ]

    return []


def trunc(s: str, max_chars: int = MAX_CHARS) -> str:
    return s[:max_chars] if len(s) > max_chars else s


def make_id(row: dict, idx: int) -> str:
    """Return a stable string ID — fallback to row index if the field is missing."""
    raw = row.get("id")
    if raw is not None:
        return str(raw)
    return f"row_{idx}"


# ── Load dataset ──────────────────────────────────────────────────────────────
log.info(f"Loading '{DS_NAME}' config='{CONFIG}' split='{SPLIT}' ...")
ds = load_dataset(DS_NAME, CONFIG, split=SPLIT)
log.info(f"Loaded {len(ds)} rows | fields: {list(ds.features.keys())}")

# ── Step 1 : Golden dataset (first 30 rows) ───────────────────────────────────
golden_n = min(GOLDEN_N, len(ds))
log.info(f"Writing golden dataset ({golden_n} examples) → {GOLDEN_PATH}")

with GOLDEN_PATH.open("w", encoding="utf-8") as f:
    for i, row in enumerate(ds.select(range(golden_n))):
        ex_id    = make_id(row, i)
        contexts = as_text_list(row.get("documents"), row.get("documents_sentences"))

        # context_ids are the exact Pinecone vector IDs for this example's docs.
        # No sub-chunking is applied during ingestion, so the corpus record ID
        # "{ex_id}_d{k}" IS the Pinecone ID — direct equality match at eval time.
        context_ids = [f"{ex_id}_d{k}" for k in range(len(contexts))]

        item = {
            "id":           ex_id,
            "input":        row.get("question", ""),
            "reference":    row.get("response", ""),
            "contexts":     contexts,
            "context_ids":  context_ids,   # exact Pinecone IDs for Recall/Precision/F1
            "dataset_name": row.get("dataset_name", CONFIG),
            "source":       f"{DS_NAME}/{CONFIG}",
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

log.info(f"Golden dataset written → {GOLDEN_PATH.resolve()}")

# ── Step 2 : Corpus dataset (first 500 rows, flat per-doc records) ─────────────
corpus_n = min(CORPUS_N, len(ds))
log.info(f"Writing corpus dataset ({corpus_n} source rows) → {CORPUS_PATH}")

gold_ids = {make_id(row, i) for i, row in enumerate(ds.select(range(golden_n)))}

with CORPUS_PATH.open("w", encoding="utf-8") as f:
    for i, row in enumerate(ds.select(range(corpus_n))):
        ex_id = make_id(row, i)
        docs  = as_text_list(row.get("documents"), row.get("documents_sentences"))

        if not docs:
            log.warning(f"Row {i} (id={ex_id}) produced no document text — skipping")
            continue

        for k, doc_text in enumerate(docs):
            rec = {
                # This ID is used as-is as the Pinecone vector ID.
                # Must match context_ids[k] in the golden dataset exactly.
                "id":   f"{ex_id}_d{k}",
                "text": trunc(doc_text),
                "metadata": {
                    "example_id":       ex_id,
                    "question":         trunc(row.get("question", ""), 2_000),
                    "reference_answer": trunc(row.get("response",  ""), 2_000),
                    "dataset":          CONFIG,
                    "source":           f"{DS_NAME}/{CONFIG}",
                    "doc_index":        k,
                    "in_golden":        ex_id in gold_ids,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

log.info(f"Corpus dataset written → {CORPUS_PATH.resolve()}")
log.info("Done.")
log.info("  Golden  → evaluation set  (LangSmith / MLflow / RAGAS)")
log.info("  Corpus  → Pinecone vector store")