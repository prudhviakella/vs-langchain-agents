"""
Ingest RAGBench HotpotQA corpus into Pinecone (pure OpenAI + Pinecone, no LangChain)
--------------------------------------------------------------------------------------
Reads:  data/rag_corpus_hotpotqa_500.jsonl
Does:   embed each doc as-is (OpenAI) → upsert (Pinecone)

No sub-chunking is applied. HotpotQA corpus docs are short Wikipedia paragraphs
(typically 200–600 characters) that fit comfortably within the embedding model's
token limit, so splitting would only fragment context without any benefit.

ID contract (important for eval Recall/Precision/F1)
----------------------------------------------------
Corpus record ID == Pinecone vector ID == "{example_id}_d{doc_index}"
e.g. "5ae7473f5542991bbc9761d2_d0"

These IDs match context_ids in golden_hotpotqa_30.jsonl exactly.
Evaluator match logic is simple equality:
    retrieved_id in set(context_ids)

Env vars:
  export OPENAI_API_KEY="sk-..."
  export PINECONE_API_KEY="pcn-..."

Optional:
  PINECONE_INDEX   (default: hotpotqa-ragbench-mini)
  PINECONE_NS      (default: hotpotqa)
  EMBEDDING_MODEL  (default: text-embedding-3-large)
  BATCH_SIZE       (default: 100)
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Iterable

import openai
from pinecone import Pinecone, ServerlessSpec

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("pinecone_ingest")

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH     = Path("data/rag_corpus_hotpotqa_500.jsonl")
INDEX_NAME      = os.getenv("PINECONE_INDEX", "hotpotqa-ragbench-mini")
NAMESPACE       = os.getenv("PINECONE_NS", "hotpotqa")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "100"))   # Pinecone upsert limit is 100 vectors

MODEL_DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}
DIMENSION  = MODEL_DIMS.get(EMBEDDING_MODEL, 3072)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_env():
    missing = [k for k in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_records(records: List[Dict]) -> List[Dict]:
    """
    Drop any records that have no usable text.
    Each remaining record is embedded as a single vector — no splitting.
    """
    clean = []
    for rec in records:
        text = (rec.get("text") or "").strip()
        if not text:
            log.warning(f"Record id={rec.get('id')} has no text — skipping")
            continue
        clean.append(rec)
    log.info(f"  {len(clean)}/{len(records)} records have usable text")
    return clean


def embed_texts(texts: List[str], client: openai.OpenAI) -> List[List[float]]:
    """Call OpenAI Embeddings API for a batch of texts."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # response.data is ordered the same as input
    return [item.embedding for item in response.data]


def batched(items: List, size: int) -> Iterable[List]:
    for i in range(0, len(items), size):
        yield items[i: i + size]


def ensure_index(pc: Pinecone):
    existing = {idx.name for idx in pc.list_indexes()}
    if INDEX_NAME in existing:
        log.info(f"Index '{INDEX_NAME}' already exists — skipping creation")
        return
    log.info(f"Creating index '{INDEX_NAME}' dim={DIMENSION} metric=cosine ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait until ready
    for _ in range(20):
        status = pc.describe_index(INDEX_NAME).status
        if status.get("ready"):
            break
        log.info("Waiting for index to become ready ...")
        time.sleep(3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_env()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}. Run 1_prepare_dataset.py first."
        )

    # Clients
    oai = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc  = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    ensure_index(pc)
    index = pc.Index(INDEX_NAME)

    # Load → filter (no chunking — each record becomes exactly one vector)
    log.info(f"Loading corpus from {CORPUS_PATH} ...")
    records = load_jsonl(CORPUS_PATH)
    log.info(f"Loaded {len(records)} records")

    records = filter_records(records)
    log.info(f"Embedding {len(records)} records as single vectors (no sub-chunking)")

    # Embed → upsert in batches
    total = 0
    for batch in batched(records, BATCH_SIZE):
        texts  = [rec["text"] for rec in batch]
        embeds = embed_texts(texts, oai)

        vectors = [
            {
                # Pinecone ID == corpus record ID == "{example_id}_d{doc_index}"
                # This must stay in sync with context_ids in the golden dataset.
                "id":     rec["id"],
                "values": emb,
                "metadata": {
                    **rec.get("metadata", {}),
                    "text": rec["text"],   # stored so retrieval can return the text
                },
            }
            for rec, emb in zip(batch, embeds)
        ]

        index.upsert(vectors=vectors, namespace=NAMESPACE)
        total += len(batch)
        log.info(f"Upserted {total}/{len(records)}")

    log.info("Ingestion complete!")
    log.info(f"Index: {INDEX_NAME} | Namespace: {NAMESPACE} | Vectors: {total}")
    log.info("")
    log.info("Pinecone ID schema:  {example_id}_d{doc_index}")
    log.info("  e.g.  5ae7473f5542991bbc9761d2_d0")
    log.info("")
    log.info("Evaluator match (Recall/Precision/F1):  retrieved_id in set(context_ids)")


if __name__ == "__main__":
    main()