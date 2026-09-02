"""What documents are actually in the index right now?

    python check_index_contents.py

Answers one question directly, rather than inferring it from search results:
which doc_id values does this index actually hold. Two identical, unrelated
questions both returning only one document's chunks is the symptom of the
scleroderma form never having been synced into THIS index — this checks that
without guessing.

Uses a real embedding (any text) with a per-doc_id FILTER and top_k=1. A
filter always returns whatever matches it, ranked by the query vector — the
query's content is irrelevant here, only whether anything comes back at all.
"""

from dotenv import load_dotenv
load_dotenv()

from rag import index as index_module
from rag.clients import EMBED_DIMS
from rag.embedding import embed

# Every doc_id this session has produced records for. Slugified the way
# config.slugify() would from the filename.
CANDIDATES = [
    "ai-enablers-adopters-research-report",
    "nct02014597-scleroderma-study",
    "nct03164772-nsclc-mrna-vaccine",
    "nct03235752-ulcerative-colitis",
    "nct04614948-moderna-vaccine",
]


def main() -> None:
    idx = index_module.open_index()
    probe_vector = embed(["probe"])[0].tolist()

    print(f"{'doc_id':<45}{'in index?':>12}{'sample chunk_id'}")
    print("-" * 100)
    for doc_id in CANDIDATES:
        matches = idx.query(
            vector=probe_vector, top_k=1, namespace="",
            include_metadata=True,
            filter={"doc_id": {"$eq": doc_id}},
        ).matches
        present = "YES" if matches else "NO"
        sample = matches[0].metadata.get("chunk_id", "") if matches else ""
        print(f"{doc_id:<45}{present:>12}  {sample}")

    stats = idx.describe_index_stats()
    print(f"\ntotal vectors in index: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
