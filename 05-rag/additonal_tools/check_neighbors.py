"""Does with_neighbors actually do anything against the real index?

    python check_neighbors.py
    python check_neighbors.py "your own question here"

Every stub test for with_neighbors passed in isolation, against a fake index
built by hand. That proves the LOGIC is right. It says nothing about whether
a real Pinecone `$in` filter on chunk_id behaves the way the fake index
assumed, or whether the field actually made it into your index's metadata.
This is that missing check, run once, against the real thing.

WHAT IT DOES

    1. connects to the index you already built
    2. asks the same question twice — once through search() alone, once
       through search() + with_neighbors() — and prints both result lists
       side by side
    3. runs the full answer() path and shows what ended up in `sources`

WHAT TO LOOK FOR

    If with_neighbors is doing nothing, the "before" and "after" lists are
    identical — same chunk_ids, same count. That is the failure case: it
    would mean either no result was short enough to trigger expansion (raise
    --min-tokens, or ask a question more likely to hit a small chunk), or the
    real index query is failing silently somehow.

    If it worked, "after" has entries "before" does not — chunk_ids that
    would not have matched the question on their own, pulled in only because
    they sit next to something that did.

Point PDF_HINT at whichever document you know has short chunks — the
scleroderma form's 'Section N: ... / None' records are the clearest case in
this corpus, at 8-30 tokens each.
"""

import sys

# MUST run before any `rag` import. rag.clients constructs the OpenAI client
# and probes EMBED_DIMS with a real API call AT IMPORT TIME, not lazily
# inside a function — see clients.py's own docstring for why. Loading the
# .env inside main() is too late: `from rag.clients import EMBED_DIMS` below
# has already run by the time main() is even called, so it reads an
# environment that does not have OPENAI_API_KEY in it yet.
from dotenv import load_dotenv
load_dotenv()

from rag import index as index_module
from rag.clients import EMBED_DIMS
from rag.retrieval import answer, search, with_neighbors

DEFAULT_QUESTION = "What is the funding source for this study?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"question: {question!r}\n")

    idx = index_module.open_index()

    # ── before: search alone, no expansion ──────────────────────────────
    before = search(idx, question, top_k=5)
    if not before:
        print("No results at all — check the index has data before "
              "debugging with_neighbors specifically.")
        return

    print(f"BEFORE with_neighbors — {len(before)} result(s):")
    for r in before:
        flag = " <-- under 100 tokens, should trigger expansion" \
               if r["n_tokens"] < 100 else ""
        print(f"  {r['chunk_id']:<55} {r['n_tokens']:>5}t  p{r['page']}{flag}")

    # ── after: the same results, run through with_neighbors ────────────
    after = with_neighbors(idx, before, embed_dims=EMBED_DIMS)
    print(f"\nAFTER with_neighbors — {len(after)} result(s):")
    before_ids = {r["chunk_id"] for r in before}
    for r in after:
        flag = "  <-- NEW, pulled in as a neighbour" \
               if r["chunk_id"] not in before_ids else ""
        print(f"  {r['chunk_id']:<55} {r['n_tokens']:>5}t  p{r['page']}{flag}")

    added = len(after) - len(before)
    if added > 0:
        print(f"\n{added} chunk(s) added by expansion — with_neighbors fired.")
    else:
        print("\nNo chunks added. Either nothing in this result set was "
              "short/truncated, or the neighbours were already present on "
              "their own merit. Try a question more likely to land on a "
              "short chunk, or lower with_neighbors' min_tokens.")

    # ── the real path: full answer(), same as production use ───────────
    print("\n" + "=" * 70)
    print("Full answer() path, with neighbours on (the default):")
    result = answer(idx, question, top_k=5, embed_dims=EMBED_DIMS)
    print(f"\n{result['answer']}\n")
    print("sources:")
    for chunk_id, page, content_type in result["sources"]:
        print(f"  {chunk_id:<55} p{page}  {content_type}")
    print(f"\nstats: {result['stats']}")


if __name__ == "__main__":
    main()
