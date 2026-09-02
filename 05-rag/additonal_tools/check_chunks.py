"""Is the chunking any good?

    python check_chunks.py reports/<doc>.chunks.json

Six things go wrong, and they have different fixes. This says which one you
have. It reads only the chunk report, so it needs no API keys and no parse.

    1  page furniture still indexed    the drop filter did not run
    2  duplicate vectors               identical text indexed several times
    3  prose too small to answer        the merge could not reach
    4  chunks at the ceiling            something could not be split
    5  junk heading paths               a layout mistake, embedded as context
    6  figures without an exhibit label  the caption was never linked
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Anything shorter than this cannot answer a question on its own.
FRAGMENT_TOKENS = 40

# Patterns that should have been removed by chunking.is_furniture. Kept in step
# with that function on purpose — if it stops matching something, this notices.
FURNITURE_HINTS = ("source:", "sources:")


def _texts_from_md(json_path: Path) -> dict[int, str]:
    """Chunk bodies, read from the Markdown report next to the JSON one.

        reports/<doc>.chunks.json   metadata only — no text field
        reports/<doc>.chunks.md     the text, in fenced blocks

    The JSON report deliberately does not carry chunk text: it would double the
    file for something the Markdown report already shows readably. Two of the
    checks below need the text anyway, so it is read from there.

    Returns an empty dict when the Markdown report is missing, and the checks
    that need it say so rather than reporting zero.
    """
    md_path = json_path.with_suffix("").with_suffix(".chunks.md")
    if not md_path.exists():
        return {}
    raw = md_path.read_text(encoding="utf-8")
    bodies = {}
    for block in re.split(r"\n## \[\s*", raw)[1:]:
        try:
            index = int(block.split("]")[0])
        except ValueError:
            continue
        fence = re.search(r"```\n(.*?)\n```", block, re.S)
        if fence:
            bodies[index] = fence.group(1)
    return bodies


def main() -> None:
    path = Path(sys.argv[1])
    records = json.loads(path.read_text())
    bodies = _texts_from_md(path)
    ceiling = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    types = Counter(r["content_type"] for r in records)
    text_only = [r for r in records if r["content_type"] == "text"]

    # CHUNK_TOKENS is a CEILING, not a target. Nothing pads a chunk up to it.
    # A section whose whole content is 80 tokens produces an 80-token chunk,
    # and that is correct. What matters is whether a chunk carries enough to
    # answer something, and whether its heading context is right.
    print(f"{len(records)} records. Ceiling {ceiling} tokens.\n")
    print(f"{'type':<16}{'count':>7}{'median':>8}{'min':>6}{'max':>7}")
    print("-" * 44)
    for kind in types:
        sizes = sorted(r["n_tokens"] for r in records
                       if r["content_type"] == kind)
        print(f"{kind:<16}{len(sizes):>7}{statistics.median(sizes):>8.0f}"
              f"{sizes[0]:>6}{sizes[-1]:>7}")

    # ── 1. page furniture ───────────────────────────────────────────────────
    #
    # The page really does say "Source: Morgan Stanley Research". That is not a
    # parse failure — it is content nobody will ever ask about, taking up a
    # vector and competing for top-k.
    furniture = []
    for index, r in enumerate(records):
        if r["content_type"] != "text":
            continue
        body = bodies.get(index, "")
        for h in r.get("headings") or []:
            if body.startswith(h):
                body = body[len(h):].strip()
        if r["n_tokens"] <= 15 and body.lower().startswith(FURNITURE_HINTS):
            furniture.append((r["page"], body[:50]))

    if not bodies:
        print("\n1  page furniture: NOT CHECKED — no .chunks.md next to the JSON")
    else:
        print(f"\n1  page furniture still indexed: {len(furniture)}")
    for page, sample in furniture[:5]:
        print(f"     p{page}  {sample!r}")
    if furniture:
        print("   chunking.is_furniture should have removed these. Either the")
        print("   loaded code predates the drop filter — run check_wiring.py —")
        print("   or these need a new pattern in FURNITURE_PATTERNS.")

    # ── 2. duplicate vectors ────────────────────────────────────────────────
    #
    # Identical text embeds to an identical vector. Six copies of one point in
    # the index can occupy an entire top-k between them and answer nothing.
    by_hash = Counter(r["content_hash"] for r in records)
    dupes = {h: n for h, n in by_hash.items() if n > 1}
    print(f"\n2  distinct texts: {len(by_hash)} of {len(records)}")
    if dupes:
        print(f"   {len(dupes)} text(s) indexed more than once:")
        for h, n in sorted(dupes.items(), key=lambda kv: -kv[1])[:5]:
            index, sample = next((i, r) for i, r in enumerate(records)
                                 if r["content_hash"] == h)
            print(f"     {n}x  p{sample['page']}  "
                  f"{bodies.get(index, '')[:50]!r}")
        print("   Legitimate for a disclaimer printed on every page. For")
        print("   anything else it is furniture the drop filter missed.")

    # ── 3. prose size ───────────────────────────────────────────────────────
    #
    # The merge joins consecutive prose under one heading path. It cannot cross
    # a heading boundary, so a document with many short sections keeps small
    # prose chunks however the merge is tuned — and that is correct.
    per_heading = defaultdict(int)
    for r in text_only:
        per_heading[" > ".join(r.get("headings") or []) or "(none)"] += 1

    sizes = sorted(r["n_tokens"] for r in text_only) or [0]
    fragments = [r for r in text_only if r["n_tokens"] < FRAGMENT_TOKENS]
    print(f"\n3  prose: {len(text_only)} chunks across {len(per_heading)} "
          f"headings, median {statistics.median(sizes):.0f} tokens")
    print(f"   under {FRAGMENT_TOKENS} tokens: {len(fragments)}")
    alone = sum(1 for n in per_heading.values() if n == 1)
    if alone == len(per_heading) and len(text_only) > 3:
        print("   EVERY heading owns exactly one prose chunk, so the merge had")
        print("   nothing left to join. If the median is still low, the")
        print("   headings are splitting prose that belongs together — see 5.")
    elif statistics.median(sizes) < 80:
        print("   Low. Try MERGE_ACROSS_EXHIBITS=1 if it is off, then raise")
        print("   PROSE_TARGET_TOKENS. If neither moves it, the limit is the")
        print("   number of headings, not the merge.")

    # ── 4. the ceiling ──────────────────────────────────────────────────────
    #
    # If nothing is near it, the setting is not doing anything and lowering it
    # would start splitting tables rather than making prose more even.
    near = [r for r in records if r["n_tokens"] >= ceiling * 0.8]
    truncated = [r for r in records if r.get("truncated")]
    print(f"\n4  within 20% of the ceiling: {len(near)}   "
          f"truncated: {len(truncated)}")
    if not near:
        print(f"   NOTHING is near {ceiling}. The ceiling is not binding on")
        print("   this document, so changing it will not make chunks more even.")
    if truncated:
        print("   Content was DROPPED. An element could not be split below the")
        print("   budget — usually one enormous table row.")

    # ── 5. junk heading paths ───────────────────────────────────────────────
    #
    # The heading path is prepended to the text BEFORE embedding, so a wrong
    # heading is welded onto the front of the chunk and every query is matched
    # against it. It is also the entire merge predicate, so a false heading
    # blocks merges that should have happened.
    def looks_wrong(heading: str) -> bool:
        h = heading.strip()
        return (h.endswith((".", ",")) or h.lower().startswith(("the ", "a ", "and "))
                or (h.endswith(":") and len(h) < 14) or len(h) > 90)

    suspect = sorted({(r["page"], h) for r in records
                      for h in (r.get("headings") or []) if looks_wrong(h)})
    print(f"\n5  suspicious heading paths: {len(suspect)}")
    for page, heading in suspect[:8]:
        print(f"     p{page}  {heading!r}")
    if suspect:
        print("   headings.py should have demoted these before chunking. If it")
        print("   reported demotions and these survived, the demotion did not")
        print("   take effect — run check_wiring.py.")

    # ── 6. figures without an exhibit label ─────────────────────────────────
    #
    # A figure record with no exhibit number cannot answer "what does Exhibit 9
    # show". The caption is what carries that, and the layout model links it —
    # or fails to, silently.
    figures = [(i, r) for i, r in enumerate(records)
               if r["content_type"] == "figure"]
    unlabelled = [r for i, r in figures
                  if not re.search(r"(exhibit|figure|table)\s*\d",
                                   bodies.get(i, ""), re.IGNORECASE)]
    linked = sum(1 for _, r in figures if r.get("has_caption"))
    print(f"\n6  figures: {len(figures)}, with a linked caption: {linked}"
          + (f", with no exhibit label anywhere: {len(unlabelled)}"
             if bodies else ", label check skipped (no .chunks.md)"))
    if unlabelled and bodies:
        print("   These are retrievable by description only. The fix is")
        print("   upstream: try a different LAYOUT_MODEL and compare the")
        print("   caption-linked count in the extraction report.")


if __name__ == "__main__":
    main()
