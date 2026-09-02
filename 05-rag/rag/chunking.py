"""Turning a parsed document into the records that will be embedded.

    parsed document
          |
          v
    clean_headings()      false headings demoted, before any boundary is set
          |
          v
    HybridChunker         splits on structure, then on a token budget
          |
          v
    _to_entries()         classify, drop page furniture, mark figure slots
          |
          v
    _merge_prose()        adjacent text under one heading
          |
          v
    _apply_floor()        small records absorb forward, across headings
          |
          v
    _to_records()         figure slots expand to one record per figure
          |
          v
    _table_summaries()    one per table — extra records, not replacements
          |
          v
    _finalise()           reading order, prev/next links, size limits
          |
          v
      records[]  ->  embedding.py  ->  index.py  ->  Pinecone

WHAT THIS FILE DOES NOT DO

    It does NOT call the embedding model.
    It does NOT write to Pinecone.
    It does NOT parse the PDF or run any vision model.

THE THREE THINGS THE CHUNKER WILL NOT DO FOR YOU

HybridChunker only splits, and merges consecutive chunks with an equal heading
path. Four stages, and only the last combines anything:

    HierarchicalChunker      one chunk per detected element
    _split_by_doc_items      window the items to fit the token budget
    _split_using_plain_text  semchunk whatever is still oversized
    _merge_chunks_with_matching_metadata     only when merge_peers=True

Nothing filters, nothing enforces a minimum, and the merge is blind to element
type. So three policies are ours:

    what is not worth indexing        _to_entries, the drop filter
    how small a prose record may be   _merge_prose and _apply_floor
    figures are one record each       _to_records

Every non-obvious decision here was measured. The measurements, and the
failures that produced them, are in DESIGN.md. This file says WHAT; that one
says WHY, and is worth reading once before changing anything.
"""

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from docling_core.types.doc import PictureItem

from .config import (ACCESS_GROUPS, CHUNK_TOKENS, EMBED_MODEL, ENCODING,
                     PINECONE_METADATA_BYTES)
from .docling_io import chart_data, picture_description
from .headings import clean_headings
from .tables import (needs_summary, summarize_table, summarize_table_image,
                     table_cells, table_looks_broken, table_markdown, table_ref_of)

# ── tuning ──────────────────────────────────────────────────────────────────
# Not in config.py: that holds settings ingestion and retrieval must AGREE on.
# These only shape what goes into the index. See DESIGN.md for the numbers.

# Above this a chunk is never dropped, whatever it looks like.
FURNITURE_MAX_TOKENS = 15

# Where _merge_prose stops. Well under CHUNK_TOKENS: the merge should improve
# small chunks, not manufacture maximal ones.
PROSE_TARGET_TOKENS = 400

# Whether _merge_prose may reach back past an intervening figure or table.
MERGE_ACROSS_EXHIBITS = True

# 0 = off. Above 0, a record under this size absorbs the next one even across
# a heading boundary. Set per corpus: a form wants a floor, a report does not.
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "0"))

# Page furniture. Every pattern came from a real chunk in a real run. Anchored
# at the start, and gated by the token ceiling above, because a pattern that is
# too greedy deletes content and nothing downstream can tell.
FURNITURE_PATTERNS = [
    (re.compile(r"^sources?\s*:", re.IGNORECASE), "an attribution line"),
    (re.compile(r"^[A-Za-z]{1,2}$"), "a single glyph"),
    (re.compile(r"^page\s+\d+\b", re.IGNORECASE), "a page marker"),
    (re.compile(r"^\d{1,3}$"), "a bare page number"),
]


# ════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _tokens(text: str) -> int:
    """Token count, by the embedding model's own tokenizer.

    Character counts are a proxy that fails silently: the API accepts an
    oversized input, embeds the first N tokens, and returns a valid-looking
    vector for half a chunk.
    """
    return len(ENCODING.encode(text))


def content_type(chunk) -> str:
    """Coarse type: text, table, figure, formula or code.

    Reads element labels only — it does NOT look at the content. Order
    matters: a chunk holding a table and a caption is a table, not text.
    """
    labels = " ".join(str(getattr(item, "label", "")).lower()
                      for item in chunk.meta.doc_items)
    for needle, label in (("table", "table"), ("picture", "figure"),
                          ("figure", "figure"), ("formula", "formula"),
                          ("equation", "formula"), ("code", "code")):
        if needle in labels:
            return label
    return "text"


def is_furniture(body: str) -> str | None:
    """Reason this text is page furniture, or None if it should be indexed.

    Takes the BODY, not the contextualized text — the heading path is
    prepended to everything and would defeat every anchored pattern.

    Returns a reason rather than a bool so the caller can print what it
    removed. A silent deletion is one nobody checks.
    """
    stripped = body.strip()
    if not stripped:
        return "empty"
    if _tokens(stripped) > FURNITURE_MAX_TOKENS:
        return None
    for pattern, reason in FURNITURE_PATTERNS:
        if pattern.match(stripped):
            return reason
    return None


def document_date(pdf: Path, head: str) -> str:
    """When the document is ABOUT — not when it was processed.

    A corpus accumulates versions of the same report, and "relevant but two
    years old" is a failure users hit constantly. `ingested_at` is the other
    thing and is not this.

    The heuristic is deliberately simple and will be wrong on documents whose
    first four-digit number is a protocol code or a street address. Replace it
    with a real header parse where dates carry weight.
    """
    year = re.search(r"\b(19|20)\d{2}\b", head)
    if year:
        quarter = re.search(r"\bQ([1-4])\s*(19|20)\d{2}\b", head)
        return f"{year.group(0)}-Q{quarter.group(1)}" if quarter else year.group(0)
    return datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m")


def _section_id(headings: list[str]) -> str:
    """Groups every record under the same top-level heading.

    Nothing reads it yet. Written now because adding it later means
    re-embedding the corpus.
    """
    return hashlib.sha256(
        (headings[0] if headings else "").encode()).hexdigest()[:12]


# ════════════════════════════════════════════════════════════════════════════
# THE CHUNKER
# ════════════════════════════════════════════════════════════════════════════

def _make_chunker():
    """HybridChunker, configured.

    Two settings carry the weight:

      serializer   tables as markdown, not Docling's `**Column**, row 1 = x`
                   triplet form, which flattens the grid and embeds badly.

      merge_peers  OFF. Its predicate is heading equality, NOT element type,
                   so it welds prose to tables — measured: one 1007-token
                   chunk holding four paragraphs, a figure and a contact
                   table. _merge_prose does the same merge, prose only.

    There is no min_tokens parameter, whatever some doc mirrors say. Pydantic
    ignores unknown keyword arguments, so passing one looks like it worked.
    DESIGN.md has the full case.
    """
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer, ChunkingSerializerProvider,
    )
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

    class MarkdownTableProvider(ChunkingSerializerProvider):
        # The parameter must be named `doc` — HybridChunker calls this by
        # keyword. **kwargs absorbs whatever a future release adds.
        def get_serializer(self, doc, **kwargs):
            return ChunkingDocSerializer(
                doc=doc, table_serializer=MarkdownTableSerializer())

    return HybridChunker(
        tokenizer=OpenAITokenizer(tokenizer=ENCODING, max_tokens=CHUNK_TOKENS),
        serializer_provider=MarkdownTableProvider(),
        merge_peers=False,
    )


# ════════════════════════════════════════════════════════════════════════════
# PASS A — chunks become entries
# ════════════════════════════════════════════════════════════════════════════

def _to_entries(chunks, chunker, figure_uris: dict) -> tuple[list[dict], list]:
    """Classify each chunk, drop page furniture, mark figure slots.

    An entry is a plain dict, not a Docling chunk — later passes merge and
    reorder, and rebuilding DocMeta by hand buys nothing.

        kind    "text" | "table" | "figure_slot" | "formula" | "code"
        body    serialized content, WITHOUT the heading path
        head    the heading path, prepended at record time

    A chunk whose only content is pictures and their captions becomes a
    figure_slot holding the refs; _to_records expands it to one record per
    picture. Captions must be allowed in: when the layout model DOES link one,
    the serializer emits caption and picture together, and a stricter test
    rejects exactly the well-formed case.

    Returns:
        (entries, dropped) where dropped is [(page, reason, sample)].
    """
    entries, dropped = [], []

    for chunk in chunks:
        items = chunk.meta.doc_items
        headings = list(chunk.meta.headings or [])
        pages = sorted({prov.page_no
                        for item in items
                        for prov in (getattr(item, "prov", []) or [])})
        page = pages[0] if pages else -1
        page_end = pages[-1] if pages else -1

        labels = [str(getattr(i, "label", "")).lower() for i in items]
        pictures = [i for i, label in zip(items, labels) if "picture" in label]
        beside = [label for label in labels if "picture" not in label]

        # Pictures, alone or with their captions. A table disqualifies it —
        # tables carry values that must be indexed as themselves.
        if pictures and all("caption" in label for label in beside):
            entries.append({
                "kind": "figure_slot",
                "refs": [r for i in pictures
                         if (r := getattr(i, "self_ref", None))],
                "head": headings, "page": page, "page_end": page_end,
            })
            continue

        kind = content_type(chunk)

        # Text only. A table or a figure is never dropped.
        if kind == "text":
            reason = is_furniture(chunk.text)
            if reason:
                dropped.append((page, reason, chunk.text.strip()[:60]))
                continue

        image_uri = ""
        for item in items:
            key = getattr(item, "self_ref", None)
            if key and key in (figure_uris or {}):
                image_uri = figure_uris[key]
                break

        entries.append({
            "kind": kind,
            "body": chunk.text,
            # Kept for entries that survive unmerged, so their embedded text
            # is byte-identical to what the chunker would have produced.
            "contextualized": chunker.contextualize(chunk=chunk),
            "merged": False,
            "head": headings, "page": page, "page_end": page_end,
            "ref": table_ref_of(chunk),
            "image_uri": image_uri,
        })

    return entries, dropped


# ════════════════════════════════════════════════════════════════════════════
# PASS B — merge adjacent prose, then apply the floor
# ════════════════════════════════════════════════════════════════════════════

def _merge_prose(entries: list[dict]) -> tuple[list[dict], int]:
    """Merge consecutive text entries that share a heading path.

    The chunker's own predicate — consecutive, equal heading path, under a
    budget — plus the type check it does not offer: BOTH sides must be text.

    With MERGE_ACROSS_EXHIBITS the search reaches back past a figure or table,
    stopping at the first entry under a DIFFERENT heading. Without it, a
    document whose prose is interleaved with exhibits never merges at all,
    because every run is length one.

    Does NOT cross a heading boundary — that is _apply_floor's job, and only
    when asked. Does NOT reorder: a merged entry keeps its first position.
    """
    out: list[dict] = []
    merges = 0

    for entry in entries:
        target = None

        if entry["kind"] == "text":
            for candidate in reversed(out):
                if candidate["head"] != entry["head"]:
                    break
                if candidate["kind"] == "text":
                    target = candidate
                    break
                if not MERGE_ACROSS_EXHIBITS:
                    break

        if (target is not None
                and _tokens(target["body"] + "\n" + entry["body"])
                <= PROSE_TARGET_TOKENS):
            target["body"] += "\n" + entry["body"]
            target["page_end"] = max(target["page_end"], entry["page_end"])
            target["merged"] = True
            target["image_uri"] = target["image_uri"] or entry["image_uri"]
            merges += 1
        else:
            out.append(entry)

    return out, merges


def _apply_floor(entries: list[dict]) -> tuple[list[dict], int]:
    """Small text entries absorb forward, ACROSS heading boundaries.

    The only rule here that crosses a heading, and it runs only when
    MIN_CHUNK_TOKENS > 0. It absorbs while the accumulated entry is still
    under the floor and stops the moment it clears — so a record that was
    already big enough is never touched.

    Tables, figures and code end the run. Each heading crossed is written into
    the body so its context is still embedded; `headings` metadata keeps the
    first path, so section_id and filtering stay stable.

    The trade: fewer, fatter, less precise records. A form wants this; a
    document whose sections are real does not.
    """
    if MIN_CHUNK_TOKENS <= 0:
        return entries, 0

    out: list[dict] = []
    merges = 0

    for entry in entries:
        previous = out[-1] if out else None

        if (previous is not None
                and previous["kind"] == "text" and entry["kind"] == "text"
                and _tokens(previous["body"]) < MIN_CHUNK_TOKENS):
            crossed = ([h for h in entry["head"] if h not in previous["head"]]
                       if entry["head"] != previous["head"] else [])
            previous["body"] += "\n" + "\n".join([*crossed, entry["body"]])
            previous["page_end"] = max(previous["page_end"], entry["page_end"])
            previous["merged"] = True
            previous["image_uri"] = previous["image_uri"] or entry["image_uri"]
            merges += 1
        else:
            out.append(entry)

    return out, merges


# ════════════════════════════════════════════════════════════════════════════
# PASS C — entries become records
# ════════════════════════════════════════════════════════════════════════════

def _figure_records(entry: dict, doc, items_by_ref: dict, figure_uris: dict,
                    make) -> tuple[list[dict], int, int]:
    """Expand one figure slot into one record per picture.

    Each figure description is a self-contained fact about a different chart.
    Packing several into one vector produces a vector representing none of
    them.

    The caption comes from the DOCUMENT, via caption_text(doc). Without it the
    caption becomes a separate 25-token chunk saying "Exhibit 18: ..." and
    nothing else, while the figure it names carries no exhibit number — so
    neither is retrievable by the string a reader would search for.

    Returns:
        (records, indexed, skipped)
    """
    records, indexed, skipped = [], 0, 0

    for ref in entry["refs"]:
        item = items_by_ref.get(ref)
        if item is None:
            continue

        description = picture_description(item)
        if not description:
            # A record saying only "a figure was here" matches every query
            # about figures and answers none. The extraction report flags it.
            skipped += 1
            continue

        try:
            caption = (item.caption_text(doc) or "").strip()
        except Exception:
            caption = ""

        # Context first, then content — the same shape as every other record.
        # Built by hand because contextualize() takes a chunk and this is not
        # one.
        parts = [*entry["head"]]
        if caption:
            parts.append(caption)
        parts.append(description)
        text = "\n".join(parts)

        # Chart series in the embedded text, not metadata only, so a question
        # about a specific value has something to match.
        series = chart_data(item)
        if series is not None:
            text += f"\n\nchart data: {str(series)[:800]}"

        records.append(make(text, {
            "page": entry["page"], "page_end": entry["page_end"],
            "headings": entry["head"],
            "section_id": _section_id(entry["head"]),
            "content_type": "figure",
            "table_id": "",
            "image_uri": (figure_uris or {}).get(ref, ""),
            "has_caption": bool(caption),
            "has_chart_data": series is not None,
        }, entry["position"]))
        indexed += 1

    return records, indexed, skipped


def _to_records(entries: list[dict], doc, items_by_ref: dict,
                figure_uris: dict, make) -> tuple[list[dict], dict, dict]:
    """Entries to records, expanding figure slots.

        ENTRY IN                              RECORD OUT

        {"kind": "text",                      {"text": "...",  <- what gets embedded
         "body": "...",                        "meta": {
         "contextualized": "...",                 "chunk_id": "doc:hash:0",
         "merged": False,                         "content_hash": "hash",
         "head": ["Section A"],                    "occurrence": 0,
         "page": 1, "page_end": 1,                 "doc_id": "...", "source": "...",
         "ref": None,                              "doc_date": "2025",
         "image_uri": ""}                          "ingested_at": 169...,
                                                     "position": 4,
                                                     "embed_model": "...",
                                                     "access": ["public"],
                                                     "n_tokens": 221,
                                                     "page": 1, "page_end": 1,
                                                     "headings": ["Section A"],
                                                     "section_id": "a3f9c1...",
                                                     "content_type": "text",
                                                     "table_id": "",
                                                     "image_uri": "",
                                                     # added later, by _finalise:
                                                     "n_positions": 34,
                                                     "prev_id": "...",
                                                     "next_id": "...",
                                                 }}

    A `figure_slot` entry is different going in — no `body` yet, just refs to
    resolve:

        {"kind": "figure_slot", "refs": ["#/pictures/2"],
         "head": [...], "page": 2, "page_end": 2}

    One entry becomes exactly one record — EXCEPT a figure_slot, which
    becomes zero or more (one per picture inside it, via _figure_records),
    and a table entry, which also seeds `table_groups` for _table_summaries
    to turn into one further record.

    Builds two things that are not the same:

        records       every record, in reading order
        table_groups  those records belonging to a table, grouped by table

    A large table appears once in `records` per fragment, and once in
    `table_groups` as the list of all its fragments.
    """
    records: list[dict] = []
    table_groups: defaultdict = defaultdict(list)
    indexed = skipped = 0
    refs_seen: set = set()

    for position, entry in enumerate(entries):
        entry["position"] = position

        if entry["kind"] == "figure_slot":
            refs_seen.update(entry["refs"])
            new, n_indexed, n_skipped = _figure_records(
                entry, doc, items_by_ref, figure_uris, make)
            records += new
            indexed += n_indexed
            skipped += n_skipped
            continue

        # An unmerged entry keeps the chunker's own contextualized string. A
        # merged one is rebuilt: joining two contextualized strings would
        # repeat the heading path in the middle of the chunk.
        text = ("\n".join([*entry["head"], entry["body"]]) if entry["merged"]
                else entry["contextualized"])

        ref = entry["ref"]
        record = make(text, {
            "page": entry["page"], "page_end": entry["page_end"],
            # A list, not a joined string: Pinecone filters a list with $in
            # and cannot filter a comma-joined string at all.
            "headings": entry["head"],
            "section_id": _section_id(entry["head"]),
            "content_type": entry["kind"],
            "table_id": hashlib.sha256(ref.encode()).hexdigest()[:12] if ref else "",
            "image_uri": entry["image_uri"],
        }, position)

        records.append(record)
        if ref:
            table_groups[ref].append(record)

    # Pictures that never reached a slot, because they shared a chunk with
    # something else. Their description is inside that chunk's text, so they
    # are not separately retrievable — worth knowing.
    merged_in = sum(
        1 for item, _ in doc.iterate_items()
        if isinstance(item, PictureItem)
        and getattr(item, "self_ref", None) not in refs_seen)

    stats = {"figures_indexed": indexed, "figures_skipped": skipped,
             "figures_merged": merged_in}
    return records, table_groups, stats


# ════════════════════════════════════════════════════════════════════════════
# PASS D — one summary per table
# ════════════════════════════════════════════════════════════════════════════

def _table_summaries(table_groups: dict, tables: dict, doc,
                     items_by_ref: dict, make) -> tuple[list[dict], dict]:
    """One summary per logical table, however many fragments it produced.

        logical table
              |
         +----+----+
         v    v    v
       chunk chunk chunk        fragments, each already its own record
         +----+----+
              |
         same table_id
              |
              v
         ONE summary            an ADDITIONAL record

    The fragments hold the exact values; the summary holds the words someone
    would search for. Neither replaces the other.

    Read from the complete grid on the document, not stitched back together
    from fragments. When the grid itself is wrong, the rendered image is
    described instead — summarising a broken grid produces a confident
    description of a table that does not exist.
    """
    records, summarised, skipped, repaired = [], 0, [], []

    for ref, fragments in table_groups.items():
        first = fragments[0]["meta"]
        markdown = tables.get(ref, "")

        if not markdown:
            skipped.append((first["page"], "could not be serialised"))
            continue
        if not needs_summary(markdown):
            skipped.append((first["page"],
                            f"{len(table_cells(markdown))} cells, treated as layout"))
            continue

        problems = table_looks_broken(markdown)
        summary, source = None, "markdown"

        if problems:
            item = items_by_ref.get(ref)
            if item is None:
                print(f"    table on p{first['page']} is broken and its element "
                      "could not be found to render", flush=True)
            else:
                try:
                    summary = summarize_table_image(item, doc, first["headings"])
                    if summary is None:
                        # get_image() returned nothing — almost always a parse
                        # made before generate_table_images was enabled.
                        print(f"    table on p{first['page']} is broken but has "
                              "no rendered image; the parse predates "
                              "generate_table_images. Delete the cache and "
                              "re-parse.", flush=True)
                    else:
                        source = "image"
                except Exception as exc:
                    print(f"    table image fallback failed on "
                          f"p{first['page']}: {exc}", flush=True)
            repaired.append((first["page"], problems, summary is not None))

        if summary is None:
            summary = summarize_table(markdown, first["headings"])
            source = "markdown"

        records.append(make(summary, {
            "page": first["page"], "page_end": fragments[-1]["meta"]["page_end"],
            "headings": first["headings"], "section_id": first["section_id"],
            "content_type": "table_summary",
            # Same table_id as the fragments: how retrieval walks from a
            # matched summary to the rows carrying the exact values.
            "table_id": first["table_id"],
            "table_chars": len(markdown),
            "n_fragments": len(fragments),
            "summary_source": source,
        }, first["position"]))
        summarised += 1

    return records, {"summarised": summarised, "skipped": skipped,
                     "repaired": repaired}


# ════════════════════════════════════════════════════════════════════════════
# PASS E — reading order, links, size limits
# ════════════════════════════════════════════════════════════════════════════

def _finalise(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Sort into reading order, link neighbours, bound metadata, truncate.

    Summaries were appended last but carry the position of their table's first
    fragment, so sorting on (position, is-not-a-summary) puts each one
    immediately before the rows it describes.

    prev_id / next_id are stored explicitly because content-addressed ids
    cannot be derived from position — you cannot compute record 43's id from
    record 42's.

    Returns:
        (records, truncated) — truncated is the ones that lost content.
    """
    records.sort(key=lambda r: (r["meta"]["position"],
                                r["meta"]["content_type"] != "table_summary"))

    for i, record in enumerate(records):
        record["meta"]["position"] = i
        record["meta"]["n_positions"] = len(records)
        if i:
            record["meta"]["prev_id"] = records[i - 1]["meta"]["chunk_id"]
        if i + 1 < len(records):
            record["meta"]["next_id"] = records[i + 1]["meta"]["chunk_id"]

    if not records:
        return records, []

    # Pinecone caps metadata at 40 KB per vector. Measure what the structural
    # fields cost and give the text whatever is left, rather than guessing a
    # character limit. record["text"] is embedded; meta["text"] is a bounded
    # copy so retrieval can read it off the query result.
    overhead = len(json.dumps({**records[0]["meta"], "text": ""}).encode())
    budget = max(512, PINECONE_METADATA_BYTES - overhead - 1024)
    for record in records:
        record["meta"]["text"] = record["text"][:budget]

    # The chunker cannot split an atomic element smaller than itself: one
    # enormous table row, one very long code block. Truncating loses that one
    # item; raising would abort a 250-page document over a single row. The
    # flag makes the loss visible instead of silent.
    over = [r for r in records if r["meta"]["n_tokens"] > CHUNK_TOKENS]
    for record in over:
        record["text"] = ENCODING.decode(
            ENCODING.encode(record["text"])[:CHUNK_TOKENS])
        record["meta"]["n_tokens"] = CHUNK_TOKENS
        record["meta"]["truncated"] = True

    return records, over


# ════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════

def _report(records, chunks, tables, table_groups, stats) -> None:
    """What happened, in enough detail to act on.

    Not "document processed successfully". Extraction and chunking fail
    quietly; this is where that becomes visible, before the data reaches
    retrieval and someone wonders why the answers are wrong.
    """
    sizes = sorted(r["meta"]["n_tokens"] for r in records)
    types = Counter(r["meta"]["content_type"] for r in records)

    print(f"  {len(records)} records from {len(chunks)} chunks "
          f"({stats['summarised']} of {len(tables)} tables summarised)",
          flush=True)
    print(f"  types: {dict(types)}", flush=True)
    print(f"  size: median {sizes[len(sizes) // 2]} tokens, "
          f"{sum(1 for s in sizes if s < 50)} under 50, max {sizes[-1]}",
          flush=True)

    dropped = stats["dropped"]
    if dropped:
        reasons = Counter(reason for _, reason, _ in dropped)
        print(f"  dropped {len(dropped)} chunk(s) as page furniture: "
              f"{dict(reasons)}", flush=True)
        for page, reason, sample in dropped:
            print(f"    p{page} {reason}: {sample!r}", flush=True)

    if stats["merges"]:
        print(f"  merged {stats['merges']} adjacent prose chunk(s) under a "
              "shared heading", flush=True)
    if stats["floor_merges"]:
        print(f"  merged {stats['floor_merges']} chunk(s) ACROSS a heading "
              f"boundary to reach the {MIN_CHUNK_TOKENS}-token floor. Records "
              "now answer about more than one section each.", flush=True)

    indexed = stats["figures_indexed"]
    captioned = sum(1 for r in records
                    if r["meta"].get("content_type") == "figure"
                    and r["meta"].get("has_caption"))
    print(f"  figures: {indexed} indexed one-per-record, "
          f"{captioned} with a linked caption"
          + (f", {stats['figures_skipped']} with no description"
             if stats["figures_skipped"] else "")
          + (f", {stats['figures_merged']} inside a mixed chunk"
             if stats["figures_merged"] else ""), flush=True)
    if indexed and captioned < indexed:
        print(f"  NOTE: {indexed - captioned} figure(s) have no caption linked "
              "in the parse. They are retrievable by description but not by "
              "exhibit number unless the vision model happened to read it.",
              flush=True)
    if stats["figures_merged"]:
        print(f"  NOTE: {stats['figures_merged']} figure(s) share a chunk with "
              "other content, so they are not separately retrievable.",
              flush=True)

    for page, reason in stats["skipped"]:
        print(f"    skipped table on p{page}: {reason}", flush=True)
    for page, problems, used_image in stats["repaired"]:
        route = ("described from the rendered image" if used_image
                 else "FELL BACK TO BROKEN MARKDOWN — the summary may be wrong")
        print(f"    table on p{page} has bad structure "
              f"({'; '.join(problems)}): {route}", flush=True)
    if stats["truncated"]:
        print(f"  WARNING: truncated {len(stats['truncated'])} unsplittable "
              f"chunks to {CHUNK_TOKENS} tokens", flush=True)

    # A table found by extraction but never reaching a record is invisible in
    # the loop above, so compare the counts directly. Zero groups against a
    # non-zero table count means the chunk-to-table link is broken, not that
    # the document lacks tables.
    orphaned = len(tables) - len(table_groups)
    if tables and not table_groups:
        print(f"  ERROR: {len(tables)} tables were extracted but none could be "
              "linked to a record. No summaries were generated and table_id is "
              "empty on every record. Check table_ref_of() against this docling "
              "version.", flush=True)
    elif orphaned:
        print(f"  WARNING: {orphaned} of {len(tables)} tables produced no "
              "record", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# RECORD IDENTITY
# ════════════════════════════════════════════════════════════════════════════

def _record_factory(pdf: Path, doc_id: str, doc_date: str):
    """Returns make(text, meta_extra, position) -> record.

    CONTENT-ADDRESSED IDS, and this single decision is what makes the
    incremental sync in sync.py possible:

        chunk_id = {doc_id}:{sha256(text)[:16]}:{occurrence}

      doc_id      scopes the hash. Two PDFs sharing a boilerplate disclaimer
                  would otherwise produce the same id, and upsert is
                  last-write-wins — one document silently deletes the other's
                  chunk, with no error anywhere.

      hash        a positional id changes for every chunk after an edit,
                  forcing a full re-embed when one paragraph changed. A hash
                  changes only where the text changed.

      occurrence  the same text can legitimately repeat — a disclaimer on
                  every page. Without a counter all copies collapse into one
                  record and the other page numbers are lost.

    Note what occurrence does NOT solve: repeated text that is worthless stays
    repeated, once per copy. That is the drop filter's job, and it runs first.
    """
    ingested_at = int(datetime.now(timezone.utc).timestamp())
    occurrences: defaultdict = defaultdict(int)

    def make(text: str, meta_extra: dict, position: int) -> dict:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        occurrence = occurrences[digest]
        occurrences[digest] += 1
        return {"text": text, "meta": {
            "chunk_id": f"{doc_id}:{digest}:{occurrence}",
            "content_hash": digest, "occurrence": occurrence,
            "doc_id": doc_id, "source": pdf.name,
            "doc_date": doc_date, "ingested_at": ingested_at,
            "position": position,
            # Recording the embedding model lets retrieval refuse to query an
            # index built with a different one. That mismatch produces
            # plausible rankings and no error.
            "embed_model": EMBED_MODEL, "access": ACCESS_GROUPS,
            "n_tokens": _tokens(text),
            **meta_extra,
        }}

    return make


# ════════════════════════════════════════════════════════════════════════════
# THE PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def build_records(doc, pdf: Path, doc_id: str, doc_date: str,
                  figure_uris: dict[str, str] | None = None) -> list[dict]:
    """Turn a parsed document into the records that will be embedded.

    Returns records. Does NOT embed them and does NOT write them anywhere.
    """
    # Headings first. HybridChunker never merges across a heading boundary and
    # the heading path is its whole merge predicate, so a heading the layout
    # model got wrong is both a boundary that should not exist and a wrong
    # string prepended into every vector beneath it.
    clean_headings(doc)

    chunker = _make_chunker()
    chunks = list(chunker.chunk(dl_doc=doc))
    tables = table_markdown(doc)

    # Original elements by ref. NOT for grouping table fragments — that is
    # table_groups. This is for recovering a TableItem whose markdown came out
    # wrong, so it can be rendered and looked at.
    items_by_ref = {}
    for item, _ in doc.iterate_items():
        ref = getattr(item, "self_ref", None)
        if ref:
            items_by_ref[ref] = item

    # The id factory. Created once per document because it holds the
    # occurrence counter — the thing that keeps a disclaimer repeated on every
    # page as several distinct records instead of one.
    make = _record_factory(pdf, doc_id, doc_date)
    uris = figure_uris or {}

    # ── the three chunk-shaping passes ───────────────────────────────────
    # Each takes a list of entries and returns a shorter one. Nothing here
    # touches the document again, so these are pure list transformations and
    # can be reasoned about (and tested) on their own.
    #
    #   _to_entries    chunks -> entries. Drops furniture, marks figure slots.
    #   _merge_prose   joins text under a shared heading, up to the target.
    #   _apply_floor   the only pass that crosses a heading. Off by default.
    #
    # Order matters. Dropping before merging is what stops six identical
    # "Source: ..." lines merging into one 96-token chunk of nothing.
    entries, dropped = _to_entries(chunks, chunker, uris)
    entries, merges = _merge_prose(entries)
    entries, floor_merges = _apply_floor(entries)

    # Entries become records, and figure slots expand here — one record per
    # picture. `table_groups` comes back alongside: the same table records,
    # grouped by which logical table they came from, which is what the next
    # step needs.
    records, table_groups, fig_stats = _to_records(
        entries, doc, items_by_ref, uris, make)

    # One summary per table, ADDED to the records rather than replacing the
    # fragments. Runs after _to_records because it needs the fragments' ids
    # and positions to link and place each summary.
    summaries, table_stats = _table_summaries(
        table_groups, tables, doc, items_by_ref, make)
    records += summaries

    # Sort, link neighbours, bound the metadata, truncate anything oversized.
    # Must run last: it assigns final positions, and the summaries appended
    # above still carry their table's position rather than their own.
    records, truncated = _finalise(records)

    # An empty result is legitimate — a cover page, or a document that was
    # entirely furniture. Say so rather than printing a report about nothing.
    if not records:
        print("  no records — every chunk was dropped or the document is empty",
              flush=True)
        return records

    # Counters from every pass, merged into one dict. Collected rather than
    # printed as they happen so the summary reads in a sensible order instead
    # of interleaved with the work.
    _report(records, chunks, tables, table_groups,
            {**fig_stats, **table_stats, "dropped": dropped, "merges": merges,
             "floor_merges": floor_merges, "truncated": truncated})
    return records
