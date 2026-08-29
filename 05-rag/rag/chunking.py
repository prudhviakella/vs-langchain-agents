"""Turning a parsed document into the records that will be embedded.

    parsed document
          |
          v
    clean_headings()         false headings demoted, before any boundary is set
          |
          v
    HybridChunker            splits on structure, then on a token budget
          |                  merge_peers=False — see STEP 1
          v
    pass A: chunk -> entry   classify, drop page furniture, mark figure slots
          |
          v
    pass B: merge prose      adjacent text entries, same heading path
          |
          v
    pass C: entry -> record  figure slots expand to one record per figure
          |
          v
    pass D: one summary per table    extra records, not replacements
          |
          v
    pass E: reading order, prev/next links, size limits
          |
          v
      records[]
          |
          v
    embedding.py  ->  vectors  ->  index.py  ->  Pinecone

WHAT THIS FILE DOES NOT DO

    It does NOT call the embedding model.
    It does NOT write to Pinecone.
    It does NOT parse the PDF or run any vision model.

It prepares records. `embedding.py` turns them into vectors and `sync.py`
writes them. One PDF does not produce one embedding — it produces one per
record.

THE THREE THINGS THE CHUNKER WILL NOT DO FOR YOU

HybridChunker only ever splits, and merges consecutive chunks with an equal
heading path. Read the source if you want to confirm this; it is four stages
and the only one that combines anything is the last:

    HierarchicalChunker      one chunk per detected element
    _split_by_doc_items      window the items to fit the token budget
    _split_using_plain_text  semchunk whatever is still oversized
    _merge_chunks_with_matching_metadata     only when merge_peers=True

Nothing in there filters, nothing enforces a minimum, and the merge is blind
to element type. So three policies are ours:

    1. what is not worth indexing        pass A, the drop filter
    2. how small a prose chunk may be    pass B, the merge
    3. figures are one record each       pass C

EACH FIGURE GETS ITS OWN RECORD

Each figure description is a self-contained fact about a different chart.
Packing several into one vector produces a vector that represents none of
them, and hands the model five descriptions when it asked about one.

A LARGE TABLE IS NOT FORCED INTO ONE CHUNK

    logical table
          |
     +----+----+
     v    v    v
   chunk chunk chunk        physical fragments, each its own record
     +----+----+
          |
     same table_id
          |
          v
     ONE summary            an additional record, sharing their table_id

The fragments hold the exact values. The summary holds the words someone
would search for. Neither replaces the other.

Chunk identity is content-addressed. That single decision is what makes the
incremental sync in `sync.py` possible, and what makes a retried run safe.
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

# ═══════════════════════════════════════════════════════════════════════════
# TUNING
#
# These are NOT in config.py, and that is deliberate. config.py holds the
# settings the ingestion and retrieval halves must agree on — get one wrong
# and queries silently return nonsense. These two only shape what goes into
# the index. Retrieval never reads them.
# ═══════════════════════════════════════════════════════════════════════════

# Below this, a text chunk is a candidate for the drop filter. Above it, the
# chunk is kept whatever it looks like — a long passage that happens to start
# with "Source:" is a passage, not a source line.
FURNITURE_MAX_TOKENS = 15

# Pass B stops merging once a prose entry reaches this. Well under
# CHUNK_TOKENS, because the merge should improve small chunks, not manufacture
# maximal ones — a 1,000-token chunk retrieves a page when the answer is a
# paragraph.
PROSE_TARGET_TOKENS = 400

# Whether pass B may merge two prose entries that have an exhibit between
# them. See the STEP 6 banner for what this costs. Measured on a 7-page
# research report:
#
#     off   47 records, prose median  61 tokens
#     on    37 records, prose median 171 tokens
#
# On a document whose prose runs in uninterrupted blocks this changes
# nothing, because the adjacent case already caught everything.
MERGE_ACROSS_EXHIBITS = True

# A FLOOR, and the only rule here that crosses a heading boundary.
#
# 0 disables it. Anything above 0 means: a prose record smaller than this
# absorbs the next one even if they are under different headings, stopping as
# soon as it clears the floor.
#
# WHY THIS EXISTS
#
# The chunker never merges across a heading, and that is right when headings
# mean sections. On a form it is not. Measured on a 15-page IRB protocol with
# 57 headings across 15 pages:
#
#     43 of 48 heading paths owned exactly ONE record
#     median 86 tokens, 18 records under 30
#
#     'Section N:  Sample Collection\nNone'                    8 tokens
#     'A5.  Funding Source:\nBaylor College (Internal Only)'  17 tokens
#
# Each is a complete answer to one question, which is the argument for leaving
# them alone. Against that: retrieval returns k of them, and k fragments of 15
# tokens is 200 tokens of context for the model to answer from. At corpus
# scale it is thousands of near-identical vectors whose embedding is mostly
# the heading.
#
# WHAT IT COSTS
#
# It joins sections that are genuinely separate. A record can then answer
# about funding AND about institution, and a query for one retrieves both.
# That is the trade: fewer, fatter, less precise records.
#
# Every heading crossed is kept IN THE TEXT, so the context of each part
# survives and is embedded:
#
#     A5.  Funding Source:
#     Baylor College of Medicine (Internal Funding Only)
#     A6a.  Institution(s) where work will be performed:
#     Baylor College of Medicine
#
# `headings` metadata keeps only the FIRST path, so section_id and filtering
# stay stable.
#
# Measured on that protocol: 150 -> 32 records median 280; 250 -> 27 records
# median 363; 400 -> 18 records median 501.
#
# Set it per corpus. A form wants a floor. A prose report does not — its
# sections are real, and 0 leaves this pass switched off entirely.
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "0"))

# ═══════════════════════════════════════════════════════════════════════════
# WHAT IS NOT WORTH A VECTOR
#
# Every pattern below came from a real chunk in a real run, not from
# imagination. On a 7-page research report these matched 20 of 55 text
# chunks — attribution lines, and the publisher's logo glyph read as a text
# element on every page.
#
# Six of those 20 were byte-identical to each other. Identical text embeds to
# an identical vector, so they were six copies of one point in the index,
# collectively able to occupy an entire top-k.
#
# WHY THIS IS NOT A PARSING BUG
#
# The page really does say "Source: Morgan Stanley Research". Docling read it
# correctly and reported it as a TEXT element. There is no parse setting that
# suppresses it, and there should not be — the extraction report is supposed
# to show you everything on the page. The judgement about what deserves a
# vector belongs here, at the point where vectors are decided.
#
# THE RISK, STATED PLAINLY
#
# A pattern that is too greedy deletes content, and nothing downstream can
# tell. That is why the token ceiling above is a hard gate, why every drop is
# counted and printed, and why the patterns are anchored at the start of the
# string rather than searched anywhere in it.
# ═══════════════════════════════════════════════════════════════════════════
FURNITURE_PATTERNS = [
    # Attribution lines under an exhibit. "Source: FactSet, Morgan Stanley
    # Research" carries no fact a question could be asked about.
    (re.compile(r"^sources?\s*:", re.IGNORECASE), "an attribution line"),
    # A publisher's logo, OCR'd to one or two letters, on every page.
    (re.compile(r"^[A-Za-z]{1,2}$"), "a single glyph"),
    # Running headers and footers.
    (re.compile(r"^page\s+\d+\b", re.IGNORECASE), "a page marker"),
    (re.compile(r"^\d{1,3}$"), "a bare page number"),
]


def content_type(chunk) -> str:
    """Coarse type: text, table, table_summary, figure, formula or code.

        chunk  ->  read its element labels  ->  first special type wins

    This does NOT read or summarise the content. It answers one question:
    what KIND of thing is this, so a query can filter on it.

    Order matters — a chunk holding a table and its caption is a table, not
    text.

    This is also how you would tell apart a pipeline that handles prose well
    and tables badly from one that is mediocre at both, which look identical
    in a single overall score.
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

        body  ->  over the token ceiling?  ->  keep, always
               ->  matches a pattern?      ->  the reason
               ->  otherwise               ->  keep

    Takes the BODY, not the contextualized text. The heading path is
    prepended to everything and would defeat every anchored pattern here.

    Returns a reason rather than a bool so the caller can print what it
    removed. A silent deletion is one nobody checks.
    """
    stripped = body.strip()
    if not stripped:
        return "empty"
    if len(ENCODING.encode(stripped)) > FURNITURE_MAX_TOKENS:
        return None
    for pattern, reason in FURNITURE_PATTERNS:
        if pattern.match(stripped):
            return reason
    return None


def document_date(pdf: Path, head: str) -> str:
    """Publication date of the document, for filtering stale content at query time.

        opening text  ->  find a year  ->  find a quarter  ->  "2025" or "2025-Q3"
                                                    |
                                            none found?
                                                    |
                                          the PDF's own mtime

    NOT the same as `ingested_at`. This is when the document is ABOUT; that is
    when we processed it. A corpus accumulates versions of the same report,
    and "relevant but two years old" is a failure users hit constantly.

    The heuristic — first year, plus a quarter if present, in the opening
    pages — is deliberately simple and will be wrong on documents whose first
    four-digit number is a reference or a street address. Replace it with a
    real header parse for any corpus where dates carry weight.
    """
    year = re.search(r"\b(19|20)\d{2}\b", head)
    if year:
        quarter = re.search(r"\bQ([1-4])\s*(19|20)\d{2}\b", head)
        return f"{year.group(0)}-Q{quarter.group(1)}" if quarter else year.group(0)
    return datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m")


def build_records(doc, pdf: Path, doc_id: str, doc_date: str,
                  figure_uris: dict[str, str] | None = None) -> list[dict]:
    """Turn a parsed document into the records that will be embedded and indexed.

        1  configure the chunker    structure first, then token budget
        2  correct headings, chunk
        3  index elements by ref    for recovering a broken table
        4  record identity          content-addressed ids
        5  pass A                   chunks -> entries, furniture dropped
        6  pass B                   merge adjacent prose
        7  pass C                   entries -> records, figures expanded
        8  pass D                   one summary per table
        9  pass E                   reading order and prev/next links
       10  metadata budget          Pinecone caps it at 40 KB
       11  oversized chunks         truncate, and say so
       12  diagnostics              what went wrong, before anyone asks

    Returns records. Does NOT embed them and does NOT write them anywhere.
    """
    # ═════════════════════════════════════════════════════════════════════
    # STEP 1 — configure the chunker
    #
    #   tokenizer            how to count            the embedding model's own
    #   max_tokens           how big a chunk may be  measured in tokens
    #   serializer_provider  how elements become text
    #
    # Sizing in characters is a proxy that fails silently: the API accepts an
    # oversized input, embeds the first N tokens, and returns a valid-looking
    # vector for half a chunk.
    # ═════════════════════════════════════════════════════════════════════
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer, ChunkingSerializerProvider,
    )
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

    class MarkdownTableProvider(ChunkingSerializerProvider):
        """Serialize tables as markdown rather than Docling's default triplet form.

        The default renders a cell as `**Column**, row 1 = value`, which
        flattens the grid and embeds poorly: the relationship between a value
        and its column header is exactly what a tabular query depends on.
        """

        # The parameter must be named `doc`: HybridChunker calls this by
        # keyword, as get_serializer(doc=dl_doc). **kwargs absorbs anything a
        # future release adds rather than raising on an unexpected argument.
        def get_serializer(self, doc, **kwargs):
            """Return a serializer that renders tables as markdown."""
            return ChunkingDocSerializer(
                doc=doc, table_serializer=MarkdownTableSerializer())

    chunker = HybridChunker(
        tokenizer=OpenAITokenizer(tokenizer=ENCODING, max_tokens=CHUNK_TOKENS),
        serializer_provider=MarkdownTableProvider(),
        # ─────────────────────────────────────────────────────────────────
        # merge_peers OFF, and this is the most consequential setting here.
        #
        # WHAT IT ACTUALLY DOES — from the source, not the docs:
        #
        #     if headings == current_headings and fits(candidate):
        #         merge
        #
        # Two conditions, both mechanical. The heading paths must be EQUAL,
        # the chunks must be CONSECUTIVE, and the result must fit the budget.
        # Element type is never consulted. "Peers" does not mean siblings in
        # the tree; it means an equal heading path.
        #
        # WHY THAT IS WRONG HERE
        #
        # A section heading owns its prose, its exhibits and its tables
        # alike. They all carry the same heading path, so they all merge.
        # Measured on page 1 of a research report with it ON:
        #
        #     ONE chunk, 1007 tokens, labelled "table", containing
        #       the section heading
        #       four paragraphs of prose
        #       the Exhibit 1 caption and figure description
        #       a contact table of eight names, emails and phone numbers
        #       the legal disclaimer
        #
        # It also ran BEFORE pass C could ever see the figure, so Exhibit 1's
        # description was indexed twice — once inside the merged chunk, once
        # as its own record.
        #
        # WHAT WE DO INSTEAD
        #
        # Pass B below performs the same merge, restricted to prose. Same
        # predicate — consecutive, equal heading path, under a budget — with
        # a type check the chunker does not offer. There is no setting that
        # merges prose with prose but not with tables, which is why this is
        # our code and not a flag.
        #
        # NOTE: there is no min_tokens parameter on HybridChunker. Some doc
        # mirrors list one. It does not exist in the source, and pydantic
        # ignores unknown keyword arguments silently, so passing it looks
        # like it worked and does nothing.
        # ─────────────────────────────────────────────────────────────────
        merge_peers=False,
    )

    # ═════════════════════════════════════════════════════════════════════
    # STEP 2 — correct the headings, THEN chunk
    #
    # HybridChunker never merges across a heading boundary, and the heading
    # path is the whole merge predicate. So every heading the layout model
    # got wrong is both a boundary that should not be there and a wrong
    # string prepended into every vector beneath it.
    #
    # There is no chunker option for this. The document is corrected first.
    # ═════════════════════════════════════════════════════════════════════
    clean_headings(doc)

    chunks = list(chunker.chunk(dl_doc=doc))
    tables = table_markdown(doc)
    ingested_at = int(datetime.now(timezone.utc).timestamp())

    # ═════════════════════════════════════════════════════════════════════
    # STEP 3 — index the original elements by reference
    #
    # NOT for grouping table fragments — `table_groups` does that. This
    # exists for one case: the extracted markdown is wrong, and we need the
    # original element back so we can render it and look at it.
    #
    #     table_ref  ->  items_by_ref[ref]  ->  the original TableItem
    # ═════════════════════════════════════════════════════════════════════
    items_by_ref = {}
    for item, _ in doc.iterate_items():
        ref = getattr(item, "self_ref", None)
        if ref:
            items_by_ref[ref] = item

    # ═════════════════════════════════════════════════════════════════════
    # STEP 4 — record identity
    #
    # The same text can legitimately appear several times: a disclaimer on
    # every page. Identical text hashes identically, so without a counter all
    # copies would collapse into one record and the other page numbers would
    # be lost.
    #
    # Note what this counter does NOT solve: repeated text that is worthless
    # stays repeated, once per occurrence. That is the drop filter's job, in
    # pass A, and it runs first.
    # ═════════════════════════════════════════════════════════════════════
    occurrences: defaultdict = defaultdict(int)

    def make(text: str, meta_extra: dict, position: int) -> dict:
        """Build one record, assigning a content-addressed id.

            chunk_id = {doc_id}:{sha256(text)[:16]}:{occurrence}

        Each part earns its place:

          doc_id      scopes the hash. Two PDFs sharing a boilerplate
                      disclaimer would otherwise produce the same id, and
                      upsert is last-write-wins — one document silently
                      deletes the other's chunk, with no error anywhere.

          hash        makes the id derivable from the text. A positional id
                      changes for every chunk after an edit, forcing a full
                      re-embed when one paragraph changed. A hash changes
                      only where the text changed, which is what makes the
                      incremental sync possible.

          occurrence  distinguishes text that legitimately repeats within one
                      document, so each copy stays separately retrievable
                      with its own page number.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        occurrence = occurrences[digest]
        occurrences[digest] += 1
        return {"text": text, "meta": {
            "chunk_id": f"{doc_id}:{digest}:{occurrence}",
            "content_hash": digest, "occurrence": occurrence,
            "doc_id": doc_id, "source": pdf.name,
            "doc_date": doc_date, "ingested_at": ingested_at,
            "position": position,
            # Recording the embedding model lets the retrieval side refuse to
            # query an index built with a different one. That mismatch
            # produces plausible rankings and no error, which makes it the
            # hardest failure to notice.
            "embed_model": EMBED_MODEL, "access": ACCESS_GROUPS,
            "n_tokens": len(ENCODING.encode(text)),
            **meta_extra,
        }}

    # ═════════════════════════════════════════════════════════════════════
    # STEP 5 — pass A: chunks become entries
    #
    # An entry is a plain dict, not a Docling chunk. The passes that follow
    # merge and reorder, and constructing valid DocChunk objects for merged
    # content means rebuilding DocMeta by hand for no benefit.
    #
    #     kind    "text" | "table" | "figure_slot" | "formula" | "code"
    #     body    the serialized content, WITHOUT the heading path
    #     head    the heading path, prepended at record time
    #
    # Three things happen here:
    #
    #   figure_slot   a chunk that is nothing but pictures becomes a
    #                 placeholder, holding the refs. Pass C expands it to one
    #                 record per picture. The placeholder keeps its position,
    #                 so reading order survives.
    #
    #   drop          page furniture is removed. Text chunks only — a table
    #                 or a figure is never dropped, whatever it looks like.
    #
    #   contextualize is called HERE, while the original chunk still exists,
    #                 and only for entries that never merge. See pass C.
    # ═════════════════════════════════════════════════════════════════════
    entries: list[dict] = []
    dropped: list[tuple[int, str, str]] = []

    for chunk in chunks:
        items = chunk.meta.doc_items
        headings = list(chunk.meta.headings or [])
        pages = sorted({prov.page_no
                        for item in items
                        for prov in (getattr(item, "prov", []) or [])})
        page = pages[0] if pages else -1
        page_end = pages[-1] if pages else -1

        # A chunk whose only real content is pictures, with nothing beside
        # them but their captions.
        #
        # WHY THE CAPTION MUST BE ALLOWED IN HERE
        #
        # An earlier version tested `all items are pictures`. That excluded
        # exactly the well-formed case: when the layout model DOES link a
        # caption to its figure, the serializer emits both as one chunk, so
        # the strict test rejected it and the figure fell through to the
        # generic path below.
        #
        # There the record text comes from the serializer, which renders the
        # caption and the classification but not a description attached after
        # the parse. Measured: 10 of 16 figures came out as
        # "Exhibit 4: ... \n\n Bar chart" — 30 tokens, no description, while
        # the 6 UNLINKED figures got full ones. The better the parse, the
        # worse the record.
        #
        # A table still disqualifies the chunk. Tables carry values that must
        # be indexed as themselves.
        labels = [str(getattr(i, "label", "")).lower() for i in items]
        pictures = [i for i, label in zip(items, labels) if "picture" in label]
        beside = [label for label in labels if "picture" not in label]

        if pictures and all("caption" in label for label in beside):
            entries.append({
                "kind": "figure_slot",
                "refs": [r for i in pictures
                         if (r := getattr(i, "self_ref", None))],
                "head": headings, "page": page, "page_end": page_end,
            })
            continue

        kind = content_type(chunk)

        if kind == "text":
            reason = is_furniture(chunk.text)
            if reason:
                dropped.append((page, reason, chunk.text.strip()[:60]))
                continue

        # Attach the stored PNG if this chunk contains a figure we saved.
        image_uri = ""
        for item in items:
            key = getattr(item, "self_ref", None)
            if key and key in (figure_uris or {}):
                image_uri = figure_uris[key]
                break

        entries.append({
            "kind": kind,
            "body": chunk.text,
            # The exact string contextualize() would produce, kept for
            # entries that survive pass B unmerged, so their embedded text is
            # byte-identical to what the chunker would have given us.
            "contextualized": chunker.contextualize(chunk=chunk),
            "merged": False,
            "head": headings, "page": page, "page_end": page_end,
            "ref": table_ref_of(chunk),
            "image_uri": image_uri,
        })

    # ═════════════════════════════════════════════════════════════════════
    # STEP 6 — pass B: merge adjacent prose
    #
    #     entry  entry  entry            same heading path, all text
    #       \      |      /
    #        +-----+-----+
    #              |
    #              v
    #         one entry                  up to PROSE_TARGET_TOKENS
    #
    # The same predicate the chunker uses, plus the type check it does not
    # offer: BOTH sides must be "text". A table, a figure slot, a formula or
    # a code block ends the run — it is never absorbed and never absorbs.
    #
    # WHY A TARGET WELL BELOW THE BUDGET
    #
    # CHUNK_TOKENS is a ceiling, not a goal. Merging to the ceiling produces
    # chunks that retrieve a page when the answer is a paragraph, and the
    # reranker then has to find the paragraph again inside it.
    #
    # MERGING ACROSS AN EXHIBIT
    #
    # Adjacent-only merging almost never fires on a document whose prose is
    # interleaved with exhibits:
    #
    #     paragraph   figure   caption   paragraph   figure   paragraph
    #
    # Every run is length one. Measured on a 7-page research report, the
    # adjacent rule left the prose median at 61 tokens; allowing a merge to
    # reach back past intervening exhibits, while the heading path still
    # matches, took it to 171.
    #
    # WHAT THAT COSTS, PLAINLY
    #
    # A merged record's page range then spans exhibits it does not contain,
    # and prev_id / next_id no longer step through strict document order for
    # those records. Retrieval uses those links only for neighbour expansion,
    # so the damage is small — but it is real, and MERGE_ACROSS_EXHIBITS
    # turns it off.
    #
    # The search stops at the first entry with a DIFFERENT heading path, not
    # at the first non-text entry. A heading boundary still ends everything.
    #
    # WHAT THIS DOES NOT DO
    #
    # It does not merge across a heading boundary. Two sections stay two
    # things. On a document with many short sections that leaves prose chunks
    # smaller than the target, and that is the correct outcome — the
    # alternative is joining text that does not belong together.
    #
    # It does not reorder anything. A merged entry keeps the position of its
    # FIRST fragment, so the exhibits it reached over stay where they are.
    # ═════════════════════════════════════════════════════════════════════
    merged_entries: list[dict] = []
    merges = 0

    for entry in entries:
        target = None

        if entry["kind"] == "text":
            # Walk back through entries already kept, stopping at the first
            # one under a different heading.
            for candidate in reversed(merged_entries):
                if candidate["head"] != entry["head"]:
                    break
                if candidate["kind"] == "text":
                    target = candidate
                    break
                if not MERGE_ACROSS_EXHIBITS:
                    break

        if target is not None and len(ENCODING.encode(
                target["body"] + "\n" + entry["body"])) <= PROSE_TARGET_TOKENS:
            target["body"] += "\n" + entry["body"]
            target["page_end"] = max(target["page_end"], entry["page_end"])
            target["merged"] = True
            target["image_uri"] = target["image_uri"] or entry["image_uri"]
            merges += 1
        else:
            merged_entries.append(entry)

    entries = merged_entries

    # ═════════════════════════════════════════════════════════════════════
    # STEP 6b — the floor
    #
    #     entry (17t)  entry (14t)  entry (8t)        different headings
    #         \            |           /
    #          +-----------+----------+
    #                      |
    #                      v
    #                 one entry                       up to MIN_CHUNK_TOKENS
    #
    # The ONLY rule in this file that crosses a heading boundary, and it runs
    # only when MIN_CHUNK_TOKENS is set above 0.
    #
    # It absorbs forward while the accumulated entry is still under the floor,
    # and stops the moment it clears it — so a record that was already big
    # enough is never touched, and a merged one lands just past the floor
    # rather than at the target.
    #
    # WHAT THIS DOES NOT DO
    #
    #     It does not absorb a table, a figure or a code block. Those end the
    #     run, as they do in pass B.
    #
    #     It does not discard the headings it crosses. Each one is written
    #     into the body, so the context of every part is embedded with it.
    #     Only `headings` metadata keeps the first path, so section_id stays
    #     stable and filtering still works.
    # ═════════════════════════════════════════════════════════════════════
    floor_merges = 0
    if MIN_CHUNK_TOKENS > 0:
        floored: list[dict] = []
        for entry in entries:
            previous = floored[-1] if floored else None

            if (previous is not None
                    and previous["kind"] == "text" and entry["kind"] == "text"
                    and len(ENCODING.encode(previous["body"])) < MIN_CHUNK_TOKENS):
                # The heading being crossed goes into the text, not lost.
                crossed = ([h for h in entry["head"] if h not in previous["head"]]
                           if entry["head"] != previous["head"] else [])
                previous["body"] += "\n" + "\n".join([*crossed, entry["body"]])
                previous["page_end"] = max(previous["page_end"], entry["page_end"])
                previous["merged"] = True
                previous["image_uri"] = previous["image_uri"] or entry["image_uri"]
                floor_merges += 1
            else:
                floored.append(entry)
        entries = floored

    # ═════════════════════════════════════════════════════════════════════
    # STEP 7 — pass C: entries become records
    #
    # Two structures are built here and they are not the same thing:
    #
    #   records       every record in reading order
    #   table_groups  those records that belong to a table, grouped by table
    #
    # A large table appears once in `records` per fragment, and once in
    # `table_groups` as a list of all its fragments.
    # ═════════════════════════════════════════════════════════════════════
    records: list[dict] = []
    table_groups: defaultdict = defaultdict(list)
    figures_indexed = figures_skipped = 0
    figure_refs_seen: set = set()

    def section_id(headings: list[str]) -> str:
        """Groups every record under the same top-level heading.

        Nothing reads it yet. It is written now because adding it later means
        re-embedding the corpus, and it is what parent-section retrieval
        would need.
        """
        return hashlib.sha256(
            (headings[0] if headings else "").encode()).hexdigest()[:12]

    for position, entry in enumerate(entries):

        # ── a figure slot expands to one record per picture ──────────────
        if entry["kind"] == "figure_slot":
            for ref in entry["refs"]:
                figure_refs_seen.add(ref)
                item = items_by_ref.get(ref)
                if item is None:
                    continue

                description = picture_description(item)
                if not description:
                    # Nothing to embed. A record holding only "a figure was
                    # here" matches every query about figures and answers
                    # none of them. The extraction report already flags it.
                    figures_skipped += 1
                    continue

                # THE CAPTION.
                #
                # Docling links a picture to its caption element and resolves
                # it through caption_text(doc). Without this call the caption
                # becomes a separate 25-token text chunk saying
                # "Exhibit 18: Our Conceptual Roadmap for AI Developments"
                # and nothing else, while the figure record it names has no
                # exhibit number in it at all — so neither is retrievable by
                # the one string a reader would actually search for.
                #
                # The vision model sometimes reads the exhibit title off the
                # image, which is why some figure records already contain it.
                # That is luck, not linkage.
                caption = ""
                try:
                    caption = (item.caption_text(doc) or "").strip()
                except Exception:
                    caption = ""

                # Same shape as every other record: context first, then
                # content. Built by hand because contextualize() takes a
                # chunk and this record is not one.
                parts = [*entry["head"]]
                if caption:
                    parts.append(caption)
                parts.append(description)
                text = "\n".join(parts)

                # Chart series, when the extraction produced any. In the
                # embedded text rather than metadata only, so a question
                # about a specific value has something to match.
                series = chart_data(item)
                if series is not None:
                    text += f"\n\nchart data: {str(series)[:800]}"

                records.append(make(text, {
                    "page": entry["page"], "page_end": entry["page_end"],
                    "headings": entry["head"],
                    "section_id": section_id(entry["head"]),
                    "content_type": "figure",
                    "table_id": "",
                    "image_uri": (figure_uris or {}).get(ref, ""),
                    "has_caption": bool(caption),
                    "has_chart_data": series is not None,
                }, position))
                figures_indexed += 1
            continue

        # ── everything else ──────────────────────────────────────────────
        # An unmerged entry uses the chunker's own contextualized string, so
        # its text is byte-identical to what it would have been before pass B
        # existed. A merged one is rebuilt: contextualize() prepends the
        # heading path, and joining two contextualized strings would repeat
        # it in the middle of the chunk.
        text = ("\n".join([*entry["head"], entry["body"]]) if entry["merged"]
                else entry["contextualized"])

        ref = entry["ref"]
        record = make(text, {
            "page": entry["page"], "page_end": entry["page_end"],
            # A list, not a joined string: Pinecone can filter a list with
            # $in and cannot filter a comma-joined string at all.
            "headings": entry["head"],
            "section_id": section_id(entry["head"]),
            "content_type": entry["kind"],
            # Links every fragment of a table to its summary.
            "table_id": hashlib.sha256(ref.encode()).hexdigest()[:12] if ref else "",
            "image_uri": entry["image_uri"],
        }, position)

        records.append(record)
        if ref:
            table_groups[ref].append(record)

    # Pictures that never reached a figure slot, because they were inside a
    # chunk holding something else. Their description is already in that
    # chunk's text, so indexing them again would put the same content in two
    # vectors — but it also means they are not separately retrievable, which
    # is worth knowing.
    figures_merged = sum(
        1 for item, _ in doc.iterate_items()
        if isinstance(item, PictureItem)
        and getattr(item, "self_ref", None) not in figure_refs_seen)

    # ═════════════════════════════════════════════════════════════════════
    # STEP 8 — pass D: one summary per table
    #
    # Iterates table_groups, not records. However many fragments the chunker
    # produced from one table, that table gets ONE summary — read from the
    # complete grid on the document object, not stitched back together from
    # the fragments.
    #
    # The summary is an ADDITIONAL record. The fragments stay exactly as they
    # are.
    # ═════════════════════════════════════════════════════════════════════
    summarised, skipped, repaired = 0, [], []

    for ref, fragments in table_groups.items():
        first = fragments[0]["meta"]
        markdown = tables.get(ref, "")

        if not markdown:
            # export_to_markdown() failed. The rows are still indexed as
            # chunks, but without a summary they are close to unreachable.
            skipped.append((first["page"], "could not be serialised"))
            continue
        if not needs_summary(markdown):
            skipped.append((first["page"],
                            f"{len(table_cells(markdown))} cells, treated as layout"))
            continue

        # If the grid itself is wrong, summarising it produces a confident
        # description of a table that does not exist. Look at the rendered
        # table instead, which is what a person would do.
        structure_problems = table_looks_broken(markdown)
        summary, source = None, "markdown"
        if structure_problems:
            item = items_by_ref.get(ref)
            if item is None:
                print(f"    table on p{first['page']} is broken and its element "
                      "could not be found to render", flush=True)
            else:
                try:
                    summary = summarize_table_image(item, doc, first["headings"])
                    if summary is None:
                        # get_image() returned nothing. Almost always a parse
                        # made before generate_table_images was enabled — the
                        # crop was never rendered, so there is nothing to
                        # look at.
                        print(f"    table on p{first['page']} is broken but has no "
                              "rendered image; the parse predates "
                              "generate_table_images. Delete the cache and re-parse.",
                              flush=True)
                    else:
                        source = "image"
                except Exception as exc:
                    print(f"    table image fallback failed on p{first['page']}: "
                          f"{exc}", flush=True)
            repaired.append((first["page"], structure_problems, summary is not None))

        if summary is None:
            summary = summarize_table(markdown, first["headings"])
            source = "markdown"

        records.append(make(summary, {
            "page": first["page"], "page_end": fragments[-1]["meta"]["page_end"],
            "headings": first["headings"], "section_id": first["section_id"],
            "content_type": "table_summary",
            # Same table_id as the fragments: this is how retrieval walks
            # from a matched summary to the rows carrying the exact values.
            "table_id": first["table_id"],
            "table_chars": len(markdown),
            "n_fragments": len(fragments),
            "summary_source": source,
        }, first["position"]))
        summarised += 1

    # ═════════════════════════════════════════════════════════════════════
    # STEP 9 — pass E: reading order
    #
    # Summaries were appended at the end but carry the position of their
    # table's first fragment. Sorting on (position, is-not-a-summary) puts
    # each one immediately before the rows it describes:
    #
    #     section chunk
    #     table summary        <- moved here
    #     table fragment 1
    #     table fragment 2
    #     paragraph chunk
    # ═════════════════════════════════════════════════════════════════════
    records.sort(key=lambda r: (r["meta"]["position"],
                                r["meta"]["content_type"] != "table_summary"))

    for i, record in enumerate(records):
        record["meta"]["position"] = i
        record["meta"]["n_positions"] = len(records)

    # Reading-order links. Content-addressed ids cannot be derived from
    # position — you cannot compute record 43's id from record 42's — so the
    # edges are stored explicitly. Without them, expanding a result to its
    # neighbours at query time would require scanning every chunk in the
    # document on every query.
    for i, record in enumerate(records):
        if i:
            record["meta"]["prev_id"] = records[i - 1]["meta"]["chunk_id"]
        if i + 1 < len(records):
            record["meta"]["next_id"] = records[i + 1]["meta"]["chunk_id"]

    # ═════════════════════════════════════════════════════════════════════
    # STEP 10 — metadata budget
    #
    # Pinecone caps metadata at 40 KB per vector. Rather than guessing a
    # character limit, measure what the structural fields actually cost and
    # give the text whatever is left.
    #
    #   record["text"]          the canonical text, embedded
    #   record["meta"]["text"]  a bounded copy, so retrieval and reranking
    #                           can read it straight off the query result
    #
    # At much larger scale this pattern flips to storing a pointer.
    # ═════════════════════════════════════════════════════════════════════
    if not records:
        print("  no records — every chunk was dropped or the document is empty",
              flush=True)
        return records

    overhead = len(json.dumps({**records[0]["meta"], "text": ""}).encode())
    budget = max(512, PINECONE_METADATA_BYTES - overhead - 1024)
    for record in records:
        record["meta"]["text"] = record["text"][:budget]

    # ═════════════════════════════════════════════════════════════════════
    # STEP 11 — oversized chunks
    #
    # The chunker respects the budget but cannot split an atomic element
    # smaller than itself: one enormous table row, one very long code block.
    # Pass B cannot create one — it checks the target before merging.
    #
    # Truncating loses that one item; raising would abort a 250-page document
    # over a single row. The flag makes the loss visible instead of silent.
    # Ideally this list is empty.
    # ═════════════════════════════════════════════════════════════════════
    over = [r for r in records if r["meta"]["n_tokens"] > CHUNK_TOKENS]
    for record in over:
        record["text"] = ENCODING.decode(
            ENCODING.encode(record["text"])[:CHUNK_TOKENS])
        record["meta"]["n_tokens"] = CHUNK_TOKENS
        record["meta"]["truncated"] = True

    # ═════════════════════════════════════════════════════════════════════
    # STEP 12 — diagnostics
    #
    # Not "document processed successfully". How many records, how many
    # tables got summaries, what was dropped, what was skipped, what was
    # repaired, what was truncated.
    #
    # Extraction and chunking fail quietly. This is where that becomes
    # visible, before the data reaches retrieval and someone wonders why the
    # answers are wrong.
    # ═════════════════════════════════════════════════════════════════════
    sizes = sorted(r["meta"]["n_tokens"] for r in records)
    median = sizes[len(sizes) // 2]
    types = Counter(r["meta"]["content_type"] for r in records)

    print(f"  {len(records)} records from {len(chunks)} chunks "
          f"({summarised} of {len(tables)} tables summarised)", flush=True)
    print(f"  types: {dict(types)}", flush=True)
    print(f"  size: median {median} tokens, "
          f"{sum(1 for s in sizes if s < 50)} under 50, max {sizes[-1]}",
          flush=True)

    if dropped:
        reasons = Counter(reason for _, reason, _ in dropped)
        print(f"  dropped {len(dropped)} chunk(s) as page furniture: "
              f"{dict(reasons)}", flush=True)
        for page, reason, sample in dropped:
            print(f"    p{page} {reason}: {sample!r}", flush=True)
    if merges:
        print(f"  merged {merges} adjacent prose chunk(s) under a shared heading",
              flush=True)
    if floor_merges:
        print(f"  merged {floor_merges} chunk(s) ACROSS a heading boundary to "
              f"reach the {MIN_CHUNK_TOKENS}-token floor. Records now answer "
              "about more than one section each.", flush=True)

    captioned = sum(1 for r in records
                    if r["meta"].get("content_type") == "figure"
                    and r["meta"].get("has_caption"))
    print(f"  figures: {figures_indexed} indexed one-per-record, "
          f"{captioned} with a linked caption"
          + (f", {figures_skipped} with no description" if figures_skipped else "")
          + (f", {figures_merged} inside a mixed chunk" if figures_merged else ""),
          flush=True)
    if figures_indexed and captioned < figures_indexed:
        print(f"  NOTE: {figures_indexed - captioned} figure(s) have no caption "
              "linked in the parse. They are retrievable by description but not "
              "by exhibit number unless the vision model happened to read it.",
              flush=True)
    if figures_merged:
        print(f"  NOTE: {figures_merged} figure(s) share a chunk with other "
              "content, so they are not separately retrievable.", flush=True)

    for page, reason in skipped:
        print(f"    skipped table on p{page}: {reason}", flush=True)
    for page, structure_problems, used_image in repaired:
        route = ("described from the rendered image" if used_image
                 else "FELL BACK TO BROKEN MARKDOWN — the summary may be wrong")
        print(f"    table on p{page} has bad structure "
              f"({'; '.join(structure_problems)}): {route}", flush=True)
    if over:
        print(f"  WARNING: truncated {len(over)} unsplittable chunks to "
              f"{CHUNK_TOKENS} tokens", flush=True)

    # A table found by extraction but never reaching a record is invisible to
    # the loop above, so compare the two counts directly. Zero groups against
    # a non-zero table count means the chunk-to-table link is broken, not
    # that the document lacks tables — a distinction that is otherwise
    # indistinguishable in the output.
    orphaned = len(tables) - len(table_groups)
    if tables and not table_groups:
        print(f"  ERROR: {len(tables)} tables were extracted but none could be "
              "linked to a record. No table summaries were generated and "
              "table_id is empty on every record. Check table_ref_of() against "
              "this docling version.", flush=True)
    elif orphaned:
        print(f"  WARNING: {orphaned} of {len(tables)} tables produced no record",
              flush=True)

    return records
