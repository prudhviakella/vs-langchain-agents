"""Turning a parsed document into the records that will be embedded.

Three things happen here, each replacing machinery that pipelines commonly
hand-write: structure-aware chunking, heading context folded into the embedded
string, and one summary chunk per table.

Chunk identity is content-addressed. That single decision is what makes the
incremental sync in `sync.py` possible, and what makes a retried run safe.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import (ACCESS_GROUPS, CHUNK_TOKENS, EMBED_MODEL, ENCODING,
                     PINECONE_METADATA_BYTES)
from .tables import (needs_summary, summarize_table, summarize_table_image,
                     table_cells, table_looks_broken, table_markdown, table_ref_of)

def content_type(chunk) -> str:
    """Coarse type: text, table, table_summary, figure, formula or code.

    This is what makes filtering possible at query time — "only tables", "only
    figures". It is also how you would tell apart a pipeline that handles prose well
    and tables badly from one that is mediocre at both.
    """
    labels = " ".join(str(getattr(item, "label", "")).lower()
                      for item in chunk.meta.doc_items)
    # Ordered: a chunk containing both a table and its caption counts as a table.
    for needle, label in (("table", "table"), ("picture", "figure"),
                          ("figure", "figure"), ("formula", "formula"),
                          ("equation", "formula"), ("code", "code")):
        if needle in labels:
            return label
    return "text"


def document_date(pdf: Path, head: str) -> str:
    """Publication date of the document, for filtering stale content at query time.

    Distinct from ingestion time. A corpus accumulates versions of the same report,
    and "semantically relevant but from two years ago" is a failure mode users hit
    constantly. Filtering on doc_date is the fix.

    The heuristic — first year, plus a quarter if present, in the opening pages — is
    deliberately simple and will be wrong on documents whose first four-digit number
    is a reference or a street address. Replace it with a real header parse for any
    corpus where dates carry weight.
    """
    year = re.search(r"\b(19|20)\d{2}\b", head)
    if year:
        quarter = re.search(r"\bQ([1-4])\s*(19|20)\d{2}\b", head)
        return f"{year.group(0)}-Q{quarter.group(1)}" if quarter else year.group(0)
    return datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m")


def build_records(doc, pdf: Path, doc_id: str, doc_date: str,
                  figure_uris: dict[str, str] | None = None) -> list[dict]:
    """Turn a parsed document into the records that will be embedded and indexed.

    Three things happen here, each replacing machinery that pipelines commonly
    hand-write:

    1. HybridChunker splits on the document's own hierarchy, then refines against a
       token budget — oversized elements divided, undersized siblings under the same
       heading merged. Boundaries come from structure, not from measured similarity
       between sentences.

    2. contextualize() prepends the heading path to each chunk, and that string is
       what gets embedded. Section context therefore participates in retrieval
       instead of sitting unused in metadata.

    3. Tables get one extra chunk each: a summary carrying the vocabulary the raw
       grid lacks.
    """
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer, ChunkingSerializerProvider,
    )
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

    class MarkdownTableProvider(ChunkingSerializerProvider):
        """Serialize tables as markdown rather than Docling's default triplet form.

        The default renders a cell as `**Column**, row 1 = value`, which flattens
        the grid and embeds poorly: the relationship between a value and its column
        header is exactly what a tabular query depends on.
        """

        # The parameter must be named `doc`: HybridChunker calls this by keyword, as
        # get_serializer(doc=dl_doc). **kwargs absorbs anything a future release
        # adds rather than raising on an unexpected argument.
        def get_serializer(self, doc, **kwargs):
            """Return a serializer that renders tables as markdown."""
            return ChunkingDocSerializer(
                doc=doc, table_serializer=MarkdownTableSerializer())

    chunker = HybridChunker(
        # The tokenizer is the embedding model's own. Sizing in characters is a
        # proxy that fails silently: the API accepts an oversized input, embeds the
        # first N tokens, and returns a valid-looking vector for half a chunk.
        tokenizer=OpenAITokenizer(tokenizer=ENCODING, max_tokens=CHUNK_TOKENS),
        serializer_provider=MarkdownTableProvider(),
        merge_peers=True,
    )

    chunks = list(chunker.chunk(dl_doc=doc))
    tables = table_markdown(doc)
    ingested_at = int(datetime.now(timezone.utc).timestamp())

    # Every element by ref, so a broken table can be re-read as an image.
    items_by_ref = {}
    for item, _ in doc.iterate_items():
        ref = getattr(item, "self_ref", None)
        if ref:
            items_by_ref[ref] = item

    # How many times each content hash has been seen in this document, so genuinely
    # repeated text gets distinct ids. See the chunk_id note below.
    occurrences: defaultdict = defaultdict(int)

    def make(text: str, meta_extra: dict, position: int) -> dict:
        """Build one record, assigning a content-addressed id.

            chunk_id = {doc_id}:{sha256(text)[:16]}:{occurrence}

        Each part earns its place:

          doc_id      scopes the hash. Two PDFs sharing a boilerplate disclaimer
                      would otherwise produce the same id, and upsert is
                      last-write-wins — one document silently deletes the other's
                      chunk, with no error anywhere.

          hash        makes the id derivable from the text. A positional id (:00042)
                      changes for every chunk after an edit, forcing a full re-embed
                      when one paragraph changed. A hash changes only where the text
                      changed, which is what makes the incremental sync possible.

          occurrence  distinguishes text that legitimately repeats within one
                      document — a disclaimer printed on every page — so each copy
                      stays separately retrievable with its own page number.
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
            # Recording the embedding model lets the retrieval side refuse to query
            # an index built with a different one. That mismatch produces plausible
            # rankings and no error, which makes it the hardest failure to notice.
            "embed_model": EMBED_MODEL, "access": ACCESS_GROUPS,
            "n_tokens": len(ENCODING.encode(text)),
            **meta_extra,
        }}

    records: list[dict] = []
    table_groups: defaultdict = defaultdict(list)

    # ── Pass 1: one record per chunk ─────────────────────────────────────────
    for position, chunk in enumerate(chunks):
        # THIS is the string that gets embedded — heading path included.
        text = chunker.contextualize(chunk=chunk)

        headings = list(chunk.meta.headings or [])

        # A chunk can span pages, so record both ends for citation.
        pages = sorted({prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in (getattr(item, "prov", []) or [])})

        ref = table_ref_of(chunk)

        # Attach the stored PNG if this chunk contains a figure we saved.
        image_uri = ""
        for item in chunk.meta.doc_items:
            key = getattr(item, "self_ref", None)
            if key and key in (figure_uris or {}):
                image_uri = figure_uris[key]
                break

        record = make(text, {
            "page": pages[0] if pages else -1,
            "page_end": pages[-1] if pages else -1,
            # A list, not a joined string: Pinecone can filter a list with $in and
            # cannot filter a comma-joined string at all.
            "headings": headings,
            # Groups every chunk under the same top-level heading. Nothing uses it
            # yet; it is written now because adding it later means re-embedding the
            # corpus, and it is what parent-section retrieval would need.
            "section_id": hashlib.sha256(
                (headings[0] if headings else "").encode()).hexdigest()[:12],
            "content_type": content_type(chunk),
            # Links every fragment of a table to its summary.
            "table_id": hashlib.sha256(ref.encode()).hexdigest()[:12] if ref else "",
            "image_uri": image_uri,
        }, position)

        records.append(record)
        if ref:
            table_groups[ref].append(record)

    # ── Pass 2: one summary per table ────────────────────────────────────────
    # Iterates table_groups, not chunks: one summary per table, however many
    # fragments the chunker produced from it.
    summarised, skipped, repaired = 0, [], []

    for ref, fragments in table_groups.items():
        first = fragments[0]["meta"]
        markdown = tables.get(ref, "")

        if not markdown:
            # export_to_markdown() failed. The rows are still indexed as chunks, but
            # without a summary they are close to unreachable.
            skipped.append((first["page"], "could not be serialised"))
            continue
        if not needs_summary(markdown):
            skipped.append((first["page"],
                            f"{len(table_cells(markdown))} cells, treated as layout"))
            continue

        # If the grid itself is wrong, summarising it produces a confident
        # description of a table that does not exist. Look at the rendered table
        # instead, which is what a person would do.
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
                        # get_image() returned nothing. Almost always a parse made
                        # before generate_table_images was enabled — the crop was
                        # never rendered, so there is nothing to look at.
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
            # The summary spans the whole table's page range.
            "page": first["page"], "page_end": fragments[-1]["meta"]["page_end"],
            "headings": first["headings"], "section_id": first["section_id"],
            "content_type": "table_summary",
            # Same table_id as the fragments: this is how retrieval walks from a
            # matched summary to the rows carrying the exact values.
            "table_id": first["table_id"],
            "table_chars": len(markdown),
            "n_fragments": len(fragments),
            "summary_source": source,
        }, first["position"]))
        summarised += 1

    # ── Pass 3: restore reading order, then assign positions and links ───────
    # Summaries were appended at the end of the list but carry the position of their
    # table's first fragment. Sorting on (position, is-not-a-summary) puts each
    # summary immediately before the rows it describes.
    records.sort(key=lambda r: (r["meta"]["position"],
                                r["meta"]["content_type"] != "table_summary"))

    for i, record in enumerate(records):
        record["meta"]["position"] = i
        record["meta"]["n_positions"] = len(records)

    # Reading-order links. Content-addressed ids cannot be derived from position —
    # you cannot compute chunk 43's id from chunk 42's — so the edges are stored
    # explicitly. Without them, expanding a result to its neighbours at query time
    # would require scanning every chunk in the document on every query.
    for i, record in enumerate(records):
        if i:
            record["meta"]["prev_id"] = records[i - 1]["meta"]["chunk_id"]
        if i + 1 < len(records):
            record["meta"]["next_id"] = records[i + 1]["meta"]["chunk_id"]

    # ── Metadata budget ──────────────────────────────────────────────────────
    # Pinecone allows 40 KB of metadata per vector. Rather than guessing a character
    # cap, measure what the structural fields cost and give the body the remainder,
    # with a safety margin. The text is stored because the reranker and the LLM both
    # read it; at much larger scale the pattern flips to storing a pointer.
    overhead = len(json.dumps({**records[0]["meta"], "text": ""}).encode())
    budget = max(512, PINECONE_METADATA_BYTES - overhead - 1024)
    for record in records:
        record["meta"]["text"] = record["text"][:budget]

    # ── Oversized chunks ─────────────────────────────────────────────────────
    # HybridChunker respects the token budget, but it cannot split an atomic element
    # smaller than itself: one enormous table row, one very long code block. Raising
    # here would abort a 250-page document over a single row. Truncating loses that
    # one item and keeps everything else, and the flag makes the loss visible rather
    # than silent.
    over = [r for r in records if r["meta"]["n_tokens"] > CHUNK_TOKENS]
    for record in over:
        record["text"] = ENCODING.decode(
            ENCODING.encode(record["text"])[:CHUNK_TOKENS])
        record["meta"]["n_tokens"] = CHUNK_TOKENS
        record["meta"]["truncated"] = True

    # ── Report ───────────────────────────────────────────────────────────────
    types = Counter(r["meta"]["content_type"] for r in records)
    print(f"  {len(records)} chunks ({summarised} of {len(tables)} tables "
          f"summarised)", flush=True)
    print(f"  types: {dict(types)}", flush=True)
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
    # A table detected in the document but never reaching a chunk is invisible to
    # the loop above, so compare the two counts directly.
    # A table found by extraction but never reaching a chunk is invisible to the
    # loop above, so compare the two counts directly. Zero groups against a non-zero
    # table count means the chunk-to-table link is broken, not that the document
    # lacks tables — a distinction that is otherwise indistinguishable in the output.
    orphaned = len(tables) - len(table_groups)
    if tables and not table_groups:
        print(f"  ERROR: {len(tables)} tables were extracted but none could be linked "
              "to a chunk. No table summaries were generated and table_id is empty "
              "on every record. Check table_ref_of() against this docling version.",
              flush=True)
    elif orphaned:
        print(f"  WARNING: {orphaned} of {len(tables)} tables produced no chunk",
              flush=True)

    return records
