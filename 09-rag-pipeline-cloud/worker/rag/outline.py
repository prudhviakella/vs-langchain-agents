"""The document's own declared section hierarchy, read from the PDF outline.

    PDF outline                 chunk
    (level, title, page, y)     (page, y)
          |                        |
          +--------- match --------+
                     |
                     v
            full heading path
                     |
          +----------+----------+
          |                     |
    chunk.meta.headings    Neo4j Section nodes
    (what gets embedded)   (what the graph traverses)

ONE MAPPING, TWO CONSUMERS

The chunk's metadata and the graph MUST agree about which section a chunk
belongs to. If the chunk says `I. Disease Parameters` and the graph says
`8 Appendices > 8.5 RECIST Guidelines`, the same chunk has two different
parents depending on which system you ask, and an agent walking the graph and
then reading chunk metadata gets a contradiction. So both read from the mapping
this module computes, once.

WHY THE OUTLINE AND NOT INFERENCE

Docling infers heading depth from bookmarks, numbering markers and visual
style, then compresses the result relative to what is present in each scope.
That is the right approach for a document with no declared structure. It is the
wrong approach for one that has it.

Measured on a 113-page clinical protocol embedding a RECIST guideline that
numbers its own subsections I./II./III. inside a document numbering its
sections in Arabic: seven Docling configurations were tried — scheme order,
bookmark threshold, style off, bookmarks off, max_level, a different PDF
backend — and none produced a correct tree. Every heuristic fix surfaced a new
intruder: fixing `I.` promoted `21.` and `26.` from an amendment change-log to
top-level sections.

The same document declares the correct tree in its own outline: 125 entries,
five levels, `8 Appendices` properly containing `8.1` through `8.8`, and none
of the intruders present at all. The outline already knows `21.` is not a
section.

So this module stops inferring. A chunk on page 101 belongs to
`8.5 RECIST 1.1 and irRECIST Guidelines` because page 101 falls inside 8.5's
range — not because anything classified its heading text.

WHY POSITION AND NOT JUST PAGE

Sections routinely start partway down a page. On page 25 of that protocol,
five sections begin:

    3.1.10.1 Treatment beyond Progression        y=720   (top)
    3.1.11   Subject Evaluability                y=537
    3.1.12   Optional Study Treatment Extension  y=397
    3.1.13   Interim Analysis                    y=355
    3.1.14   Safety Monitoring                   y=286   (bottom)

Page alone cannot separate those five. (page, y) can.

PDF coordinates count UP from the bottom of the page, so a LARGER y is HIGHER
on the page and therefore EARLIER in reading order. That inversion is the one
thing easy to get backwards here, and getting it backwards silently assigns
every chunk to the wrong section.

WHAT THIS DOES NOT DO

    It does NOT change where the chunker splits. Boundaries still come from
    docling's own headings, which were never the problem.
    It does NOT work on documents with no outline — 10 of a 20-document corpus
    had none, and those keep the heading-text inference in headings.py.
    It does NOT read heading text at all. No title matching, no fuzzy
    comparison, no numbering rules.
"""

import os
from pathlib import Path

# Below this many resolved entries an outline is present but too thin to
# describe the document — a 15-page form with 2 entries is not a hierarchy.
# Measured across a 20-document corpus: usable outlines carried 53 to 271
# entries; the one unusable outline carried 2.
MIN_OUTLINE_ENTRIES = int(os.getenv("MIN_OUTLINE_ENTRIES", "5"))

# An outline needs enough TOP-LEVEL entries to describe a document's sections,
# not just enough entries overall. Measured across a 20-document corpus: usable
# outlines declared 9 to 26 top-level entries; the one unusable outline
# declared 2 for its 162 entries, and every position in the document resolved
# to the same root.
MIN_TOP_LEVEL_ENTRIES = int(os.getenv("MIN_TOP_LEVEL_ENTRIES", "3"))


def read_outline(pdf: Path) -> list[dict]:
    """The PDF's own table of contents, as a flat list in document order.

    Each entry:

        {"level": 0, "title": "8 Appendices", "page": 67, "y": 720.0}

    `level` is 0-based nesting depth as the PDF declares it. `page` is a
    0-based page index. `y` is the vertical position in PDF coordinates —
    larger is higher on the page.

    Returns an empty list when the document has no outline, when it has too few
    entries to be useful, or when pypdfium2 is unavailable. Every one of those
    is a normal condition, not an error: the caller falls back to inference.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print("  outline: pypdfium2 not installed, cannot read the PDF outline",
              flush=True)
        return []

    try:
        document = pdfium.PdfDocument(str(pdf))
        bookmarks = list(document.get_toc())
    except Exception as exc:
        print(f"  outline: could not be read ({type(exc).__name__}: {exc})",
              flush=True)
        return []

    entries = []
    for bookmark in bookmarks:
        try:
            title = bookmark.get_title()
            destination = bookmark.get_dest()
            if destination is None:
                # A bookmark with no destination at all — a heading in the
                # outline that points nowhere. Seen on a real 162-page
                # protocol, where it raised AttributeError on get_view().
                # Nothing to position it by, so it cannot participate.
                continue
            page = destination.get_index()
            view = destination.get_view()
            # view is (mode, [coords]); for the common XYZ mode the second
            # coordinate is the vertical position. A destination that only
            # names a page carries no coordinates at all, in which case the
            # entry starts at the top of its page.
            coords = view[1] if view and len(view) > 1 else []
            y = float(coords[1]) if len(coords) > 1 else float("inf")
        except Exception:
            # One malformed entry must not discard a whole usable outline.
            continue
        if page is None:
            continue
        entries.append({"level": bookmark.level, "title": title,
                        "page": page, "y": y})

    if len(entries) < MIN_OUTLINE_ENTRIES:
        return []

    # An outline can have plenty of entries and still declare no usable
    # hierarchy. Measured on a 162-page protocol whose outline carries 162
    # entries but only TWO at the top level: every position resolved to the
    # same root, a document identifier rather than a section, covering 98% of
    # the document. Nine other documents in the same corpus had no root above
    # 37%.
    #
    # A single root swallowing nearly everything is the same failure the page
    # ranges were meant to fix, arriving by a different route — and unlike the
    # cross-reference case, there is nothing better to extract, because the
    # structure was never declared. Falling back to heading-text inference is
    # strictly better than one root for the whole document.
    tops = sum(1 for e in entries if e["level"] == 0)
    if tops < MIN_TOP_LEVEL_ENTRIES:
        print(f"  outline: {len(entries)} entries but only {tops} at the top "
              "level — too flat to describe the document, keeping docling's "
              "heading paths", flush=True)
        return []

    return entries


def _before_or_at(chunk_page: int, chunk_y: float,
                  entry_page: int, entry_y: float) -> bool:
    """Does a section starting at (entry_page, entry_y) begin at or before a
    chunk at (chunk_page, chunk_y)?

    A later page is always after. On the SAME page, "after" means LOWER on the
    page, which in PDF coordinates means a SMALLER y — the inversion noted in
    the module docstring.
    """
    if chunk_page > entry_page:
        return True
    if chunk_page < entry_page:
        return False
    return chunk_y <= entry_y


def path_for(entries: list[dict], page: int, y: float) -> list[str]:
    """The full heading path for a position, root first.

        path_for(entries, page=100, y=400)
        -> ["8 Appendices", "8.5 RECIST 1.1 and irRECIST Guidelines"]

    WHY THIS IGNORES THE OUTLINE'S OWN PARENT/CHILD NESTING

    An outline declares two things: a nesting tree, and a target position per
    entry. It is tempting to trust the tree — it is the author's own
    structure — but nesting does NOT reliably mean containment.

    Measured across a 20-document corpus, counting child entries whose target
    page lies outside the page range their declared parent actually covers:

        NSCLC       125 entries,  9 top-level,   0 out-of-range children
        Pfizer      239 entries, 14 top-level,  17 out-of-range children
        Prostate    271 entries, 26 top-level,  31 out-of-range children
        Lupus       162 entries,  2 top-level, 160 out-of-range children

    The prostate protocol is the clearest case. Its outline nests every table
    in the document under a top-level `LIST OF TABLES` entry on page 12:

        L0  p12   LIST OF TABLES
        L1  p19     Table 1. Phase 1b Schedule of Activities
        L1  p26     Table 2. Phase 3 Schedule of Activities
        L1  p112    Table 10. Bendamustine Dose Reduction

    Those children are cross-references pointing INTO the body, not sections
    inside a two-page list. Following the declared parentage put 386 of 467
    chunks — 83% of the document — under `LIST OF TABLES`, and buried every
    real section from `1. INTRODUCTION` to `16. REFERENCES`.

    So the tree is rebuilt from positions instead. Entries are sorted by
    (page, then down the page), each one owns the span until the next entry
    begins, and a path is assembled from the entries whose spans genuinely
    enclose the position. `Table 1` on page 19 then simply becomes an entry
    starting at page 19, correctly enclosed by whichever section covers page
    19 — regardless of what the outline claimed about its parentage.

    Levels are still used, but only to decide which enclosing entries are
    ancestors of the innermost one rather than siblings of it.

    Returns an empty list for a position before the first entry — front
    matter, a title page. That is real structure, not a failure.
    """
    if not entries:
        return []

    # Position order, not list order. An outline may list a cross-reference
    # far from where it points, so the declared sequence is not the reading
    # sequence. Larger y is higher on the page, hence -y.
    order = sorted(range(len(entries)),
                   key=lambda i: (entries[i]["page"], -entries[i]["y"]))

    # The innermost entry is the last one that starts at or before the
    # position. Everything after it begins later in the document.
    innermost = None
    for index in order:
        entry = entries[index]
        if _before_or_at(page, y, entry["page"], entry["y"]):
            innermost = index
        else:
            break

    if innermost is None:
        return []

    # Walk back through POSITION order collecting strictly shallower entries.
    # Those are the ancestors: an entry earlier in the document at a shallower
    # level is still open at this position, because nothing shallower has
    # closed it yet.
    path = [entries[innermost]["title"]]
    level = entries[innermost]["level"]
    position_of = order.index(innermost)
    for index in reversed(order[:position_of]):
        if entries[index]["level"] < level:
            path.append(entries[index]["title"])
            level = entries[index]["level"]
            if level == 0:
                break
    path.reverse()
    return path


def chunk_position(chunk, page_heights: dict[int, float] | None = None):
    """A chunk's (page, y), or None when it carries no usable provenance.

    Takes the FIRST provenance item — a chunk spanning pages starts where its
    first element starts, which is what decides the section it belongs to.

    Coordinates are normalised to bottom-left origin so they can be compared
    against outline entries, which are always bottom-left. docling may report
    either convention and says which in `coord_origin`; converting needs the
    page height, so without it a top-left bbox is skipped rather than compared
    in the wrong coordinate system — a silent inversion that would assign the
    chunk to the wrong section.
    """
    items = getattr(getattr(chunk, "meta", None), "doc_items", None) or []
    for item in items:
        for prov in (getattr(item, "prov", None) or []):
            page_no = getattr(prov, "page_no", None)
            bbox = getattr(prov, "bbox", None)
            if page_no is None or bbox is None:
                continue

            origin = getattr(getattr(bbox, "coord_origin", None), "value", None)
            if origin == "TOPLEFT":
                height = (page_heights or {}).get(page_no)
                if height is None:
                    continue
                bbox = bbox.to_bottom_left_origin(height)

            # docling page numbers are 1-based; outline indices are 0-based.
            return page_no - 1, float(bbox.t)
    return None


def page_heights_of(doc) -> dict[int, float]:
    """Page height per page number, for converting top-left bounding boxes."""
    heights = {}
    for number, page in (getattr(doc, "pages", None) or {}).items():
        size = getattr(page, "size", None)
        if size is not None and getattr(size, "height", None):
            heights[number] = float(size.height)
    return heights


def apply_to_chunks(chunks, doc, pdf: Path, verbose: bool = True) -> int:
    """Overwrite each chunk's heading path from the outline.

    Runs AFTER chunking, deliberately. Changing headings BEFORE the chunker
    would mean matching outline entries to docling's heading items by title —
    fuzzy matching, which is exactly what docling's own bookmark signal does
    and what fails on these documents. After chunking, positions are compared
    as numbers, and numbers need no matching.

    Returns the number of chunks whose path changed. Zero means the document
    has no usable outline and docling's own headings were kept.
    """
    entries = read_outline(pdf)
    if not entries:
        if verbose:
            print("  outline: none usable — keeping docling's heading paths",
                  flush=True)
        return 0

    heights = page_heights_of(doc)
    changed = unpositioned = front_matter = 0

    for chunk in chunks:
        position = chunk_position(chunk, heights)
        if position is None:
            unpositioned += 1
            continue
        path = path_for(entries, *position)
        if not path:
            front_matter += 1
            continue
        if list(getattr(chunk.meta, "headings", None) or []) != path:
            chunk.meta.headings = path
            changed += 1

    if verbose:
        depth = max(e["level"] for e in entries) + 1
        print(f"  outline: {len(entries)} entries, {depth} levels — "
              f"{changed} chunk heading path(s) replaced", flush=True)
        if front_matter:
            print(f"    {front_matter} chunk(s) sit before the first outline "
                  "entry (title page, contents) and keep no path", flush=True)
        if unpositioned:
            print(f"    {unpositioned} chunk(s) carry no usable position and "
                  "keep docling's path", flush=True)
    return changed
