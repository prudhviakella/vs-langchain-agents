"""Reading tables, judging whether the parse worked, and describing them.

    TableItem
        |
        v
    export_to_markdown()
        |
        v
    table_looks_broken()
        |
     +--+--+
     |     |
    no    yes
     |     |
     v     v
  summarise   render the table as an image
  the text    and describe what it LOOKS like
     |     |
     +--+--+
        |
        v
    one summary, plus the raw fragments, sharing a table_id

WHY A TABLE NEEDS A SUMMARY AT ALL

A grid of numbers shares almost no vocabulary with "how did revenue grow" —
the words revenue and grow appear in no cell. Embedding the grid alone leaves
the table in the index and unreachable.

The summary supplies the missing words. It does NOT replace the rows: the
fragments stay, carrying the exact values, and retrieval walks from one to the
other through `table_id`.


A grid of numbers shares almost no vocabulary with a question like "how did revenue
grow" — those words appear in no cell. Embedding the grid alone leaves the table in
the index and unreachable. The summary supplies the missing vocabulary; the raw rows
stay alongside it carrying the exact values.

A structurally broken table is worse than a missing one: the rows still index, the
summary still generates, and it describes a grid that does not exist — confidently,
because nothing upstream signalled a problem. `table_looks_broken` is the check, and
`summarize_table_image` is what happens when it fires.
"""

import base64
import hashlib
import io
import os
import re

# CACHE_DIR:
#     Where table summaries are cached on disk, keyed by table content.
#
# TABLE_MODEL:
#     The text model that writes a summary from a table's Markdown.
#
# TABLE_PROMPT:
#     The instruction given to that model. Shared by the text path and the
#     image fallback, so both produce summaries of the same shape.
#
# TABLE_MIN_CELLS:
#     Below this many cells, something Docling labelled a table is treated as
#     layout rather than data, and gets no summary.
#
# VISION_MODEL:
#     Used only for the image fallback, when a table's parsed grid is unusable.
from .config import CACHE_DIR, TABLE_MODEL, TABLE_PROMPT, TABLE_MIN_CELLS, VISION_MODEL

# ---------------------------------------------------------------------------
# THRESHOLDS FOR THE "LABEL AND NUMBER IN ONE CELL" SIGNAL
#
# Both must be exceeded before a table is called broken. See
# table_looks_broken() for why a proportion alone is not enough.
# ---------------------------------------------------------------------------

# Fewer crammed cells than this and it is almost certainly a legitimate
# identifier like "Study 12345", not a missed column boundary.
MIN_CRAMMED = 3

# And they must be more than this share of the table, so a large table with a
# few such values is not flagged either.
CRAMMED_SHARE = 0.10


def table_cells(markdown: str) -> list[str]:
    """Cell contents of a Markdown table.

    Used to decide whether something is a real table at all.

    Examples:

        | Sector | 2024 |     ->  4 cells
        |--------|------|         (the separator row contributes nothing)
        | Banks  |  34% |

    Returns:
        A list of cell strings, separator rows and empty splits removed.


    The count decides whether something is a real table, so it has to be honest.
    Two things would inflate it: the separator row, which is all dashes and colons
    and carries no content, and the empty strings that splitting on the leading and
    trailing pipe produces on every row.
    """
    # ------------------------------------------------------------------
    # WHY TWO CONDITIONS
    # ------------------------------------------------------------------
    #
    # The count decides whether something is a real table, so it has to be
    # honest. Two things would inflate it:
    #
    #     1. The separator row, which is all dashes and colons and carries
    #        no content of its own.
    #
    #     2. The empty strings produced by splitting on the leading and
    #        trailing pipe of every row.
    #
    # Counting either would make a two-column layout block look like data.
    return [
        cell.strip()
        for line in markdown.splitlines()
        for cell in line.split("|")
        if cell.strip() and not set(cell.strip()) <= set("-: ")
    ]


def needs_summary(markdown: str) -> bool:
    """Whether this is a real table rather than a layout artefact.

    Four cells is the smallest grid that can carry a relationship: two
    columns and two rows.

    Below that, what Docling labelled a table is almost always something
    else — an author panel, a running header, a date line. Summarising those
    adds noise to the index without adding anything findable.

    Returns:
        True when the table has enough cells to be worth a summary.
    """
    return len(table_cells(markdown)) >= TABLE_MIN_CELLS


def table_looks_broken(markdown: str) -> list[str]:
    """Detect a table whose grid came out wrong, and say why.

    WHY THIS MATTERS MORE THAN A MISSING TABLE
    ------------------------------------------

    A table that fails to extract is obvious: nothing appears.

    A table whose STRUCTURE is wrong is not. The rows still index, the
    summary still generates, and the model writes a confident description of
    a grid that does not exist — because nothing upstream signalled anything
    was wrong.

    An analyst reads that summary and has no way to tell it apart from a
    correct one.

    WHAT A BROKEN GRID LOOKS LIKE
    -----------------------------

    From a real run, where a stacked pair of tables was merged into one:

        | Row Lab Adopter Adopte 31,912 | Row Lab Adopter Adopte 31,912 |
        |-------------------------------|-------------------------------|
        | Enabler/ 426 Protecte         | 0                             |

    Two things went wrong there, and each is detectable:

        1. The header cell is repeated side by side — one column was split
           across two.

        2. "Enabler/ 426 Protecte" holds a row label, a number, and the
           start of the next label, all in one cell.

    Returns:
        A list of human-readable reasons. Empty means the grid looks sound.


    A structurally broken table is worse than a missing one. The rows still index,
    the summary still generates, and the model describes a grid that does not exist
    — confidently, because nothing upstream signalled a problem.

    Two signals catch the common failures, both from stacked or nested tables where
    column boundaries were mis-detected:

      duplicate adjacent header cells   one column was split across two, so the same
                                        content appears twice side by side

      label and value in one cell       a row label ran into the next column's
                                        number, as in "Enabler/ 426 Protecte"
    """
    # ------------------------------------------------------------------
    # COLLECT THE CONTENT ROWS
    # ------------------------------------------------------------------
    #
    # The separator row is skipped, so rows[0] is the real header rather
    # than a line of dashes.
    rows = [
        line
        for line in markdown.splitlines()
        if "|" in line and not set(line.strip()) <= set("|-: ")
    ]

    if not rows:
        return []

    issues = []

    # ------------------------------------------------------------------
    # SIGNAL 1 — A SPLIT HEADER COLUMN
    # ------------------------------------------------------------------
    #
    # Column boundaries were mis-detected, so one column's content was
    # written into two:
    #
    #     | Row Lab Adopter Adopte 31,912 | Row Lab Adopter Adopte 31,912 |
    #
    # WHAT THIS MUST NOT FLAG — AND USED TO
    #
    # Markdown has no colspan. A header cell that spans several columns is
    # correctly rendered by repeating its text across every column it covers:
    #
    #     | Dose Level Table | Dose Level Table | ... (x7)     spans all 7
    #     | Dose Level | Arm A Escalation | Arm A Escalation | Arm B* | Arm B* |
    #
    # That is a CORRECT table. Flagging it sent 21 of 45 tables in one
    # protocol down the image-fallback path — a vision call each, producing a
    # description of a grid that was fine, while the accurate markdown was
    # discarded. The counts did not move when do_cell_matching was flipped,
    # because there was never anything wrong to fix.
    #
    # THE TEST THAT SEPARATES THEM
    #
    # A span covers a contiguous run that something FURTHER DOWN subdivides —
    # that is the whole point of a spanning header. A split column duplicates
    # a cell at every depth, because the split runs the full height of the
    # table.
    #
    #     span            | Arm A Escalation | Arm A Escalation |
    #                     | BI 1361849       | Durvalumab       |   <- differ
    #
    #     split column    | Row Lab 31,912   | Row Lab 31,912   |
    #                     | Enabler/ 426     | Enabler/ 426     |   <- same
    #
    # EVERY row is checked, not just the one below. Spans nest: a title
    # spanning seven columns sits above headers spanning two and three, so
    # the row immediately beneath a span is often a span itself. Checking one
    # row down flags the outer span as broken.
    #
    # A repeated header with no row beneath to judge by claims nothing.
    # ------------------------------------------------------------------
    def cells(row: str) -> list[str]:
        return [cell.strip() for cell in row.split("|")[1:-1]]

    # A cell holding a label welded to a number. A legitimate spanning header
    # is a title — "Arm A Escalation", "Dose Level Table". A header cell that
    # already contains a value is not a title, it is wreckage, and a
    # duplicated one is wreckage smeared across two columns.
    WELDED = re.compile(r"[A-Za-z]{3,}\s+[\d,]{3,}")

    grid = [cells(row) for row in rows]
    header = grid[0]

    split_columns = False
    for i, (a, b) in enumerate(zip(header, header[1:])):
        if not a or a != b:
            continue
        # A duplicated header that is itself welded is broken whatever sits
        # below it. This is what catches a stacked pair of tables merged into
        # one grid, where the row-label column was smeared across two columns
        # and the proportion of welded cells is too small for SIGNAL 2 to see
        # — 4 welded cells out of 200 is 2%, well under the share threshold.
        if WELDED.search(a):
            split_columns = True
            break
        # Otherwise: does anything below ever tell these two columns apart?
        subdivided = any(len(row) > i + 1 and row[i] != row[i + 1]
                         for row in grid[1:])
        if subdivided or len(grid) < 2:
            continue
        split_columns = True
        break

    if split_columns:
        issues.append("duplicate adjacent header cells")

    # ------------------------------------------------------------------
    # SIGNAL 2 — A LABEL AND A NUMBER IN ONE CELL
    # ------------------------------------------------------------------
    #
    # A word of three or more letters followed by a three-or-more digit
    # number, inside a single cell, is text and data that belong in
    # different columns:
    #
    #     "Enabler/ 426 Protecte"     <- label, value, next label
    #
    # TWO CONDITIONS, NOT ONE
    #
    # A proportion alone is wrong on small tables. Consider:
    #
    #     | Study       | Enrolment |
    #     |-------------|-----------|
    #     | Study 12345 | 400       |
    #
    # "Study 12345" is a perfectly legitimate value. But that is one cell
    # out of four, which is 25% — and a bare 10% threshold flags a correct
    # table as broken.
    #
    # So both must hold:
    #
    #     at least MIN_CRAMMED cells       rules out a small table with one
    #                                      legitimate identifier in it
    #     more than CRAMMED_SHARE of them  rules out a large table with a
    #                                      handful of such values
    #
    # A guard that fires on legitimate tables gets switched off, and then it
    # protects nothing. Being slightly too permissive here is the safer
    # error: a missed broken table is caught by reading the report, while a
    # disabled guard catches nothing at all.
    cells = table_cells(markdown)

    crammed = sum(
        1
        for cell in cells
        if re.search(r"[A-Za-z]{3,}\s+[\d,]{3,}", cell)
    )

    if crammed >= MIN_CRAMMED and crammed / len(cells) > CRAMMED_SHARE:
        issues.append(f"{crammed}/{len(cells)} cells hold a label and a number")

    return issues


def table_markdown(doc) -> dict[str, str]:
    """Full markdown for every table in the document, keyed by its element ref.

    HybridChunker may split one large table across several chunks. Summarising each
    chunk separately would describe pieces of a table rather than the table — the
    third fragment has no header row and no idea what it is. Reading the complete
    grid once from the document object avoids that.
    """
    from docling_core.types.doc import TableItem

    tables = {}
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            ref = getattr(item, "self_ref", None) or str(id(item))
            try:
                tables[ref] = item.export_to_markdown(doc)
            except Exception:
                # A malformed table should cost its own summary, not the document.
                # The empty string is reported as a problem by the caller.
                tables[ref] = ""
    return tables


def table_ref_of(chunk) -> str | None:
    """The element ref of the table this chunk came from, or None.

    This is what groups fragments of the same table so they share one summary and
    one table_id.

    Matches on the element's label rather than its Python type. `doc.iterate_items()`
    yields real TableItem instances, but `chunk.meta.doc_items` does not — the chunk
    carries lighter references that report `label=TABLE` while failing an isinstance
    check. Testing the type here returns None for every chunk, which produces no
    table groups, no summaries, and an empty table_id on every record — silently,
    because a document with no tables looks exactly the same.
    """
    for item in chunk.meta.doc_items:
        label = str(getattr(item, "label", "")).lower()
        if "table" in label:
            ref = getattr(item, "self_ref", None)
            if ref:
                return ref
    return None


def summarize_table(markdown: str, headings: list[str]) -> str:
    """Describe a table in natural language so vector search can find it.

    A grid of quarterly figures shares almost no vocabulary with "how did revenue
    grow" — the words revenue and growth appear in no cell. Embedding the grid alone
    makes the table effectively unsearchable. The summary supplies that vocabulary;
    the raw rows still carry the exact values.

    Cached by table content, so re-runs, retries, and a second document containing
    the same standard table are all free.
    """
    from openai import OpenAI

    digest = hashlib.sha256(markdown.encode()).hexdigest()[:20]
    cached = CACHE_DIR / "tables" / f"{digest}.txt"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        return cached.read_text()

    # The heading path tells the model where the table sits, which changes the
    # summary: the same numbers mean something different under "Adverse Events" than
    # under "Baseline Demographics".
    context = " > ".join(headings) if headings else ""

    response = OpenAI().chat.completions.create(
        model=TABLE_MODEL,
        # Same determinism requirement as figure descriptions: this text becomes
        # part of a chunk whose id is a hash of that text.
        temperature=0, seed=0, max_tokens=500,
        messages=[
            {"role": "system", "content": TABLE_PROMPT},
            # Truncated because a pathological table can be enormous, and the first
            # 12k characters carry the structure plus enough values to describe it.
            {"role": "user", "content": f"Section: {context}\n\n{markdown[:12000]}"},
        ],
    )
    summary = response.choices[0].message.content.strip()
    cached.write_text(summary)
    return summary


def summarize_table_image(item, doc, headings: list[str]) -> str | None:
    """Describe a table by looking at it, when its parsed structure is unusable.

    Costs roughly $0.004 against $0.0015 for the text path, and is only reached when
    the markdown is known to be wrong. The choice is not "spend more" but "spend
    more or index a description of a grid that does not exist".

    Cached like the text path, keyed on the rendered bytes.
    """
    from openai import OpenAI

    image = item.get_image(doc)
    if image is None:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()

    digest = hashlib.sha256(raw).hexdigest()[:20]
    cached = CACHE_DIR / "tables" / f"{digest}.img.txt"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        return cached.read_text()

    context = " > ".join(headings) if headings else ""
    response = OpenAI().chat.completions.create(
        model=VISION_MODEL, temperature=0, seed=0, max_tokens=600,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"Section: {context}\n\n{TABLE_PROMPT}"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(raw).decode(),
                # Table text is small; low detail loses the numbers entirely.
                "detail": "high"}},
        ]}],
    )
    summary = response.choices[0].message.content.strip()
    cached.write_text(summary)
    return summary
