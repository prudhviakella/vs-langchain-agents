"""Reading tables, judging whether the parse worked, and describing them.

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

from .config import CACHE_DIR, TABLE_MODEL, TABLE_PROMPT, TABLE_MIN_CELLS, VISION_MODEL

def table_cells(markdown: str) -> list[str]:
    """Cell contents of a markdown table.

    The count decides whether something is a real table, so it has to be honest.
    Two things would inflate it: the separator row, which is all dashes and colons
    and carries no content, and the empty strings that splitting on the leading and
    trailing pipe produces on every row.
    """
    return [
        cell.strip()
        for line in markdown.splitlines()
        for cell in line.split("|")
        # A cell of only dashes, colons and spaces is the separator row.
        if cell.strip() and not set(cell.strip()) <= set("-: ")
    ]


def needs_summary(markdown: str) -> bool:
    """Whether this is a real table rather than a layout artefact."""
    return len(table_cells(markdown)) >= TABLE_MIN_CELLS


def table_looks_broken(markdown: str) -> list[str]:
    """Signals that TableFormer failed to recover the grid, with reasons.

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
    rows = [line for line in markdown.splitlines()
            if "|" in line and not set(line.strip()) <= set("|-: ")]
    if not rows:
        return []

    issues = []

    header = [cell.strip() for cell in rows[0].split("|") if cell.strip()]
    if any(a == b and a for a, b in zip(header, header[1:])):
        issues.append("duplicate adjacent header cells")

    cells = table_cells(markdown)
    # A word of three or more letters followed by a three-or-more digit number
    # inside one cell is text and data that belong in different columns.
    crammed = sum(1 for cell in cells if re.search(r"[A-Za-z]{3,}\s+[\d,]{3,}", cell))
    if cells and crammed / len(cells) > 0.10:
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
