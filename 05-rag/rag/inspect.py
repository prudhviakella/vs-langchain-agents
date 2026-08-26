"""Verifying the parse, before anything is chunked or embedded.

    parsed document
          |
          v
    inspect()                counts, printed, with warnings
          |
          v
    write_extraction_report()   every element, readable, next to the PDF
          |
          v
      you read it
          |
          v
    only then: chunk and embed

WHY THIS EXISTS

Extraction failures do NOT raise. A setting left off yields an empty annotation
or a placeholder, every later stage runs happily on top of it, and you get an
index that looks complete and is missing its tables or its equations.

Nothing downstream can detect that. This is the only place it becomes visible,
and it runs before any money is spent on embeddings.


Extraction failures do not raise. A disabled enrichment yields an empty annotation
or a placeholder, and every later stage runs happily on top of it, producing an
index that looks complete and is missing its tables or its equations. Nothing
downstream can detect that.

This module is where it becomes visible — as counts in the log, and as a report you
can read next to the PDF.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .config import REPORT_DIR, VISION_MODEL
from .docling_io import chart_data, picture_description
from .tables import table_cells, table_looks_broken

def inspect(doc, markdown: str) -> dict:
    """Report what the parse produced, and warn on the silent failures.

    Extraction failures do not raise. A disabled enrichment yields an empty
    annotation or a placeholder, and every later stage runs happily on top of it,
    producing an index that looks complete and is missing its tables or equations.
    Nothing downstream can detect that. This is where it becomes visible, before any
    money is spent on embeddings.
    """
    from docling_core.types.doc import PictureItem, TableItem

    items = [item for item, _ in doc.iterate_items()]
    pictures = [item for item in items if isinstance(item, PictureItem)]
    tables = [item for item in items if isinstance(item, TableItem)]

    # Counted separately from the pictures themselves: pictures found but zero
    # described means the vision step no-op'd.
    described = sum(1 for p in pictures if picture_description(p))
    with_data = sum(1 for p in pictures if chart_data(p) is not None)

    suspect = 0
    for table in tables:
        try:
            if table_looks_broken(table.export_to_markdown(doc)):
                suspect += 1
        except Exception:
            suspect += 1

    report = {
        "pages": len(doc.pages),
        "tables": len(tables),
        "tables_suspect": suspect,
        "pictures": len(pictures),
        "pictures_described": described,
        "charts_with_data": with_data,
        # The placeholder Docling emits when formula enrichment is off. Counting it
        # in the exported markdown is simpler than walking the tree for formula
        # nodes and catches the same failure.
        "formulas_undecoded": markdown.count("formula-not-decoded"),
    }
    print("  " + "  ".join(f"{k}={v}" for k, v in report.items()), flush=True)

    if pictures and not described:
        print("  WARNING: no figure descriptions produced", flush=True)
    if report["formulas_undecoded"]:
        print(f"  WARNING: {report['formulas_undecoded']} undecoded formulas",
              flush=True)
    if suspect:
        print(f"  WARNING: {suspect} tables have suspect structure — their summaries "
              "will be generated from the rendered image", flush=True)
    return report


def _element_kind(item) -> str:
    """Short label for an element, used as its heading in the report."""
    from docling_core.types.doc import PictureItem, TableItem

    if isinstance(item, TableItem):
        return "TABLE"
    if isinstance(item, PictureItem):
        return "FIGURE"
    label = str(getattr(item, "label", "")).upper()
    # DocItemLabel renders as "DOCITEMLABEL.SECTION_HEADER"; keep the last part.
    return label.rsplit(".", 1)[-1] or "TEXT"


def write_extraction_report(doc, doc_id: str, pdf: Path) -> Path:
    """Write a human-readable dump of everything extraction produced.

    Ordering follows the document and page breaks mirror the PDF's, so the report
    can be read side by side with the original.
    """
    from docling_core.types.doc import PictureItem, TableItem

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"{doc_id}.extract.md"
    json_path = REPORT_DIR / f"{doc_id}.extract.json"

    body: list[str] = []
    inventory: list[dict] = []
    problems: list[str] = []
    counts: Counter = Counter()
    current_page = None

    for index, (item, _) in enumerate(doc.iterate_items()):
        kind = _element_kind(item)
        counts[kind] += 1
        prov = getattr(item, "prov", None) or []
        page = prov[0].page_no if prov else None

        if page != current_page:
            current_page = page
            body.append(f"\n## Page {page}\n")

        record = {"index": index, "kind": kind, "page": page}

        if isinstance(item, TableItem):
            try:
                markdown = item.export_to_markdown(doc)
            except Exception as exc:
                markdown = ""
                problems.append(
                    f"p{page}: table {index} could not be serialised ({exc})")

            cells = len(table_cells(markdown)) if markdown else 0
            # Named distinctly from `problems`: rebinding that name here would
            # silently redirect every later figure and formula problem into this
            # table's issue list.
            structure_problems = table_looks_broken(markdown) if markdown else []

            body.append(f"### TABLE · {cells} cells")
            body.append("")
            if structure_problems:
                joined = "; ".join(structure_problems)
                problems.append(
                    f"p{page}: table {index} structure looks wrong ({joined})")
                body.append(f"> **STRUCTURE SUSPECT** — {joined}. The summary will "
                            "be generated from the rendered image instead.")
                body.append("")
            if markdown:
                body.append(markdown)
            else:
                # An unserialisable table still produces chunks downstream, so this
                # has to be visible or it looks like the table was never there.
                body.append("> **MISSING** — export_to_markdown() failed. The rows "
                            "will be indexed without structure.")
            body.append("")
            record.update(cells=cells, chars=len(markdown), markdown=markdown,
                          structure_problems=structure_problems)

        elif isinstance(item, PictureItem):
            description = picture_description(item)
            series = chart_data(item)

            body.append("### FIGURE")
            body.append("")
            if description:
                body.append(description)
            else:
                problems.append(f"p{page}: figure {index} has no description")
                body.append("> **MISSING** — no description was produced. This "
                            "figure is invisible to every query.")
            if series is not None:
                body.append("")
                body.append(f"```\nchart data: {str(series)[:2000]}\n```")
            body.append("")
            record.update(description=description,
                          has_chart_data=series is not None)

        else:
            text = getattr(item, "text", "") or ""
            if not text.strip():
                continue
            if "formula-not-decoded" in text:
                problems.append(f"p{page}: formula {index} not decoded")
            body.append(f"### {kind}")
            body.append("")
            body.append(text)
            body.append("")
            record.update(text=text, chars=len(text))

        inventory.append(record)

    header = [
        f"# {doc_id}", "",
        f"- source: `{pdf.name}`",
        f"- extracted: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- vision model: `{VISION_MODEL}`",
        f"- pages: {len(doc.pages)}",
        "", "## Extraction problems", "",
        *([f"- {p}" for p in problems] or ["- none detected"]),
        "", "## Element counts", "",
        *[f"- {k}: {v}" for k, v in counts.most_common()],
        "", "---",
    ]

    md_path.write_text("\n".join(header + body), encoding="utf-8")
    json_path.write_text(json.dumps({
        "doc_id": doc_id, "source": pdf.name, "pages": len(doc.pages),
        "vision_model": VISION_MODEL, "counts": dict(counts),
        "problems": problems, "elements": inventory,
    }, indent=2, default=str), encoding="utf-8")

    print(f"  extraction report: {md_path}", flush=True)
    if problems:
        print(f"  {len(problems)} extraction problems — see the report", flush=True)
    return md_path


def write_chunk_report(records: list[dict], doc_id: str) -> Path:
    """Write every chunk as it will be embedded, with table summaries in place.

    The extraction report answers "did the parse work". This answers "is the string
    being embedded the right string" — the exact text, its token count, and each
    table summary sitting immediately before the fragments it describes.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"{doc_id}.chunks.md"
    json_path = REPORT_DIR / f"{doc_id}.chunks.json"

    counts = Counter(r["meta"]["content_type"] for r in records)
    lines = [
        f"# {doc_id} — chunks", "",
        f"- total: {len(records)}",
        *[f"- {k}: {v}" for k, v in counts.most_common()],
        f"- truncated: {sum(1 for r in records if r['meta'].get('truncated'))}",
        f"- from table image: "
        f"{sum(1 for r in records if r['meta'].get('summary_source') == 'image')}",
        "", "---", "",
    ]

    for record in records:
        meta = record["meta"]
        lines.append(f"## [{meta['position']:>3}] {meta['content_type']}"
                     f" · p{meta['page']}-{meta['page_end']}"
                     f" · {meta['n_tokens']} tokens")
        lines.append("")
        lines.append(f"`{meta['chunk_id']}`")
        if meta["headings"]:
            lines.append(f"headings: {' > '.join(meta['headings'])}")
        if meta["table_id"]:
            lines.append(f"table_id: `{meta['table_id']}`")
        if meta.get("summary_source") == "image":
            lines.append("> generated from the rendered table image, because the "
                         "parsed grid was structurally unsound")
        if meta.get("image_uri"):
            lines.append(f"image: {meta['image_uri']}")
        if meta.get("truncated"):
            lines.append("> **TRUNCATED** — this element could not be split below "
                         "the token budget and its tail was dropped.")
        lines.append("")
        # The full text, not the metadata copy: this is what gets embedded.
        lines.append("```")
        lines.append(record["text"])
        lines.append("```")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(
        [{k: v for k, v in r["meta"].items() if k != "text"} for r in records],
        indent=2, default=str), encoding="utf-8")

    print(f"  chunk report: {md_path}", flush=True)
    return md_path
