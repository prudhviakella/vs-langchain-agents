"""Does this docling installation actually expose confidence grades?

    python check_confidence.py pdfs/your.pdf

Every other API check this session (VlmConvertOptions, HeadingHierarchyOptions)
worked on a bare options object — no parse needed, seconds to run. This one
can't: confidence lives on ConversionResult, produced only by a real convert()
call. `rag.docling_io.parse_pdf` discards the ConversionResult on the very
line after convert() — `.document` is kept, everything else is thrown away —
so confidence has been silently unavailable on every parse this pipeline has
ever run. This script keeps the full result to see what's actually there.

Costs a real parse. Point it at a small document you already have baseline
numbers for, not the 240-page one.

WHY EVERYTHING BELOW IS DEFENSIVE

Two field-location guesses have been wrong already this session — response_format
turned out to live on model_spec, not the top-level object; VlmPipelineOptions
raised ValueError rather than silently accepting an unknown field. This checks
every attribute with getattr and a fallback, and never assumes a nesting level
that hasn't been confirmed. A wrong guess here must print "(not found)", not
crash a real parse that just cost two minutes.
"""

import sys
from pathlib import Path


def probe(obj, path=""):
    """Print every plausible confidence-related attribute, however it's shaped."""
    if obj is None:
        print(f"  {path or '(confidence)'}: None — not populated on this result")
        return

    print(f"  type: {type(obj)}")
    print(f"  repr: {obj!r}"[:500])
    print()

    # The four component scores the docs describe, plus the two summary
    # grades. Try every plausible name — 'table' vs 'table_structure', the
    # fourth score was cut off in what documentation surfaced, so both are
    # tried rather than assumed.
    candidates = [
        "mean_grade", "low_grade",
        "layout_score", "ocr_score", "parse_score",
        "table_score", "table_structure_score",
        "mean_score", "low_score",
    ]
    for name in candidates:
        value = getattr(obj, name, "(not found)")
        print(f"    {name:<26}{value}")

    # Anything else actually on the object that the candidate list missed —
    # this is the part that catches a wrong guess above.
    real_attrs = [a for a in dir(obj)
                 if not a.startswith("_") and a not in candidates]
    if real_attrs:
        print(f"\n  other real attributes found: {real_attrs}")

    # Per-page breakdown, if it exists — would be finer-grained than the
    # document-level summary and worth knowing about either way.
    pages = getattr(obj, "pages", None)
    if pages:
        print(f"\n  per-page breakdown: {len(pages)} page(s) present")
        first_key = next(iter(pages), None)
        if first_key is not None:
            print(f"    sample page entry [{first_key!r}]: {pages[first_key]!r}"[:300])
    else:
        print("\n  no per-page breakdown found (pages attribute absent or empty)")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python check_confidence.py <path-to-pdf>")
        return
    pdf = Path(sys.argv[1])

    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    try:
        from rag.docling_io import build_pipeline_options
        options = build_pipeline_options()
        print("(using this project's real pipeline settings)\n")
    except Exception as exc:
        print(f"(could not load project settings, using docling defaults: {exc})\n")
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        options = PdfPipelineOptions()

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })

    print(f"parsing {pdf.name} — this costs a real conversion, not instant...")
    result = converter.convert(str(pdf))

    print(f"\nConversionResult attributes: "
          f"{[a for a in dir(result) if not a.startswith('_')]}\n")

    confidence = getattr(result, "confidence", "(no confidence attribute at all "
                                                "on ConversionResult — the feature "
                                                "may not exist on this version)")
    if isinstance(confidence, str):
        print(confidence)
        return

    print("=== result.confidence ===")
    probe(confidence)


if __name__ == "__main__":
    main()
