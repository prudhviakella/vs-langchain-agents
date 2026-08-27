"""Time each enrichment separately, to find where the 22 minutes went.

    python profile_parse.py pdfs/your.pdf

Runs the same PDF several times, adding one flag at a time. The difference
between two rows is what that flag costs on YOUR machine with YOUR document.

Nothing here is a guess. Run it before changing any setting.
"""

import sys
import time
from pathlib import Path


def parse_with(pdf: Path, **flags) -> tuple[float, int]:
    """Parse once with an explicit flag set. Returns (seconds, pages)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                    TableFormerMode)
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()

    # Everything off unless asked for. The default is not "off" for all of
    # these, so they are set explicitly rather than assumed.
    for name in ("do_table_structure", "do_formula_enrichment",
                 "do_code_enrichment", "do_picture_classification",
                 "do_picture_description", "generate_picture_images",
                 "generate_table_images", "generate_page_images",
                 "do_chart_extraction", "enable_remote_services"):
        if name in PdfPipelineOptions.model_fields:
            setattr(opts, name, False)

    for name, value in flags.items():
        if name == "table_mode":
            opts.table_structure_options.mode = value
        elif name == "images_scale":
            opts.images_scale = value
        elif name in PdfPipelineOptions.model_fields:
            setattr(opts, name, value)

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    started = time.time()
    doc = converter.convert(str(pdf)).document
    return time.time() - started, len(doc.pages)


def main() -> None:
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "pdfs/sample.pdf")
    if not pdf.exists():
        raise SystemExit(f"not found: {pdf}")

    from docling.datamodel.pipeline_options import TableFormerMode

    # Cumulative, so each row's cost is its difference from the row above.
    steps = [
        ("text only", {}),
        ("+ tables FAST", {"do_table_structure": True,
                           "table_mode": TableFormerMode.FAST}),
        ("+ tables ACCURATE", {"do_table_structure": True,
                               "table_mode": TableFormerMode.ACCURATE}),
        ("+ picture images 1x", {"do_table_structure": True,
                                 "table_mode": TableFormerMode.ACCURATE,
                                 "generate_picture_images": True,
                                 "images_scale": 1.0}),
        ("+ picture images 2x", {"do_table_structure": True,
                                 "table_mode": TableFormerMode.ACCURATE,
                                 "generate_picture_images": True,
                                 "images_scale": 2.0}),
        ("+ classification", {"do_table_structure": True,
                              "table_mode": TableFormerMode.ACCURATE,
                              "generate_picture_images": True,
                              "images_scale": 2.0,
                              "do_picture_classification": True}),
        ("+ formula/code", {"do_table_structure": True,
                            "table_mode": TableFormerMode.ACCURATE,
                            "generate_picture_images": True,
                            "images_scale": 2.0,
                            "do_picture_classification": True,
                            "do_formula_enrichment": True,
                            "do_code_enrichment": True}),
        ("+ chart extraction", {"do_table_structure": True,
                                "table_mode": TableFormerMode.ACCURATE,
                                "generate_picture_images": True,
                                "images_scale": 2.0,
                                "do_picture_classification": True,
                                "do_formula_enrichment": True,
                                "do_code_enrichment": True,
                                "do_chart_extraction": True}),
    ]

    print(f"{pdf.name}\n")
    print(f"{'configuration':<24}{'seconds':>9}{'delta':>9}  what that flag costs")
    print("-" * 74)

    previous = None
    for label, flags in steps:
        try:
            elapsed, pages = parse_with(pdf, **flags)
        except Exception as exc:
            print(f"{label:<24}{'FAILED':>9}  {str(exc)[:40]}")
            continue
        delta = "" if previous is None else f"+{elapsed - previous:.0f}s"
        print(f"{label:<24}{elapsed:>8.0f}s{delta:>9}")
        previous = elapsed

    print(f"\n{pages} pages. The vision descriptions are NOT included above — "
          "they are API calls,\nroughly 2-4s each, and run once per figure.")


if __name__ == "__main__":
    main()
