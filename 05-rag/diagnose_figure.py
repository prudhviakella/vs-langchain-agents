"""Why did one figure get no description?

    python diagnose_figure.py <cached-parse.json> 11
    python diagnose_figure.py pdfs/report.pdf 11

Four things can stop a figure being described, and they have different fixes:

    1  not detected            layout analysis never saw it
    2  below the area threshold  filtered before the vision model
    3  no image rendered       nothing to send
    4  the model returned nothing  it was sent and came back empty

The report says only "no description". This says which.
"""

import json
import sys
from pathlib import Path


def area_fraction(item, doc) -> float | None:
    """What share of its page this figure covers.

    This is the number `picture_area_threshold` compares against. Computing it
    here means you can see whether the threshold is the reason rather than
    assuming it.
    """
    prov = (getattr(item, "prov", None) or [None])[0]
    if prov is None or not getattr(prov, "bbox", None):
        return None
    bbox = prov.bbox
    page = doc.pages.get(prov.page_no)
    size = getattr(page, "size", None) if page else None
    if size is None:
        return None
    figure = abs(bbox.r - bbox.l) * abs(bbox.t - bbox.b)
    return figure / (size.width * size.height)


def main() -> None:
    from rag import config, docling_io

    source = Path(sys.argv[1])
    target = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if source.suffix == ".json":
        from docling_core.types.doc import DoclingDocument
        doc = DoclingDocument.model_validate_json(source.read_text())
    else:
        doc = docling_io.parse_pdf(source)

    from docling_core.types.doc import PictureItem

    print(f"threshold in config: {config.FIGURE_AREA_THRESHOLD}")
    print(f"render scale       : {config.FIGURE_RENDER_SCALE}\n")
    print(f"{'idx':>4}{'page':>6}{'area %':>9}{'image':>8}{'desc':>7}  verdict")
    print("-" * 76)

    for index, (item, _) in enumerate(doc.iterate_items()):
        if not isinstance(item, PictureItem):
            continue
        if target is not None and index != target:
            continue

        share = area_fraction(item, doc)
        rendered = item.get_image(doc) is not None
        description = docling_io.picture_description(item)
        prov = (getattr(item, "prov", None) or [None])[0]

        # Work out which of the four it is, in the order they would occur.
        if share is not None and share < config.FIGURE_AREA_THRESHOLD:
            verdict = (f"BELOW THRESHOLD — {share*100:.2f}% < "
                       f"{config.FIGURE_AREA_THRESHOLD*100:.2f}%")
        elif not rendered:
            verdict = "NO IMAGE — generate_picture_images was off at parse time"
        elif description:
            verdict = "described"
        else:
            verdict = "SENT, CAME BACK EMPTY — the model had nothing to say"

        print(f"{index:>4}{prov.page_no if prov else '?':>6}"
              f"{(f'{share*100:.2f}' if share else '?'):>9}"
              f"{'yes' if rendered else 'no':>8}"
              f"{'yes' if description else 'no':>7}  {verdict}")

        if target is not None:
            _detail(item, doc, description, rendered)


def _detail(item, doc, description, rendered) -> None:
    """Everything known about one figure, including a fresh model call."""
    print("\n── annotations on the item")
    annotations = docling_io.picture_annotations(item)
    if not annotations:
        print("   none. The vision step produced nothing for this figure.")
    for annotation in annotations:
        kind = getattr(annotation, "kind", type(annotation).__name__)
        text = getattr(annotation, "text", "")
        classes = getattr(annotation, "predicted_classes", None)
        print(f"   {kind}: {str(text)[:200] or classes}")

    if description:
        print(f"\n── description\n   {description[:400]}")
        return

    if not rendered:
        print("\n   No rendered image, so nothing could be sent. Re-parse with "
              "generate_picture_images on.")
        return

    # Send it again, on its own, and print the raw reply. This separates "the
    # model declined" from "the pipeline never asked".
    print("\n── calling the vision model directly on this crop")
    import base64
    import io
    import os

    from openai import OpenAI

    from rag.config import FIGURE_PROMPT, VISION_MODEL

    buffer = io.BytesIO()
    item.get_image(doc).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    print(f"   crop is {len(buffer.getvalue()):,} bytes")

    try:
        reply = OpenAI().chat.completions.create(
            model=VISION_MODEL, max_tokens=300, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": FIGURE_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + encoded, "detail": "high"}}]}])
        content = reply.choices[0].message.content
        reason = reply.choices[0].finish_reason
        print(f"   finish_reason: {reason}")
        print(f"   content: {content!r}")
        if not content:
            print("\n   The model was asked and returned nothing. The prompt asks "
                  "for chart type,\n   axes and units — a logo or a decorative "
                  "band has none of those, so the\n   model can legitimately have "
                  "nothing to say. That is a PROMPT fit problem,\n   not a "
                  "threshold or rendering one.")
    except Exception as exc:
        print(f"   call failed: {exc}")


if __name__ == "__main__":
    main()
