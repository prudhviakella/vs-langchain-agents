"""Parsing a PDF, and reading the objects Docling returns.

    PDF
     |
     v
   six models, in order
     |
     +--  layout          finds the boxes, labels them, sets reading order
     +--  TableFormer     rows and columns inside a table box
     +--  OCR             text on scanned pages with no text layer
     +--  classifier      chart / photo / logo / diagram
     +--  CodeFormula     equations and code
     +--  vision model    writes each chart's description        <- costs money
     |
     v
   DoclingDocument       a graph of typed objects, not a string

WHAT THIS FILE DOES NOT DO

    It does NOT chunk.
    It does NOT embed.
    It does NOT decide what is worth keeping.

It returns the parsed document. Everything after that is `inspect.py`,
`tables.py` and `chunking.py`.

THE TRAP: almost every enrichment is OFF by default. A default
PdfPipelineOptions() gives you text and little else — an equation becomes the
placeholder `formula-not-decoded` and its content is gone, with no error.


"Parsing" is several models in sequence — layout analysis, table structure, OCR,
figure classification, formula reading, chart extraction, and a remote vision model
for figure descriptions. `build_pipeline_options` documents which is which and what
each one's failure looks like.

The accessor functions exist because Docling's object model moves between releases,
and a naive read of a renamed attribute returns an empty list rather than raising —
so a capability silently stops working with no error anywhere.
"""

import io
import os
import time
import warnings
from pathlib import Path

from .config import (DO_CHART_EXTRACTION, DO_CLASSIFICATION, DO_CODE,
                     DO_FORMULA, DO_OCR, FIGURE_AREA_THRESHOLD,
                     FIGURE_PROMPT, FIGURE_RENDER_SCALE,
                     TABLE_MODE_ACCURATE, VISION_MODEL)

def picture_annotations(item) -> list:
    """Annotations attached to a picture, across docling versions."""
    meta = getattr(item, "meta", None)
    if meta is not None:
        # Current layout: annotations hang off meta.
        found = getattr(meta, "annotations", None)
        if found:
            return list(found)
        # Some builds make meta itself the sequence.
        if isinstance(meta, (list, tuple)) and meta:
            return list(meta)

    # Deprecated location. The warning would fire once per picture on every parse —
    # noise that trains people to ignore warnings. Suppressed here, where the
    # fallback is deliberate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return list(getattr(item, "annotations", None) or [])


def picture_description(item) -> str | None:
    """The natural-language description of a picture, if one was produced.

    Returns None rather than an empty string when there is none, so the caller can
    tell "the vision step did not run" apart from "it ran and had nothing to say".
    The first is a configuration problem worth warning about; the second is not.
    """
    annotations = picture_annotations(item)

    # A picture can carry several annotations — a classification, a description,
    # possibly extracted chart data. Prefer the one that says what it is.
    for annotation in annotations:
        if (getattr(annotation, "kind", "") == "description"
                and getattr(annotation, "text", None)):
            return annotation.text

    # Older builds tag it differently. Any annotation carrying text is the
    # description, since classification annotations carry predicted classes instead.
    for annotation in annotations:
        text = getattr(annotation, "text", None)
        if text:
            return text
    return None


def chart_data(item):
    """Structured series extracted from a chart, when chart extraction produced any.

    This is what turns "adoption rose sharply" into the actual values. It arrives as
    a separate annotation from the prose description.
    """
    for annotation in picture_annotations(item):
        # The field name has not settled across releases and the annotation type is
        # not stable enough to match on, so probe the plausible names. This keeps
        # working through a rename instead of silently returning nothing.
        for field in ("chart_data", "data", "series", "table"):
            value = getattr(annotation, field, None)
            if value:
                return value
    return None


def check_model_access() -> None:
    """Fail early and legibly if Docling's model downloads will be rejected.

    Docling pulls its layout, table and figure-classification models from public
    HuggingFace repositories, which need no credentials. But huggingface_hub sends
    any token it finds in the environment, and an expired or wrong-scoped token
    makes the Hub return 401 — which huggingface_hub reports as
    RepositoryNotFoundError. The message names a public repo and says it does not
    exist, which sends people looking in entirely the wrong place.

    Dropping the token is safe: these repos are public, and nothing else here
    authenticates to HuggingFace.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import HfApi
        HfApi().model_info("docling-project/DocumentFigureClassifier-v2.5")
    except Exception as exc:
        if "401" in str(exc) or "expired" in str(exc).lower():
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            print("  NOTE: the HuggingFace token in this environment is rejected; "
                  "continuing anonymously (Docling's models are public)", flush=True)
        else:
            raise


def build_pipeline_options():
    """Assemble the Docling pipeline configuration.

    Docling parses a PDF into a `DoclingDocument`: a typed object graph with real
    TableItem and PictureItem nodes, reading order, and page provenance. Everything
    downstream works on that object rather than on exported text, so structure never
    has to be recovered by pattern matching.

    The catch is that almost every enrichment defaults to False. A default
    PdfPipelineOptions() gives you text and little else — equations in particular
    become the placeholder `formula-not-decoded` and their content is gone, with no
    error raised.
    """
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, PictureDescriptionApiOptions, TableFormerMode,
    )

    # Enrichment flag names have moved between releases. Reading the model fields
    # and only setting what exists keeps this working across versions instead of
    # raising AttributeError on an unfamiliar build.
    flags = {name: field.default
             for name, field in PdfPipelineOptions.model_fields.items()
             if name.startswith(("do_", "generate_", "enable_"))}

    opts = PdfPipelineOptions()

    # TableFormer reconstructs row and column structure inside a region the layout
    # model labelled TABLE, including merged cells. ACCURATE is slower than FAST and
    # materially better on the nested headers financial and clinical tables use
    # constantly — and it is still the stage most likely to produce a wrong grid,
    # which is why table_looks_broken() exists.
    opts.do_table_structure = True
    opts.table_structure_options.mode = (TableFormerMode.ACCURATE
                                         if TABLE_MODE_ACCURATE
                                         else TableFormerMode.FAST)

    # Each flag switches on a model. See parse_pdf's docstring for what each one
    # does and where its failures show up.
    requested = {
        # CodeFormula. Equations become LaTeX rather than `formula-not-decoded`.
        # Off is correct for a corpus with no equations — it is a model pass
        # over every candidate region either way.
        "do_formula_enrichment": DO_FORMULA,
        # CodeFormula again — same model reads code blocks and detects language.
        "do_code_enrichment": DO_CODE,
        # DocumentFigureClassifier. Tags each picture chart / photo / logo.
        "do_picture_classification": DO_CLASSIFICATION,
        # The remote vision model. The only stage that costs money per call, and the
        # only thing that makes a chart searchable.
        "do_picture_description": True,
        # Renders figure crops. Required by the line above — description needs pixels.
        "generate_picture_images": True,
        # Renders table crops, so a table with a broken grid can be re-read visually.
        "generate_table_images": True,
        # Docling refuses to call any API-hosted model without this. Its default is
        # local-only, which is a deliberate data-governance posture.
        "enable_remote_services": True,
        # CONDITIONAL — runs only on regions with no extractable text layer.
        #
        # Nearly free on a digital PDF, and the only thing that reads text baked
        # into a graphic. An exhibit drawn as coloured boxes with a list inside
        # loses its entire contents without this: the caption above and the
        # source line below survive because they are real page text, and
        # everything inside the boxes vanishes.
        #
        # Switch off the unconditional models instead — they are where the time
        # actually goes.
        "do_ocr": DO_OCR,
    }

    # The chart-extraction model reads numeric series out of bar and line charts
    # rather than only describing them — the difference between "adoption rose
    # sharply" and the actual Q4 value. It works from rasterised charts with
    # detectable axes and produces nothing on vector-drawn ones, so measure whether
    # it earns its model pass on your corpus before enabling it at scale.
    #
    # The flag has shipped under several names, so take whichever this build
    # exposes rather than assuming one and silently skipping the step.
    for alias in ("do_chart_extraction", "do_chart_data_extraction",
                  "do_chart_understanding", "do_picture_data"):
        if alias in flags:
            requested[alias] = DO_CHART_EXTRACTION
            break
    else:
        print("  NOTE: this docling build exposes no chart-extraction flag; charts "
              "will be described but their numeric series not read", flush=True)

    applied, unavailable = [], []
    for flag, value in requested.items():
        if flag in flags:
            setattr(opts, flag, value)
            applied.append(flag)
        else:
            unavailable.append(flag)

    opts.images_scale = FIGURE_RENDER_SCALE
    opts.picture_description_options = PictureDescriptionApiOptions(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        # temperature=0 with a fixed seed is load-bearing, not tidiness. Chunk ids
        # are hashes of chunk text, and figure descriptions are part of that text. A
        # model that words the same chart differently on each run changes those
        # hashes, so every figure chunk looks modified forever and the incremental
        # sync in stage 5 degrades into re-embedding the whole document every time.
        params={"model": VISION_MODEL, "max_tokens": 400, "temperature": 0, "seed": 0},
        prompt=FIGURE_PROMPT,
        picture_area_threshold=FIGURE_AREA_THRESHOLD,
        timeout=120,
    )

    print(f"  enrichments: {', '.join(applied)}", flush=True)
    if unavailable:
        # A silent rename would otherwise turn into a stage that quietly stops
        # running, which is exactly the failure this pipeline is built to surface.
        print(f"  NOT AVAILABLE in this docling build: {', '.join(unavailable)}",
              flush=True)
    return opts


def parse_pdf(pdf: Path):
    """Parse a PDF into a DoclingDocument.

    "Parsing" is several models in sequence, not one. Knowing which is which is what
    lets you read a bad result and know where to look:

      Layout analysis        RT-DETR trained on DocLayNet. Finds the regions on each
                             page and labels them — title, section header, text,
                             list item, caption, table, picture, formula — and
                             establishes reading order across columns. Everything
                             else keys off its output, so a mislabelled region is
                             mislabelled for the rest of the pipeline. Always runs.

      TableFormer            Reconstructs the grid inside a region labelled TABLE:
                             rows, columns, spans, merged cells. This is what
                             export_to_markdown() serialises, and when it collapses
                             a stacked pair into one grid, table_looks_broken()
                             catches it. Runs when do_table_structure is set;
                             ACCURATE mode here.

      OCR engine             EasyOCR by default, with Tesseract and RapidOCR
                             pluggable. Only reads regions with no extractable text
                             layer — a scanned page, or a chart with text baked into
                             the image. Digitally generated PDFs skip it entirely.
                             Runs when do_ocr is set; on by default.

      DocumentFigureClassifier
                             Tags each picture as chart, photo, logo or diagram.
                             Cheap, local, and what makes a header wordmark
                             distinguishable from an exhibit after the fact. Runs
                             when do_picture_classification is set.

      CodeFormula            Reads formula and code regions. Without it an equation
                             is emitted as the placeholder `formula-not-decoded` and
                             its content is simply gone. Runs when
                             do_formula_enrichment or do_code_enrichment is set.

      Chart extraction       Torch-backed, reads numeric series out of bar and line
                             charts rather than only describing them. Works from
                             rasterised charts with detectable axes; produces
                             nothing on vector-drawn ones, which is common in
                             financial and research PDFs. Runs when the
                             chart-extraction flag is set.

      Vision model (remote)  VISION_MODEL, called over the API — the only model here
                             that is not local. Writes the natural-language
                             description of each figure, which is what makes a chart
                             searchable at all. Runs when do_picture_description is
                             set, once per figure above FIGURE_AREA_THRESHOLD.

    The first six are downloaded from public HuggingFace repositories on first use
    (~500 MB) and cached by huggingface_hub thereafter. Only the last costs money
    per call.

    Every call re-parses. There is no cache here on purpose: the output depends on
    the settings above as much as on the file, so a cache keyed on the filename
    returns work done under different settings and makes a changed setting look like
    it did nothing. Getting that key right means hashing the vision model, the area
    threshold, the render scale, the prompt and the enrichment flags — machinery
    that exists only to serve the cache, and that is wrong in a way nobody notices.

    Two consequences of that follow.

    Iterating on chunking or retrieval re-parses each time. Parse once in the
    notebook and keep `doc` in the kernel; only re-run this cell when a parse
    setting actually changes.

    On AWS, a retried task re-parses from scratch. That is the price of not
    maintaining a shared cache whose invalidation nobody owns, and the state machine
    allows a single retry, so it is paid at most once per document.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    check_model_access()
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=build_pipeline_options())
    })
    started = time.time()
    doc = converter.convert(str(pdf)).document
    elapsed = time.time() - started
    pages = len(doc.pages)
    print(f"  parsed in {elapsed:.0f}s  ({elapsed / max(pages, 1):.0f}s per page)",
          flush=True)
    # Above roughly 30s a page something is running that probably should not
    # be. Every enrichment is a model pass on CPU, per element.
    if elapsed / max(pages, 1) > 30:
        print("  SLOW. Run `python profile_parse.py <pdf>` to see which "
              "enrichment is costing this — it times each one separately.",
              flush=True)
    return doc


def save_figures(doc, doc_id: str, bucket: str | None) -> dict[str, str]:
    """Write each figure PNG to S3 and return a map of element ref -> URI.

    The pixels have already been rendered so the vision model could read them.
    Throwing them away means an answer can quote a description of a chart but never
    show the chart, which is usually what a person actually wants to see.

    Returns an empty dict when no bucket is given, so the local path skips this
    without the caller needing a separate branch.
    """
    if not bucket:
        return {}
    import boto3
    from docling_core.types.doc import PictureItem

    s3, uris = boto3.client("s3"), {}
    for n, (item, _) in enumerate(doc.iterate_items()):
        if not isinstance(item, PictureItem):
            continue
        image = item.get_image(doc)
        if image is None:
            # Detected by layout analysis but never rendered — nothing to store, and
            # its description (if any) already made it into the text.
            continue
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        key = f"figures/{doc_id}/fig_{n:04d}.png"
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(),
                      ContentType="image/png")
        # self_ref is Docling's stable identifier for an element. Keying on it lets
        # build_records attach this URI to whichever chunk contains the figure.
        uris[getattr(item, "self_ref", None) or str(id(item))] = f"s3://{bucket}/{key}"
    return uris
