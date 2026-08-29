"""Parsing a PDF, and reading the objects Docling returns.

    PDF
     |
     v
   layout model        finds the boxes, labels them, sets reading order,
     |                 and links captions to figures      <- selectable
     |
     +--  TableFormer     rows and columns inside a table box
     +--  OCR             text on scanned pages with no text layer
     +--  classifier      chart / photo / logo / diagram
     +--  CodeFormula     equations and code
     |
     v
   DoclingDocument     a graph of typed objects, not a string
     |
     v
   describe_figures()  vision model, CACHED on the rendered bytes  <- costs money
     |
     v
   DoclingDocument     same object, now with a description per figure

WHAT THIS FILE DOES NOT DO

    It does NOT chunk.
    It does NOT embed.
    It does NOT decide what is worth keeping.

It returns the parsed document. Everything after that is `inspect.py`,
`tables.py` and `chunking.py`.

THE TRAP: almost every enrichment is OFF by default. A default
PdfPipelineOptions() gives you text and little else — an equation becomes the
placeholder `formula-not-decoded` and its content is gone, with no error.

THE LAYOUT MODEL DECIDES MORE THAN IT LOOKS LIKE

Everything downstream keys off the layout model's labels. A region it calls
SECTION_HEADER becomes a heading in the chunker, which is a chunk boundary AND
the string prepended into every vector beneath it. A caption it fails to
associate with a figure becomes an orphan text element, and the figure loses
its exhibit number.

Both of those are layout-model output, not chunker behaviour, and `LAYOUT_MODEL`
below is the only setting that changes them at the source. `headings.py` exists
to repair what the model gets wrong; the cheaper fix is for it to be wrong less
often.

WHY FIGURE DESCRIPTIONS ARE CACHED HERE AND NOT LEFT TO DOCLING

Chunk ids are hashes of chunk text, and a figure's description is part of that
text. The vision model is the only non-deterministic stage in this pipeline —
temperature=0 and a fixed seed make OpenAI best-effort reproducible, not
reproducible — so re-parsing an unchanged PDF can reword sixteen descriptions,
change sixteen chunk ids, and make `sync.py` delete and re-embed every figure in
the document.

It is also the only stage that costs money per call, and `parse_pdf` re-parses
every time by design.

So the description step is taken out of Docling's pipeline and run here against
a cache keyed on the rendered image bytes, the prompt and the model — the same
shape `tables.py` already uses for table image summaries, and `embedding.py` for
vectors. Change the prompt, the model or the render scale and the key changes.
Change nothing and the description is byte-identical, forever, for free.

Set CACHE_FIGURE_DESCRIPTIONS=0 to hand the step back to Docling.
"""

import base64
import hashlib
import io
import os
import time
import warnings
from pathlib import Path

from .config import (CACHE_DIR, DO_CHART_EXTRACTION, DO_CLASSIFICATION, DO_CODE,
                     DO_FORMULA, DO_OCR, FIGURE_AREA_THRESHOLD,
                     FIGURE_PROMPT, FIGURE_RENDER_SCALE,
                     TABLE_MODE_ACCURATE, VISION_MODEL)

# ═══════════════════════════════════════════════════════════════════════════
# PARSE SETTINGS
#
# These belong in config.py alongside DO_OCR and the rest, and should move
# there next time that file is touched. They are here so this change is one
# file.
# ═══════════════════════════════════════════════════════════════════════════

# Which layout model runs. "" keeps Docling's default (Heron).
#
#   heron         the default: balanced
#   heron_101     the accuracy variant, ~78% mAP on DocLayNet
#   egret_medium  faster
#   egret_large   more accurate
#   egret_xlarge  most accurate
#
# THE EGRET MODELS ARE KNOWN BROKEN on some builds. Their HuggingFace configs
# use hyphenated label names ("List-item"), and _build_label_map normalises
# with .upper() only, producing "LIST-ITEM" against a DocItemLabel enum that
# expects "LIST_ITEM" — a KeyError at pipeline init. Heron models use
# underscores and are unaffected. Try heron_101 first.
LAYOUT_MODEL = os.getenv("LAYOUT_MODEL", "").strip().lower()

# TableFormer predicts a grid, then matches its predicted cells against the
# text cells the PDF backend extracted. That matching is usually right and is
# the classic cause of duplicated adjacent header cells when it is not.
# Setting this False makes TableFormer use its own text instead.
#
# One flag, two failure modes, no way to tell them apart without trying both.
# If table_looks_broken() fires on a table, flip this and re-parse before
# reaching for the image fallback.
TABLE_CELL_MATCHING = os.getenv("TABLE_CELL_MATCHING", "1") == "1"

# Run the vision description here, against a cache, instead of inside
# Docling's pipeline. See the module docstring.
CACHE_FIGURE_DESCRIPTIONS = os.getenv("CACHE_FIGURE_DESCRIPTIONS", "1") == "1"

# Whether elements belonging to no larger structure get a cluster of their own.
# Unset leaves the build's default alone, which is the right starting point:
# it interacts with the layout model, and two variables changed together
# explain nothing.
#
# Worth trying when a bulleted list inside an exhibit box stops appearing as
# list items and turns up inside a picture region instead.
_orphans = os.getenv("CREATE_ORPHAN_CLUSTERS")
CREATE_ORPHAN_CLUSTERS = None if _orphans is None else _orphans == "1"

# Layout detection confidence, 0.0 to 1.0. Unset leaves the build's default
# (0.3 on current Heron presets).
#
# This is the knob for over-detection. heron_101 found 26 pictures on a 7-page
# report where the default found 17, absorbing two tables and seventeen list
# items into figure regions. Raising the threshold suppresses the marginal
# boxes without changing model; lowering it recovers faint ones.
#
# Raise it in small steps and read the element counts in the extraction report.
# Too high and real exhibits disappear, which looks like a clean parse.
_threshold = os.getenv("LAYOUT_SCORE_THRESHOLD")
LAYOUT_SCORE_THRESHOLD = None if _threshold is None else float(_threshold)

FIGURE_CACHE = CACHE_DIR / "figures"


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

    # Deprecated location. The warning would fire once per picture on every
    # parse — noise that trains people to ignore warnings. Suppressed here,
    # where the fallback is deliberate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return list(getattr(item, "annotations", None) or [])


def picture_description(item) -> str | None:
    """The natural-language description of a picture, if one was produced.

    Returns None rather than an empty string when there is none, so the caller
    can tell "the vision step did not run" apart from "it ran and had nothing
    to say". The first is a configuration problem worth warning about; the
    second is not.
    """
    annotations = picture_annotations(item)

    # A picture can carry several annotations — a classification, a
    # description, possibly extracted chart data. Prefer the one that says
    # what it is.
    for annotation in annotations:
        if (getattr(annotation, "kind", "") == "description"
                and getattr(annotation, "text", None)):
            return annotation.text

    # Older builds tag it differently. Any annotation carrying text is the
    # description, since classification annotations carry predicted classes
    # instead.
    for annotation in annotations:
        text = getattr(annotation, "text", None)
        if text:
            return text
    return None


# Annotation kinds that are definitely NOT chart series. Probing field names on
# these is how a classification object's unrelated attribute ends up
# stringified into embedded text.
NON_CHART_KINDS = {"description", "classification", "molecule_data", "misc"}


def chart_data(item):
    """Structured series extracted from a chart, when chart extraction produced any.

        annotation  ->  is its kind a chart kind?  ->  probe the field names
                              |
                        description / classification
                              |
                              v
                            skipped

    This is what turns "adoption rose sharply" into the actual values. It
    arrives as a separate annotation from the prose description.

    WHAT CHANGED AND WHY

    This used to probe ("chart_data", "data", "series", "table") on EVERY
    annotation and return the first truthy hit. A description or
    classification object exposing an unrelated `data` attribute — pydantic
    models grow fields between releases — would be returned as chart series
    and stringified into the embedded text of that figure. Wrong content, no
    error, invisible in every report.

    Now the annotation kind is checked first, and the value must look like a
    series rather than merely being truthy.
    """
    for annotation in picture_annotations(item):
        kind = str(getattr(annotation, "kind", "")).lower()
        if kind in NON_CHART_KINDS:
            continue
        # The field name has not settled across releases, so probe the
        # plausible ones. This keeps working through a rename instead of
        # silently returning nothing.
        for field in ("chart_data", "data", "series", "table"):
            value = getattr(annotation, field, None)
            # A series is a collection. A truthy scalar here is a field that
            # happens to share a name, not chart data.
            if value and isinstance(value, (list, tuple, dict)):
                return value
    return None


def check_model_access() -> None:
    """Fail early and legibly if Docling's model downloads will be rejected.

    Docling pulls its layout, table and figure-classification models from
    public HuggingFace repositories, which need no credentials. But
    huggingface_hub sends any token it finds in the environment, and an
    expired or wrong-scoped token makes the Hub return 401 — which
    huggingface_hub reports as RepositoryNotFoundError. The message names a
    public repo and says it does not exist, which sends people looking in
    entirely the wrong place.

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


def _require_openai_key() -> str:
    """The OpenAI key, or a message that says what to do about it.

    os.environ['OPENAI_API_KEY'] raises a bare KeyError from three frames deep
    inside pipeline construction, which reads like a bug in this file.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The vision model writes the figure "
            "descriptions, and without them every chart in the document is "
            "unsearchable. Export the key, or set "
            "CACHE_FIGURE_DESCRIPTIONS=1 with a warm .cache/figures to run "
            "from cache alone.")
    return key


def _apply_layout_model(opts) -> None:
    """Point the pipeline at the layout model named by LAYOUT_MODEL.

        LAYOUT_MODEL=""            leave Docling's default alone
        LAYOUT_MODEL="heron_101"   preset first, model_spec as fallback

    TWO APIS, AND THE OLDER ONE IS A TRANSLATION

    Newer Docling exposes LayoutObjectDetectionOptions.from_preset(
    "layout_heron_default"). The older LayoutOptions(model_spec=...) still
    works, but the docs say it is *translated* onto the object-detection path —
    and a translation is a place where a confidence threshold or a
    postprocessing default can differ from what the preset would have set.

    That matters here. The measured difference between two layout models on one
    document was large enough to change which regions came out as tables, so
    "we asked for heron_101 and got something adjacent to it" is not a
    difference this pipeline can afford to be unsure about.

    So the preset path is tried first, by name, and the legacy path is the
    fallback. Which one ran is printed, because the two are not guaranteed
    equivalent and a comparison between layout models is worthless if you do
    not know which API produced each side.

    WHAT THIS DOES NOT DO

        It does not raise. An unknown name, a build without either API, or a
        model whose weights will not load all print and leave the default in
        place. A parse that silently used a different layout model than you
        asked for is worse than one that says it could not.
    """
    if not LAYOUT_MODEL:
        return

    from docling.datamodel import pipeline_options as po

    # ── the current API: named presets ───────────────────────────────────
    preset_class = getattr(po, "LayoutObjectDetectionOptions", None)
    if preset_class is not None and hasattr(preset_class, "from_preset"):
        # Presets are named layout_<model>_<variant>. Try the exact name the
        # user gave, then the conventional "_default" suffix.
        for preset in (f"layout_{LAYOUT_MODEL}", f"layout_{LAYOUT_MODEL}_default"):
            try:
                opts.layout_options = preset_class.from_preset(preset)
                print(f"  layout model: {LAYOUT_MODEL} (preset {preset!r})",
                      flush=True)
                return
            except Exception:
                continue

    # ── the legacy API: a model spec object ──────────────────────────────
    try:
        from docling.datamodel import layout_model_specs as specs
    except ImportError:
        print("  NOTE: this docling build exposes neither layout presets nor "
              f"layout_model_specs; LAYOUT_MODEL={LAYOUT_MODEL!r} ignored",
              flush=True)
        return

    name = f"DOCLING_LAYOUT_{LAYOUT_MODEL.upper()}"
    spec = getattr(specs, name, None)
    if spec is None:
        available = sorted(n.replace("DOCLING_LAYOUT_", "").lower()
                           for n in dir(specs) if n.startswith("DOCLING_LAYOUT_"))
        print(f"  NOTE: no layout model {LAYOUT_MODEL!r}; available: "
              f"{', '.join(available)}. Using the default.", flush=True)
        return

    try:
        opts.layout_options = po.LayoutOptions(model_spec=spec)
        print(f"  layout model: {LAYOUT_MODEL} (legacy model_spec — docling "
              "translates this onto the object-detection path, so it may not "
              "be identical to the preset)", flush=True)
    except Exception as exc:
        # Egret builds raise here on the hyphen/underscore label-map bug.
        print(f"  NOTE: could not select layout model {LAYOUT_MODEL!r} "
              f"({exc}); using the default", flush=True)


def build_pipeline_options():
    """Assemble the Docling pipeline configuration.

        1  layout model      which one, if not the default
        2  table structure   mode, and cell matching
        3  enrichments       set only the flags this build exposes
        4  figure images     rendered here, described later
        5  vision model      only when CACHE_FIGURE_DESCRIPTIONS is off

    Docling parses a PDF into a `DoclingDocument`: a typed object graph with
    real TableItem and PictureItem nodes, reading order, and page provenance.
    Everything downstream works on that object rather than on exported text, so
    structure never has to be recovered by pattern matching.

    The catch is that almost every enrichment defaults to False. A default
    PdfPipelineOptions() gives you text and little else — equations in
    particular become the placeholder `formula-not-decoded` and their content
    is gone, with no error raised.
    """
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, PictureDescriptionApiOptions, TableFormerMode,
    )

    # Enrichment flag names have moved between releases. Reading the model
    # fields and only setting what exists keeps this working across versions
    # instead of raising AttributeError on an unfamiliar build.
    flags = {name: field.default
             for name, field in PdfPipelineOptions.model_fields.items()
             if name.startswith(("do_", "generate_", "enable_"))}

    opts = PdfPipelineOptions()

    # ── STEP 1 — layout model ────────────────────────────────────────────
    # Sets the labels everything downstream keys off, including which regions
    # become headings, which become tables, and which captions get linked to
    # their figure.
    _apply_layout_model(opts)

    # These two live on `layout_options`, NOT on PdfPipelineOptions. Setting
    # them on the pipeline object appears to work — Python adds the attribute,
    # nothing raises — and has no effect on anything. An earlier version did
    # exactly that and printed "(absent)" against a value that was really
    # sitting one level down, set to True.
    #
    #   create_orphan_clusters   whether an element belonging to no larger
    #                            structure gets a cluster of its own. When it
    #                            does not, it is absorbed into the region
    #                            around it — one way a bulleted list inside an
    #                            exhibit box stops being list items and becomes
    #                            part of a picture.
    #
    #   score_threshold          detection confidence. Lower means more boxes.
    #                            The direct lever when one layout model finds
    #                            26 figures where another found 17: raising it
    #                            suppresses the marginal detections rather than
    #                            changing model.
    #
    # Both are left alone unless asked. They interact with the layout model,
    # and two variables changed together explain nothing.
    layout = getattr(opts, "layout_options", None)

    if CREATE_ORPHAN_CLUSTERS is not None:
        if layout is not None and hasattr(layout, "create_orphan_clusters"):
            layout.create_orphan_clusters = CREATE_ORPHAN_CLUSTERS
        else:
            print("  NOTE: this build exposes no create_orphan_clusters on "
                  "layout_options; CREATE_ORPHAN_CLUSTERS ignored", flush=True)

    if LAYOUT_SCORE_THRESHOLD is not None:
        engine = getattr(layout, "engine_options", None) if layout else None
        if engine is not None and hasattr(engine, "score_threshold"):
            engine.score_threshold = LAYOUT_SCORE_THRESHOLD
        else:
            print("  NOTE: this build exposes no score_threshold on "
                  "layout_options.engine_options; LAYOUT_SCORE_THRESHOLD "
                  "ignored", flush=True)

    # ── STEP 2 — table structure ─────────────────────────────────────────
    # TableFormer reconstructs row and column structure inside a region the
    # layout model labelled TABLE, including merged cells. ACCURATE is slower
    # than FAST and materially better on the nested headers financial and
    # clinical tables use constantly — and it is still the stage most likely to
    # produce a wrong grid, which is why table_looks_broken() exists.
    opts.do_table_structure = True
    opts.table_structure_options.mode = (TableFormerMode.ACCURATE
                                         if TABLE_MODE_ACCURATE
                                         else TableFormerMode.FAST)
    if hasattr(opts.table_structure_options, "do_cell_matching"):
        opts.table_structure_options.do_cell_matching = TABLE_CELL_MATCHING

    # ── STEP 3 — enrichments ─────────────────────────────────────────────
    # Each flag switches on a model. See parse_pdf's docstring for what each
    # one does and where its failures show up.
    describe_in_docling = not CACHE_FIGURE_DESCRIPTIONS
    requested = {
        # CodeFormula. Equations become LaTeX rather than `formula-not-decoded`.
        # Off is correct for a corpus with no equations — it is a model pass
        # over every candidate region either way.
        "do_formula_enrichment": DO_FORMULA,
        # CodeFormula again — same model reads code blocks and detects language.
        "do_code_enrichment": DO_CODE,
        # DocumentFigureClassifier. Tags each picture chart / photo / logo.
        "do_picture_classification": DO_CLASSIFICATION,
        # The remote vision model. Off by default now — describe_figures()
        # runs it against a cache after the parse instead.
        "do_picture_description": describe_in_docling,
        # Renders figure crops. Required either way: the description step needs
        # pixels, and so does save_figures().
        "generate_picture_images": True,
        # Renders table crops, so a table with a broken grid can be re-read
        # visually.
        "generate_table_images": True,
        # Docling refuses to call any API-hosted model without this. Its
        # default is local-only, which is a deliberate data-governance posture.
        "enable_remote_services": describe_in_docling,
        # CONDITIONAL — runs only on regions with no extractable text layer.
        #
        # Nearly free on a digital PDF, and the only thing that reads text
        # baked into a graphic. An exhibit drawn as coloured boxes with a list
        # inside loses its entire contents without this: the caption above and
        # the source line below survive because they are real page text, and
        # everything inside the boxes vanishes.
        #
        # Switch off the unconditional models instead — they are where the
        # time actually goes.
        "do_ocr": DO_OCR,
    }

    # The chart-extraction model reads numeric series out of bar and line
    # charts rather than only describing them — the difference between
    # "adoption rose sharply" and the actual Q4 value. It works from rasterised
    # charts with detectable axes and produces nothing on vector-drawn ones, so
    # measure whether it earns its model pass on your corpus before enabling it
    # at scale.
    #
    # The flag has shipped under several names, so take whichever this build
    # exposes rather than assuming one and silently skipping the step.
    for alias in ("do_chart_extraction", "do_chart_data_extraction",
                  "do_chart_understanding", "do_picture_data"):
        if alias in flags:
            requested[alias] = DO_CHART_EXTRACTION
            break
    else:
        print("  NOTE: this docling build exposes no chart-extraction flag; "
              "charts will be described but their numeric series not read",
              flush=True)

    applied, disabled, unavailable = [], [], []
    for flag, value in requested.items():
        if flag not in flags:
            unavailable.append(flag)
            continue
        setattr(opts, flag, value)
        (applied if value else disabled).append(flag)

    # ── STEP 4 — figure crops ────────────────────────────────────────────
    opts.images_scale = FIGURE_RENDER_SCALE

    # ── STEP 5 — the in-pipeline vision model ────────────────────────────
    # Only configured when the cached path is off. Building these options
    # requires the API key, so an unset key must not break a run that was never
    # going to call OpenAI from inside the pipeline.
    if describe_in_docling:
        opts.picture_description_options = PictureDescriptionApiOptions(
            url="https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {_require_openai_key()}"},
            params={"model": VISION_MODEL, "max_tokens": 400,
                    "temperature": 0, "seed": 0},
            prompt=FIGURE_PROMPT,
            picture_area_threshold=FIGURE_AREA_THRESHOLD,
            timeout=120,
        )

    # ── what is actually running ─────────────────────────────────────────
    #
    # READ BACK FROM THE OBJECT, not from the dict above.
    #
    # This line used to list every flag the build ACCEPTED, whether it was set
    # True or False. It printed "do_picture_description, do_chart_extraction"
    # while both were off. That was accidentally correct while
    # do_picture_description was always True, and became a lie the moment it
    # was not — a status line that reports the opposite of the truth, in a
    # pipeline whose whole premise is that silent misreporting is the enemy.
    print(f"  enrichments ON : {', '.join(sorted(applied)) or 'none'}",
          flush=True)
    if disabled:
        print(f"  enrichments OFF: {', '.join(sorted(disabled))}", flush=True)
    if unavailable:
        # A silent rename would otherwise turn into a stage that quietly stops
        # running, which is exactly the failure this pipeline is built to
        # surface.
        print(f"  NOT AVAILABLE in this docling build: "
              f"{', '.join(sorted(unavailable))}", flush=True)

    if CACHE_FIGURE_DESCRIPTIONS:
        print("  (do_picture_description and enable_remote_services are off on "
              "purpose — describe_figures() runs after the parse, cached)",
              flush=True)

    return opts


def describe_pipeline(opts) -> None:
    """Print the settings that are actually on the built options object.

    Every value here is read back from `opts`, so it cannot disagree with what
    the converter will do. Constructing a fresh PdfPipelineOptions() to inspect
    instead shows Docling's defaults and tells you nothing — which is an easy
    mistake to make, and the reason this exists as a function rather than a
    snippet people retype.
    """
    fields = ["do_ocr", "do_table_structure", "do_picture_classification",
              "do_formula_enrichment", "do_code_enrichment",
              "generate_picture_images", "generate_table_images",
              "do_picture_description", "enable_remote_services"]
    print("  pipeline options in effect:", flush=True)
    for field in fields:
        print(f"    {field:<30}{getattr(opts, field, '(absent)')}", flush=True)
    print(f"    {'images_scale':<30}{getattr(opts, 'images_scale', '?')}",
          flush=True)
    table_opts = getattr(opts, "table_structure_options", None)
    print(f"    {'table mode':<30}{getattr(table_opts, 'mode', '?')}", flush=True)
    print(f"    {'do_cell_matching':<30}"
          f"{getattr(table_opts, 'do_cell_matching', '(absent)')}", flush=True)

    # Read from layout_options, which is where these live. Reading them off
    # `opts` reports "(absent)" for settings that are really set.
    layout = getattr(opts, "layout_options", None)
    engine = getattr(layout, "engine_options", None) if layout else None
    spec = getattr(layout, "model_spec", None) if layout else None
    print(f"    {'layout model':<30}"
          f"{getattr(spec, 'name', None) or getattr(spec, 'repo_id', '(default)')}",
          flush=True)
    print(f"    {'layout score_threshold':<30}"
          f"{getattr(engine, 'score_threshold', '(absent)')}", flush=True)
    print(f"    {'create_orphan_clusters':<30}"
          f"{getattr(layout, 'create_orphan_clusters', '(absent)')}", flush=True)
    print(f"    {'keep_empty_clusters':<30}"
          f"{getattr(layout, 'keep_empty_clusters', '(absent)')}", flush=True)


def _page_fraction(item, doc) -> float:
    """How much of its page this element covers, 0.0 to 1.0.

    Reproduces the filter Docling's own description step applies, so the two
    paths skip the same graphics. Returns 1.0 when the geometry cannot be read,
    because skipping a figure by accident is the expensive mistake.
    """
    try:
        prov = (getattr(item, "prov", None) or [None])[0]
        if prov is None:
            return 1.0
        box = prov.bbox
        area = abs(box.r - box.l) * abs(box.t - box.b)
        page = doc.pages[prov.page_no].size
        return area / (page.width * page.height)
    except Exception:
        return 1.0


def _attach_description(item, text: str) -> bool:
    """Record a description on a picture, wherever this build keeps them.

        item.meta.annotations   current
        item.annotations        deprecated, still present

    Returns False when neither exists, so the caller reports it rather than
    reporting a description that went nowhere.
    """
    from docling_core.types.doc.document import PictureDescriptionData

    annotation = PictureDescriptionData(text=text, provenance=VISION_MODEL)

    meta = getattr(item, "meta", None)
    target = getattr(meta, "annotations", None) if meta is not None else None
    if target is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            target = getattr(item, "annotations", None)
    if target is None:
        return False
    target.append(annotation)
    return True


def describe_figures(doc) -> None:
    """Attach a description to every figure, from cache where possible.

        picture  ->  already described?      ->  leave it
                 ->  no rendered image?      ->  skip, and say so
                 ->  under the area floor?   ->  skip
                 ->  sha256(png + prompt + model) in cache?  ->  read it
                 ->  otherwise                               ->  call, then write it

    WHAT THIS DOES NOT DO

        It does NOT re-render anything. The crops already exist because
        generate_picture_images ran during the parse.

        It does NOT overwrite a description Docling already produced, so
        turning CACHE_FIGURE_DESCRIPTIONS off and on again is safe.

    The cache key includes the prompt and the model, so changing either
    invalidates it without anyone having to remember to clear a directory. It
    does NOT include the heading path or the caption: those are prepended at
    record time in chunking.py, and folding them in here would mean re-calling
    the model every time a heading was corrected.
    """
    from docling_core.types.doc import PictureItem

    FIGURE_CACHE.mkdir(parents=True, exist_ok=True)
    prompt_key = hashlib.sha256(
        (FIGURE_PROMPT + "|" + VISION_MODEL).encode()).hexdigest()[:8]

    from_cache = called = skipped_small = skipped_blank = failed = 0
    client = None

    for item, _ in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        # Tested on the annotation KIND, not on picture_description(), whose
        # last-resort branch returns any annotation carrying text. A
        # classification annotation can satisfy that and make an undescribed
        # figure look described.
        if any(str(getattr(a, "kind", "")).lower() == "description"
               and getattr(a, "text", None)
               for a in picture_annotations(item)):
            continue

        image = item.get_image(doc)
        if image is None:
            # Detected by layout analysis but never rendered. Nothing to look
            # at, so nothing to describe.
            skipped_blank += 1
            continue

        if _page_fraction(item, doc) < FIGURE_AREA_THRESHOLD:
            skipped_small += 1
            continue

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw = buffer.getvalue()

        digest = hashlib.sha256(raw).hexdigest()[:20]
        cached = FIGURE_CACHE / f"{digest}.{prompt_key}.txt"

        if cached.exists():
            description = cached.read_text()
            from_cache += 1
        else:
            if client is None:
                from openai import OpenAI
                _require_openai_key()
                client = OpenAI()
            try:
                response = client.chat.completions.create(
                    model=VISION_MODEL, temperature=0, seed=0, max_tokens=400,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": FIGURE_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64,"
                                   + base64.b64encode(raw).decode(),
                            # Axis labels are small. At low detail the model
                            # cannot read them and starts inventing numbers.
                            "detail": "high"}},
                    ]}],
                )
            except Exception as exc:
                print(f"    figure description failed on p"
                      f"{getattr((getattr(item, 'prov', None) or [None])[0], 'page_no', '?')}"
                      f": {exc}", flush=True)
                failed += 1
                continue
            description = (response.choices[0].message.content or "").strip()
            if not description:
                failed += 1
                continue
            cached.write_text(description)
            called += 1

        if not _attach_description(item, description):
            print("    NOTE: this docling build exposes no annotations list on "
                  "PictureItem; descriptions could not be attached", flush=True)
            failed += 1

    total = from_cache + called
    print(f"  figures described: {total} ({from_cache} from cache, "
          f"{called} model calls)"
          + (f", {skipped_small} below the area floor" if skipped_small else "")
          + (f", {skipped_blank} with no rendered image" if skipped_blank else "")
          + (f", {failed} FAILED" if failed else ""), flush=True)
    if called and from_cache == 0 and total > 1:
        print("  NOTE: nothing came from cache. Expected on a first parse. On a "
              "repeat parse of an unchanged PDF it means the rendered bytes "
              "changed — check FIGURE_RENDER_SCALE and the layout model, "
              "because every figure chunk_id will have changed with them.",
              flush=True)


def parse_pdf(pdf: Path):
    """Parse a PDF into a DoclingDocument.

    "Parsing" is several models in sequence, not one. Knowing which is which is
    what lets you read a bad result and know where to look:

      Layout analysis        Heron by default; heron-101 and the egret family
                             are selectable through LAYOUT_MODEL. Finds the
                             regions on each page and labels them — title,
                             section header, text, list item, caption, table,
                             picture, formula — establishes reading order
                             across columns, and associates captions with the
                             figure or table they belong to. Everything else
                             keys off its output, so a mislabelled region is
                             mislabelled for the rest of the pipeline. Always
                             runs.

      TableFormer            Reconstructs the grid inside a region labelled
                             TABLE: rows, columns, spans, merged cells. This is
                             what export_to_markdown() serialises, and when it
                             collapses a stacked pair into one grid,
                             table_looks_broken() catches it. Runs when
                             do_table_structure is set; ACCURATE mode here.
                             TABLE_CELL_MATCHING controls whether its predicted
                             cells are matched against the PDF's own text cells.

      OCR engine             EasyOCR by default, with Tesseract and RapidOCR
                             pluggable. Only reads regions with no extractable
                             text layer — a scanned page, or a chart with text
                             baked into the image. Digitally generated PDFs
                             skip it entirely. Runs when do_ocr is set; on by
                             default.

      DocumentFigureClassifier
                             Tags each picture as chart, photo, logo or
                             diagram. Cheap, local, and what makes a header
                             wordmark distinguishable from an exhibit after the
                             fact. Runs when do_picture_classification is set.

      CodeFormula            Reads formula and code regions. Without it an
                             equation is emitted as the placeholder
                             `formula-not-decoded` and its content is simply
                             gone. Runs when do_formula_enrichment or
                             do_code_enrichment is set.

      Chart extraction       Torch-backed, reads numeric series out of bar and
                             line charts rather than only describing them.
                             Works from rasterised charts with detectable axes;
                             produces nothing on vector-drawn ones, which is
                             common in financial and research PDFs. Runs when
                             the chart-extraction flag is set.

      Vision model (remote)  VISION_MODEL, called over the API — the only model
                             here that is not local, and the only one that
                             costs money. Writes the description that makes a
                             chart searchable at all. Runs in describe_figures()
                             AFTER the parse, against a cache, unless
                             CACHE_FIGURE_DESCRIPTIONS is off.

    The local models are downloaded from public HuggingFace repositories on
    first use (~500 MB) and cached by huggingface_hub thereafter.

    Every call re-parses the PDF itself. There is no cache for that on purpose:
    the output depends on the settings above as much as on the file, so a cache
    keyed on the filename returns work done under different settings and makes
    a changed setting look like it did nothing. Getting that key right means
    hashing the layout model, the area threshold, the render scale and the
    enrichment flags — machinery that exists only to serve the cache, and that
    is wrong in a way nobody notices.

    The expensive part is cached anyway, one level down: describe_figures()
    keys on the rendered image bytes, which change when any setting that
    affects them changes. That gets the cost and the determinism without the
    invalidation problem.

    Two consequences still follow.

    Iterating on chunking or retrieval re-parses each time. Parse once in the
    notebook and keep `doc` in the kernel; only re-run this cell when a parse
    setting actually changes.

    On AWS, a retried task re-parses from scratch — but reads every figure
    description from the cache, so the retry is free of API cost.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    check_model_access()
    options = build_pipeline_options()
    describe_pipeline(options)
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
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

    if CACHE_FIGURE_DESCRIPTIONS:
        describe_figures(doc)

    return doc


def caption_coverage(doc) -> tuple[int, int]:
    """How many pictures the layout model linked to a caption.

        (linked, total)

    The layout model associates a CAPTION region with the figure above or below
    it. When it fails, the caption survives as a loose text element and the
    figure record loses its exhibit number — so "what does Exhibit 9 show" has
    nothing to match.

    There is no setting for this. It is layout-model quality, which makes this
    the number to watch when comparing LAYOUT_MODEL values.
    """
    from docling_core.types.doc import PictureItem

    linked = total = 0
    for item, _ in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        total += 1
        try:
            if (item.caption_text(doc) or "").strip():
                linked += 1
        except Exception:
            pass
    return linked, total


def save_figures(doc, doc_id: str, bucket: str | None) -> dict[str, str]:
    """Write each figure PNG to S3 and return a map of element ref -> URI.

    The pixels have already been rendered so the vision model could read them.
    Throwing them away means an answer can quote a description of a chart but
    never show the chart, which is usually what a person actually wants to see.

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
            # Detected by layout analysis but never rendered — nothing to
            # store, and its description (if any) already made it into the text.
            continue
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        key = f"figures/{doc_id}/fig_{n:04d}.png"
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(),
                      ContentType="image/png")
        # self_ref is Docling's stable identifier for an element. Keying on it
        # lets build_records attach this URI to whichever chunk contains the
        # figure.
        uris[getattr(item, "self_ref", None) or str(id(item))] = f"s3://{bucket}/{key}"
    return uris
