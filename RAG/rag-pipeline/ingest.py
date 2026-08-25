"""Ingest one PDF into the vector index.

This is the only file that touches documents. It runs identically on a laptop and
inside a Fargate task; the differences are where the PDF comes from and whether
audit records are written to DynamoDB.

    python ingest.py --pdf report.pdf
    python ingest.py --bucket my-bucket --key raw/small/report.pdf

Five stages:

    parse     PDF  -> DoclingDocument   layout ML, table structure, figure captions
    inspect   verify the parse produced what it claims, and write a readable report
    figures   store rendered figure images, if a bucket is configured
    chunk     DoclingDocument -> records, each carrying its text and metadata
    index     records -> Pinecone, as a three-way diff against what is already there

There is deliberately no distributed code here: no queue polling, no locking, no
heartbeats. Parallelism lives entirely in the AWS state machine, which runs many
copies of this script at once.

The organising idea is that extraction fails silently. A disabled enrichment yields
a placeholder, a mis-parsed table yields a plausible-looking grid, and every later
stage runs happily on top of either — producing an index that looks complete and
answers questions wrongly. Most of what follows exists to make those failures
visible before any money is spent on embeddings.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rag_common import (
    ACCESS_GROUPS, CACHE_DIR, CHUNK_TOKENS, EMBED_DIMS, EMBED_MODEL, ENCODING,
    NAMESPACE, PINECONE_METADATA_BYTES, PINECONE_REQUEST_BYTES, VISION_MODEL,
    embed_stream, open_index, scan_document, slugify, write_manifest,
)

# ═════════════════════════════════════════════════════════════════════════════
# Settings
# ═════════════════════════════════════════════════════════════════════════════

# A chart is pixels. An embedding model cannot read pixels, so a figure that is not
# described in words is invisible to every query.
#
# The prompt is long on purpose. A description is only useful if it contains the
# words a person would actually search for. "Shows an upward trend" is fluent and
# useless; "revenue rose from $100M in Q1 to $140M in Q4" is findable. The closing
# clause exists because vision models otherwise editorialise about significance.
FIGURE_PROMPT = (
    "Describe this figure for a search index. State the chart type, what each axis "
    "measures and its units, the time period, the series or categories shown, and "
    "the main finding including the specific numbers and labels visible in the "
    "image. Report only what is shown; do not infer or interpret."
)

# Fraction of the page a graphic must cover before it is worth a vision call.
#
# Set low deliberately. Area is a proxy for "is this decoration", and a bad one for
# a common exhibit shape: a classification band or a horizontal process diagram is
# wide and only a centimetre tall, so it sits under 2% of the page while being real
# content. Filtering on area loses exactly those.
#
# 0.01 still excludes rules and bullet glyphs. It does let header logos through, at
# roughly $0.0004 each — under a dollar across a 2,800-page corpus, against losing
# every band and flow diagram in it. Picture classification tags logos, so they stay
# identifiable if you want to filter them properly later.
FIGURE_AREA_THRESHOLD = float(os.getenv("FIGURE_AREA_THRESHOLD", "0.01"))

# Render scale for figure crops. At 1.0 the axis labels in a chart are too small for
# the vision model to read, and it invents plausible numbers rather than reporting
# real ones. 2.0 is where labels become legible without the payload becoming absurd.
FIGURE_RENDER_SCALE = 2.0

# The summary exists to state what a reader sees in a table but that appears in no
# single cell. A schedule listing weeks 0, 4, 8 and 12 never contains the phrase
# "every 4 weeks", yet that is what someone will ask for.
#
# The constraint is therefore not "do not infer" — that would forbid the one thing
# worth paying for. It is that every statement must be checkable against the cells.
# Computing an interval, a total, a range or a percentage change is reading the
# table. Claiming a result is important or expected is not.
TABLE_PROMPT = (
    "Summarise this table so it can be found by a natural-language search.\n\n"
    "Cover, in prose:\n"
    "1. What the table reports, and what one row represents.\n"
    "2. What each column measures, with its units.\n"
    "3. Patterns that hold across rows or columns but appear in no single cell: "
    "intervals and cadence stated in words such as 'every 4 weeks' or 'at each "
    "quarter end', totals, ranges, counts, and percentage or absolute change from "
    "first to last.\n"
    "4. Specific values worth naming: the largest and smallest, and any row or "
    "column that breaks the pattern the others follow.\n"
    "5. The words someone would type when looking for this table.\n\n"
    "Every statement must be verifiable by reading the cells. Computing an "
    "interval, a total, a range or a change is reading the table and is wanted. "
    "Claiming that a result is significant, expected, encouraging, or reflects some "
    "cause is not supported by the table; do not write it."
)

# Four cells is the smallest grid that can carry a relationship: two columns, two
# rows. Below that, what Docling labelled a table is an author panel, a running
# header or a date line — not data, and summarising it adds noise to the index.
#
# This is not a judgement about which tables deserve a summary. Every real table
# gets one: a rule deciding otherwise can only fail by silently skipping a table
# that needed it, and a summary costs about $0.0015 and is cached by content.
TABLE_MIN_CELLS = 4

# Model used for table summaries. A text-only call, because TableFormer has already
# recovered the structure — roughly $0.0015 against $0.004 for the vision path.
TABLE_MODEL = os.getenv("TABLE_MODEL", "gpt-4o-mini")

# Where the human-readable reports go.
REPORT_DIR = Path(os.getenv("REPORT_DIR", "reports"))


# ═════════════════════════════════════════════════════════════════════════════
# Audit trail
#
# Two record shapes in one DynamoDB table:
#
#   pk = DOC#<doc_id>   sk = RUN#<run_id>                  one per run
#   pk = DOC#<doc_id>   sk = RUN#<run_id>#STAGE#<stage>    one per stage
#
# The stage records are what make a killed container debuggable. A run record alone
# says the document failed; the stage records say it failed during `parse`, forty
# minutes in, which points at memory rather than at credentials.
# ═════════════════════════════════════════════════════════════════════════════

class Audit:
    """Writes stage records to DynamoDB. A no-op when running locally.

    Making this a no-op rather than a separate code path means the notebook and the
    Fargate task execute the same lines. There is no "local mode" behaving
    differently, and therefore no local-only bug.
    """

    def __init__(self, table_name: str | None, doc_id: str, run_id: str):
        self.table = None
        self.doc_id, self.run_id = doc_id, run_id
        if table_name:
            # Imported lazily so the local path never needs boto3 installed.
            import boto3
            self.table = boto3.resource("dynamodb").Table(table_name)

    def _put(self, sk: str, **fields) -> None:
        """Write one record. Silently does nothing when no table is configured."""
        if self.table is None:
            return
        self.table.put_item(Item={
            "pk": f"DOC#{self.doc_id}", "sk": sk,
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # DynamoDB rejects None, so empty fields are dropped rather than nulled.
            **{k: v for k, v in fields.items() if v is not None},
        })

    def stage(self, name: str, status: str, **fields) -> None:
        """Record a stage transition and echo it to the log."""
        self._put(f"RUN#{self.run_id}#STAGE#{name}", stage=name, status=status, **fields)
        # flush on every print: CloudWatch buffers stdout, and an unflushed buffer is
        # lost when a container is killed — exactly when the logs matter most.
        print(f"  [{status:9s}] {name}", flush=True)

    def run(self, status: str, **fields) -> None:
        """Record a run-level transition: STARTED, COMPLETED or FAILED."""
        self._put(f"RUN#{self.run_id}", status=status, **fields)


class Stage:
    """Times a stage and records start, success or failure.

        with Stage(audit, "parse"):
            doc = parse_pdf(...)

    Writing the STARTED record before the work begins is the point. A stage with a
    start and no end is a container that died mid-stage, which no after-the-fact
    logging can tell you.
    """

    def __init__(self, audit: Audit, name: str):
        self.audit, self.name = audit, name

    def __enter__(self) -> "Stage":
        """Start the timer and write the STARTED record before any work runs."""
        self.started = time.time()
        self.audit.stage(self.name, "STARTED")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Record the outcome, then re-raise so the caller still sees the error."""
        elapsed = round(time.time() - self.started, 1)
        if exc_type is None:
            self.audit.stage(self.name, "OK", duration_s=str(elapsed))
        else:
            self.audit.stage(self.name, "FAILED", duration_s=str(elapsed),
                             error=str(exc)[:400])
        # False re-raises. The audit record is a side effect, not a handler: main()
        # still needs the exception so the task exits non-zero.
        return False


# ═════════════════════════════════════════════════════════════════════════════
# Docling annotation access
#
# docling-core moved picture annotations from `item.annotations` to `item.meta`.
# The old attribute still resolves and still warns, and when it is removed a naive
# read returns an empty list rather than raising — so figure descriptions would
# silently vanish and `pictures_described` would report zero with no error.
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# Table helpers
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — Extraction
# ═════════════════════════════════════════════════════════════════════════════

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
    opts.table_structure_options.mode = TableFormerMode.ACCURATE

    # Each flag switches on a model. See parse_pdf's docstring for what each one
    # does and where its failures show up.
    requested = {
        # CodeFormula. Equations become LaTeX rather than `formula-not-decoded`.
        "do_formula_enrichment": True,
        # CodeFormula again — same model reads code blocks and detects the language.
        "do_code_enrichment": True,
        # DocumentFigureClassifier. Tags each picture chart / photo / logo / diagram.
        "do_picture_classification": True,
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
            requested[alias] = True
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
    print(f"  parsed in {time.time() - started:.1f}s", flush=True)
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


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Inspection
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# Reports
#
# The parse cache is a JSON blob full of base64 images — machine-readable and
# useless to a person. But extraction is where things go wrong silently, and
# `pages 7` tells you the file opened, not whether the content is good.
#
# So two readable files are written per document, the first before a single chunk
# exists and before any embedding is bought:
#
#   <doc>.extract.md    every element in reading order, with page numbers, table
#                       markdown and figure descriptions inline
#   <doc>.extract.json  the same inventory as data, for scripted checks
#   <doc>.chunks.md     every chunk as it will be embedded
#   <doc>.chunks.json   chunk metadata without the text
#
# Failures are marked inline with a banner and collected at the top, because an
# absence is exactly what you fail to notice.
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — Chunking
# ═════════════════════════════════════════════════════════════════════════════

def content_type(chunk) -> str:
    """Coarse type, for metadata filtering and per-type evaluation.

    Recording this is what lets the evaluation harness report recall separately for
    tables, figures and prose. Without it, a pipeline that retrieves prose well and
    tables badly shows up as one mediocre average and you cannot tell which half to
    fix.
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


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4 — Indexing
# ═════════════════════════════════════════════════════════════════════════════

def sync(index, doc_id: str, records: list[dict], window: int = 512) -> dict:
    """Reconcile the index with a freshly parsed document.

    Because chunk ids are derived from chunk text, re-ingestion is a set difference:

        present in both   unchanged — the vector is still correct, do nothing
        new only          embed and upsert
        indexed only      content removed from the document, delete

    Editing one section of a 200-page report typically means a handful of added and
    removed chunks against nearly two hundred untouched ones, so embedding cost is
    proportional to what changed rather than to document size.

    A fourth case matters as much: a chunk whose text is unchanged but whose
    position moved, because a paragraph was inserted above it. Its vector is still
    correct, so it needs only a metadata rewrite, which Pinecone does without
    re-sending the vector and therefore without an embedding call.

    This is also what makes retries safe. A duplicate run upserts identical vectors
    and deletes nothing, so the state machine can retry a failed task with no
    cleanup step and no risk of corrupting the index.
    """
    # Only the fields the comparison reads. Fetching full metadata and vectors for
    # every chunk is what makes a naive sync transfer megabytes per document.
    indexed = scan_document(index, doc_id)
    incoming = {r["meta"]["chunk_id"]: r for r in records}

    added = [r for cid, r in incoming.items() if cid not in indexed]
    removed = [cid for cid in indexed if cid not in incoming]
    shifted = [cid for cid in incoming if cid in indexed and (
        int(indexed[cid].get("position", -1)) != incoming[cid]["meta"]["position"]
        or int(indexed[cid].get("page", -1)) != incoming[cid]["meta"]["page"])]

    plan = {"indexed": len(indexed), "incoming": len(incoming), "added": len(added),
            "removed": len(removed), "unchanged": len(incoming) - len(added),
            "metadata_only": len(shifted)}
    print(f"  {plan}", flush=True)

    if added:
        # Batch size from measured payload against Pinecone's 2 MB request limit
        # rather than a fixed count: at 1536 dimensions a fixed 100 fits
        # comfortably, and at 3072 with large metadata the same 100 starts failing.
        per_vector = len(json.dumps(added[0]["meta"]).encode()) + EMBED_DIMS * 4 + 128
        size = max(1, min(1000, int(PINECONE_REQUEST_BYTES * 0.8 // per_vector)))

        # embed_stream yields windows rather than returning every vector at once, so
        # peak memory stays flat regardless of how many chunks the document made.
        for offset, vectors in embed_stream([r["text"] for r in added], batch=window):
            block = added[offset:offset + len(vectors)]
            for i in range(0, len(block), size):
                index.upsert(
                    vectors=[{"id": r["meta"]["chunk_id"], "values": v.tolist(),
                              "metadata": r["meta"]}
                             for r, v in zip(block[i:i + size], vectors[i:i + size])],
                    namespace=NAMESPACE,
                )
            print(f"    upserted {min(offset + len(vectors), len(added))}"
                  f"/{len(added)}", flush=True)

    # Metadata-only rewrites: no vector sent, no embedding call.
    for cid in shifted:
        index.update(id=cid, set_metadata=incoming[cid]["meta"], namespace=NAMESPACE)

    # Deletes last. Upserting the new version before removing the old means there is
    # never a window where the document is missing from the index.
    for i in range(0, len(removed), 200):
        index.delete(ids=removed[i:i + 200], namespace=NAMESPACE)

    return plan


# ═════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    """Run all five stages against one PDF.

    Returns 0 or 1 rather than raising, because the exit code is what the ECS task
    reports and what the state machine reads to decide success or failure.
    """
    ap = argparse.ArgumentParser(description="Ingest one PDF into the vector index")
    ap.add_argument("--pdf", help="local PDF path")
    ap.add_argument("--bucket", help="S3 bucket holding the PDF")
    ap.add_argument("--key", help="S3 key of the PDF")
    ap.add_argument("--audit-table", default=os.getenv("AUDIT_TABLE"))
    args = ap.parse_args()

    # Two input modes, one code path afterwards. Environment defaults mean the
    # Fargate task definition supplies cache and audit settings without the state
    # machine having to pass them per document.
    if args.bucket and args.key:
        import boto3
        pdf = Path("/tmp") / Path(args.key).name
        boto3.client("s3").download_file(args.bucket, args.key, str(pdf))
    elif args.pdf:
        pdf = Path(args.pdf)
    else:
        ap.error("provide --pdf, or --bucket and --key")

    # doc_id comes from the filename, so it is stable across re-ingestions of the
    # same document and identical whether run locally or from S3.
    doc_id = slugify(pdf.stem)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit = Audit(args.audit_table, doc_id, run_id)

    print(f"\n{doc_id}  run {run_id}", flush=True)
    started = time.time()
    audit.run("STARTED", source=pdf.name)

    try:
        with Stage(audit, "parse"):
            doc = parse_pdf(pdf)
            # Rendering the whole document to markdown is expensive on long PDFs.
            # Do it once here and reuse it for the checks and the date.
            markdown = doc.export_to_markdown()

        with Stage(audit, "inspect"):
            report = inspect(doc, markdown)
            # Written before anything is chunked or embedded, so a bad parse can be
            # caught by reading rather than by wondering why answers are wrong.
            write_extraction_report(doc, doc_id, pdf)

        with Stage(audit, "figures"):
            figure_uris = save_figures(doc, doc_id, args.bucket)
            print(f"  {len(figure_uris)} figure images stored", flush=True)

        with Stage(audit, "chunk"):
            records = build_records(doc, pdf, doc_id,
                                    document_date(pdf, markdown[:4000]), figure_uris)
            write_chunk_report(records, doc_id)

        with Stage(audit, "index"):
            index = open_index(create=True)
            plan = sync(index, doc_id, records)

        elapsed = round(time.time() - started, 1)
        # Numbers written as strings because DynamoDB's Python resource layer
        # rejects floats, and this keeps the audit records uniformly typed.
        audit.run("COMPLETED", duration_s=str(elapsed), pages=str(report["pages"]),
                  chunks=str(len(records)), added=str(plan["added"]),
                  removed=str(plan["removed"]))
        write_manifest(doc_id, len(records), extra={"source": pdf.name})
        print(f"done in {elapsed}s\n", flush=True)
        return 0

    except Exception as exc:
        # One broad handler: the Stage context manager has already recorded which
        # stage failed and why, so this only closes out the run record and signals
        # failure to the orchestrator.
        audit.run("FAILED", duration_s=str(round(time.time() - started, 1)),
                  error=str(exc)[:400])
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
