"""Every setting the pipeline reads, in one module.

Where these are used:

    config.py
        |
        |-- EMBED_MODEL, EMBED dimensions ----> ingestion AND retrieval
        |                                       both must agree, or results
        |                                       are meaningless with no error
        |
        |-- CHUNK_TOKENS ---------------------> chunking.py
        |
        |-- FIGURE_*, DO_* -------------------> docling_io.py
        |                                       what runs during a parse
        |
        |-- TABLE_* --------------------------> tables.py
        |
        |-- PINECONE_* -----------------------> index.py, sync.py
        |                                       request and metadata limits
        |
        v
    read ONCE, at import


THE IMPORT-TIME READ MATTERS
----------------------------

These values are read from the environment when this module is first
imported, and never again.

So setting an environment variable in a notebook cell AFTER importing the
package has no effect, and nothing warns you — the package simply keeps what
it read. Set them first, or restart the kernel.

Run `python check_config.py` to see what is actually in effect versus what
the environment says.


THE ONE THAT MUST MATCH ON BOTH SIDES
-------------------------------------

EMBED_MODEL.

Ingest with one model and query with another and you get results back, with
scores that look reasonable, that are meaningless. There is no error and no
other symptom.

That is what the manifest in index.py exists to prevent — it records how the
index was built and refuses to run retrieval when the settings disagree.
"""

import os
import re
from pathlib import Path

import tiktoken

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"   # what chunks and queries are embedded with


VISION_MODEL = "gpt-4o-mini"             # figure descriptions, during extraction


# Cross-encoder used to reorder candidates. Hosted by Pinecone, so no second
# provider account and no second API key.
#
# Pinecone hosts several rerankers behind one call, including Cohere's, so switching
# is a config change rather than a code change. Which is better is a question about
# your corpus, not about the models — evaluate() in notebook 2 answers it, and
# list_rerank_models() below prints what is actually available today.
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")


LLM_MODEL = "gpt-4o-mini"                # generation and query rewriting


# ─────────────────────────────────────────────────────────────────────────────
# Index
# ─────────────────────────────────────────────────────────────────────────────
INDEX_NAME = "rag-docs"


NAMESPACE = ""            # one shared namespace: cross-document search stays possible


# Fixed at index creation and not changeable afterwards without rebuilding.
# For the unit-normalised vectors these models return, dot product and cosine rank
# identically — so this costs nothing today. It is additionally required for
# sparse-dense hybrid retrieval, so choosing cosine would quietly foreclose that.
METRIC = "dotproduct"


# Stamped on every chunk and applied as a filter on every query. Written from the
# first ingestion even when there is only one group, because adding an access field
# after the fact means re-embedding the corpus.
ACCESS_GROUPS = ["public"]


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────
# Retrieval-quality target. Smaller chunks match more precisely; larger ones carry
# more of the answer in one vector.
#
# 1024 is the default because table summaries carry the semantic weight for tabular
# content, so fragments no longer have to stand alone and can stay small enough to
# keep similarity scores sharp. Above roughly 2000 tokens a single vector averages
# too many concepts together and blunts both retrieval and reranking.
#
# This is a hyperparameter, not a constant. Sweep it against the labelled set:
#   CHUNK_TOKEN_TARGET=512 python ingest.py --pdf x.pdf
CHUNK_TOKEN_TARGET = int(os.getenv("CHUNK_TOKEN_TARGET", "1024"))


# Hard sequence limits, from provider documentation. Exceeding one does not raise:
# the API accepts the input, embeds the first N tokens, and returns a valid-looking
# vector for part of a chunk. The chunk budget is capped by this for that reason.
SEQUENCE_LIMITS = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}


# ─────────────────────────────────────────────────────────────────────────────
# Documented API limits
#
# Batch sizes are derived from these rather than hard-coded, because a fixed count
# that fits comfortably at 1536 dimensions silently starts failing at 3072.
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_METADATA_BYTES = 40 * 1024      # per vector


PINECONE_REQUEST_BYTES = 2 * 1024 * 1024  # per upsert request


OPENAI_EMBED_MAX_INPUTS = 2048           # per embeddings request


OPENAI_EMBED_MAX_TOKENS = 300_000        # per embeddings request


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache")


EMBED_CACHE = CACHE_DIR / "embeddings"


MANIFEST_PATH = CACHE_DIR / "manifest.json"


CLOUD, REGION = "aws", "us-east-1"

CACHE_DIR.mkdir(exist_ok=True)
EMBED_CACHE.mkdir(exist_ok=True)

ENCODING = tiktoken.encoding_for_model(EMBED_MODEL)

CHUNK_TOKENS = min(CHUNK_TOKEN_TARGET,
                   SEQUENCE_LIMITS.get(EMBED_MODEL, CHUNK_TOKEN_TARGET))

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


# Render scale for figure crops. At 1.0 the axis labels in a chart are too small
# for the vision model to read, and it invents plausible numbers rather than
# reporting real ones. 2.0 is where labels become legible.
#
# It is also the single most expensive setting here on a figure-dense document:
# 2x means four times the pixels, rendered on CPU, for every figure.
FIGURE_RENDER_SCALE = float(os.getenv("FIGURE_RENDER_SCALE", "2.0"))

# ─────────────────────────────────────────────────────────────────────────────
# What to run, and what each costs
#
# EVERY ONE OF THESE IS A MODEL PASS, ON CPU, PER ELEMENT.
#
# A 7-page report with 17 figures is not a small document to this pipeline: it
# is 17 crops rendered at 2x, then classified, then chart-extracted, then sent
# to a vision model. Four passes over each figure.
#
# Measure before deciding. `python profile_parse.py your.pdf` times each flag
# separately on your machine with your document, and prints what each one adds.
#
#   TABLE_MODE_ACCURATE   materially better on nested headers, and several
#                         times slower than FAST. Worth it for clinical and
#                         financial tables; probably not for simple grids.
#
#   DO_CHART_EXTRACTION   reads numeric series off rasterised charts. Measured
#                         0 of 17 on vector-drawn charts — which is most
#                         financial and research PDFs. Costs a model pass per
#                         figure and returns nothing on those. Off by default
#                         for that reason.
#
#   DO_CLASSIFICATION     tags each picture chart/photo/logo. Cheap relative to
#                         the others, and what lets you tell a header wordmark
#                         from an exhibit afterwards.
#
#   DO_FORMULA / CODE     CodeFormula. Necessary if your documents contain
#                         equations — without it they become the placeholder
#                         `formula-not-decoded` and their content is gone.
#                         Pure waste if they do not.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL vs UNCONDITIONAL — the distinction that decides what to switch off
#
# Some models run once per element, whether or not that element needs them.
# Others run only where they are needed. Switching off the wrong kind saves
# almost nothing and loses content.
#
#   UNCONDITIONAL      runs on every figure, or every candidate region
#     chart extraction        17 figures = 17 model passes
#     classification          17 figures = 17 model passes
#     formula / code          every candidate region
#     rendering at 2x         four times the pixels, every figure
#
#   CONDITIONAL        runs only where there is work to do
#     OCR                     only regions with NO extractable text layer
#
# On a digital PDF, OCR is nearly free: most text is already in the layer, so
# it runs on a handful of graphic regions and skips everything else.
#
# But it is the ONLY thing that reads text baked into a graphic. An exhibit
# drawn as coloured boxes with a bulleted list inside is invisible without it —
# the caption above and the source line below survive, because those are real
# page text, and everything inside the boxes disappears.
#
# So: turn off the unconditional ones. Leave OCR on unless you have measured
# that it costs you something.
DO_OCR = os.getenv("DO_OCR", "1") == "1"

TABLE_MODE_ACCURATE = os.getenv("TABLE_MODE_ACCURATE", "1") == "1"
DO_CHART_EXTRACTION = os.getenv("DO_CHART_EXTRACTION", "0") == "1"
DO_CLASSIFICATION = os.getenv("DO_CLASSIFICATION", "1") == "1"
DO_FORMULA = os.getenv("DO_FORMULA", "1") == "1"
DO_CODE = os.getenv("DO_CODE", "1") == "1"


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


def slugify(text: str) -> str:
    """Filesystem- and identifier-safe form of arbitrary text.

    Used for document ids, cache filenames and index names, so it has to be stable:
    the same input must always produce the same slug, or cached parses stop being
    found and documents get re-ingested under a second id.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]
