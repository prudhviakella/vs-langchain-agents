"""
Extraction and chunk verification reports.

Overall verification flow:

    Parsed Docling document
            |
            v
       inspect()
            |
            |-- count extracted elements
            |-- detect silent extraction failures
            |-- print warnings
            |
            v
    write_extraction_report()
            |
            |-- write every extracted element
            |-- preserve document/page order
            |-- record extraction problems
            |-- create Markdown + JSON reports
            |
            v
        YOU READ THE REPORT
            |
            v
       only then:
       chunk + embed


WHY THIS EXISTS
---------------

Extraction failures do not necessarily raise exceptions.

For example, if a Docling enrichment feature is disabled or fails silently:

    - a figure may have no description
    - a chart may have no extracted data
    - a formula may remain as "formula-not-decoded"
    - a table may have incorrect row/column structure
    - a table may fail to serialize correctly

The important problem is that downstream processing can still continue.

The chunking stage may happily process whatever text it receives, and the
embedding stage may happily embed those chunks.

This can produce an index that looks healthy while important information
such as tables, figures, charts, or equations is missing.

Therefore, this module intentionally sits BEFORE chunking and embedding.

It gives us two levels of verification:

1. Extraction report
   ------------------
   Answers:

       "Did Docling extract the document correctly?"

   It shows every extracted element and highlights suspicious or missing
   content.

2. Chunk report
   -------------
   Answers:

       "Is the exact text that will be embedded correct?"

   It shows the final chunk text, token counts, metadata, table summaries,
   truncation, and other information relevant to retrieval quality.

This separation is important because a document can be extracted correctly
but still be chunked incorrectly.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# REPORT_DIR:
#     Directory where extraction and chunk reports will be written.
#
# VISION_MODEL:
#     Name of the vision model used for visual enrichment such as
#     figure/image descriptions.
from .config import REPORT_DIR, VISION_MODEL

# picture_description():
#     Returns the generated semantic description for a picture/figure.
#
# chart_data():
#     Returns structured chart data when available.
from .docling_io import chart_data, picture_description

# table_cells():
#     Extracts/counts cells from a Markdown table.
#
# table_looks_broken():
#     Performs structural validation on an extracted table.
from .tables import table_cells, table_looks_broken


def inspect(doc, markdown: str) -> dict:
    """Report what the parse produced and warn about silent failures.

    This function is intentionally executed BEFORE chunking and embedding.

    Its purpose is to detect extraction problems that may not raise an
    exception but can still cause information to disappear from the
    retrieval index.

    Examples:

        - Pictures exist but have no descriptions.
        - Formulas remain undecoded.
        - Tables exist but have suspicious structure.

    Returns:
        A dictionary containing basic extraction health metrics.
    """

    # Import Docling classes locally.
    #
    # Keeping these imports inside the function means the module does not
    # immediately require these classes just to be imported.
    from docling_core.types.doc import PictureItem, TableItem

    # ------------------------------------------------------------------
    # COLLECT ALL DOCUMENT ELEMENTS
    # ------------------------------------------------------------------

    # iterate_items() walks through all extracted elements in document order.
    #
    # It returns tuples containing the item and additional information.
    # We only need the item here, so the second value is ignored.
    items = [
        item
        for item, _ in doc.iterate_items()
    ]

    # Separate pictures from all other elements.
    pictures = [
        item
        for item in items
        if isinstance(item, PictureItem)
    ]

    # Separate tables from all other elements.
    tables = [
        item
        for item in items
        if isinstance(item, TableItem)
    ]

    # ------------------------------------------------------------------
    # CHECK FIGURE DESCRIPTIONS
    # ------------------------------------------------------------------

    # Count how many pictures actually have a generated description.
    #
    # We keep this separate from the total number of pictures because:
    #
    #     pictures = 10
    #     described = 0
    #
    # means that Docling found 10 figures but visual enrichment did not
    # produce descriptions for any of them.
    described = sum(
        1
        for picture in pictures
        if picture_description(picture)
    )

    # ------------------------------------------------------------------
    # CHECK CHART DATA
    # ------------------------------------------------------------------

    # Count pictures that have structured chart data.
    #
    # chart_data() returns None when no structured chart data is available.
    with_data = sum(
        1
        for picture in pictures
        if chart_data(picture) is not None
    )

    # ------------------------------------------------------------------
    # CHECK TABLE STRUCTURE
    # ------------------------------------------------------------------

    # Number of tables that appear structurally suspicious.
    suspect = 0

    for table in tables:
        try:
            # Convert the table to Markdown and inspect its structure.
            markdown_table = table.export_to_markdown(doc)

            if table_looks_broken(markdown_table):
                suspect += 1

        except Exception:
            # If table serialization itself fails, that is also a problem.
            #
            # We count it as suspicious instead of allowing the validation
            # process to fail.
            suspect += 1

    # ------------------------------------------------------------------
    # BUILD EXTRACTION REPORT
    # ------------------------------------------------------------------

    report = {
        # Total number of pages extracted.
        "pages": len(doc.pages),

        # Total number of tables.
        "tables": len(tables),

        # Number of suspicious tables.
        "tables_suspect": suspect,

        # Total number of pictures/figures.
        "pictures": len(pictures),

        # Number of pictures with generated descriptions.
        "pictures_described": described,

        # Number of pictures/charts with structured data.
        "charts_with_data": with_data,

        # Docling uses this placeholder when formula enrichment does not
        # successfully decode a formula.
        #
        # Counting the placeholder in the exported Markdown is simpler than
        # walking the document tree looking for formula nodes.
        "formulas_undecoded": markdown.count("formula-not-decoded"),
    }

    # Print the extraction metrics on one line.
    #
    # This makes the values easy to spot in terminal/CI/CD output.
    print(
        "  " + "  ".join(
            f"{key}={value}"
            for key, value in report.items()
        ),
        flush=True,
    )

    # ------------------------------------------------------------------
    # WARN ABOUT FIGURE DESCRIPTION FAILURE
    # ------------------------------------------------------------------

    # If pictures exist but none have descriptions, visual information
    # may become invisible to text-based retrieval.
    if pictures and not described:
        print(
            "  WARNING: no figure descriptions produced",
            flush=True,
        )

    # ------------------------------------------------------------------
    # WARN ABOUT UNDECODED FORMULAS
    # ------------------------------------------------------------------

    if report["formulas_undecoded"]:
        print(
            f"  WARNING: "
            f"{report['formulas_undecoded']} undecoded formulas",
            flush=True,
        )

    # ------------------------------------------------------------------
    # WARN ABOUT SUSPICIOUS TABLES
    # ------------------------------------------------------------------

    if suspect:
        print(
            f"  WARNING: {suspect} tables have suspect structure — "
            "their summaries will be generated from the rendered image",
            flush=True,
        )

    # Return the metrics so that the caller can also use them programmatically.
    return report


def _element_kind(item) -> str:
    """Return a short, human-readable label for a document element.

    Examples:

        TableItem     -> TABLE
        PictureItem   -> FIGURE
        SectionHeader -> SECTION_HEADER
        Text          -> TEXT

    The returned value is primarily used as a heading in the extraction
    report.
    """

    from docling_core.types.doc import PictureItem, TableItem

    # Tables get their own explicit category.
    if isinstance(item, TableItem):
        return "TABLE"

    # Pictures/figures get their own explicit category.
    if isinstance(item, PictureItem):
        return "FIGURE"

    # For other Docling elements, use the element's label.
    label = str(
        getattr(item, "label", "")
    ).upper()

    # Docling labels can look like:
    #
    #     DOCITEMLABEL.SECTION_HEADER
    #
    # We only want:
    #
    #     SECTION_HEADER
    return (
        label.rsplit(".", 1)[-1]
        or "TEXT"
    )


def write_extraction_report(
    doc,
    doc_id: str,
    pdf: Path,
) -> Path:
    """Write a human-readable dump of everything extraction produced.

    The extraction report answers:

        "Did the PDF parse correctly?"

    The report follows the same order as the source document and inserts
    page headings so that it can be reviewed side-by-side with the PDF.

    Two files are created:

        <doc_id>.extract.md
            Human-readable Markdown report.

        <doc_id>.extract.json
            Machine-readable report containing the extracted inventory.
    """

    from docling_core.types.doc import PictureItem, TableItem

    # ------------------------------------------------------------------
    # CREATE REPORT DIRECTORY
    # ------------------------------------------------------------------

    # Create the directory if it does not exist.
    #
    # parents=True:
    #     Also create missing parent directories.
    #
    # exist_ok=True:
    #     Do not fail if the directory already exists.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Paths for the two reports.
    md_path = REPORT_DIR / f"{doc_id}.extract.md"
    json_path = REPORT_DIR / f"{doc_id}.extract.json"

    # ------------------------------------------------------------------
    # REPORT DATA STRUCTURES
    # ------------------------------------------------------------------

    # body:
    #     Contains the human-readable Markdown body.
    body: list[str] = []

    # inventory:
    #     Contains machine-readable information about each extracted element.
    inventory: list[dict] = []

    # problems:
    #     Contains extraction problems discovered while walking the document.
    problems: list[str] = []

    # counts:
    #     Counts elements by type.
    #
    # Example:
    #
    #     TEXT: 100
    #     TABLE: 5
    #     FIGURE: 10
    counts: Counter = Counter()

    # Keeps track of the page currently being processed.
    #
    # We use this to insert a new "Page X" heading whenever the page changes.
    current_page = None

    # ------------------------------------------------------------------
    # WALK THROUGH EVERY EXTRACTED ELEMENT
    # ------------------------------------------------------------------

    for index, (item, _) in enumerate(doc.iterate_items()):

        # Convert the Docling element into a readable label.
        kind = _element_kind(item)

        # Increment the element counter.
        counts[kind] += 1

        # Get provenance information.
        #
        # Provenance tells us where the extracted element came from,
        # including the page number.
        prov = getattr(item, "prov", None) or []

        # Use the first provenance entry to determine the page.
        #
        # If no provenance exists, page remains None.
        page = (
            prov[0].page_no
            if prov
            else None
        )

        # ------------------------------------------------------------------
        # PAGE BREAK IN REPORT
        # ------------------------------------------------------------------

        # Whenever the page changes, insert a page heading.
        #
        # This makes it much easier to compare the extraction report
        # against the original PDF.
        if page != current_page:
            current_page = page
            body.append(
                f"\n## Page {page}\n"
            )

        # Basic machine-readable information for this element.
        record = {
            "index": index,
            "kind": kind,
            "page": page,
        }

        # ==============================================================
        # TABLE
        # ==============================================================

        if isinstance(item, TableItem):

            try:
                # Convert the extracted table into Markdown.
                #
                # This is the representation we can inspect visually and
                # later use as part of the chunking pipeline.
                markdown = item.export_to_markdown(doc)

            except Exception as exc:

                # If serialization fails, keep the report generation alive.
                markdown = ""

                # Record the failure so it appears in the report.
                problems.append(
                    f"p{page}: table {index} could not be "
                    f"serialised ({exc})"
                )

            # Count the cells extracted from the Markdown table.
            #
            # If serialization failed, there is nothing to count.
            cells = (
                len(table_cells(markdown))
                if markdown
                else 0
            )

            # Check whether the table structure appears suspicious.
            #
            # IMPORTANT:
            #
            # This variable is deliberately called structure_problems
            # instead of problems.
            #
            # We already have a global `problems` list that stores ALL
            # extraction problems. Reusing that name here could accidentally
            # overwrite or redirect the global problem list.
            structure_problems = (
                table_looks_broken(markdown)
                if markdown
                else []
            )

            # Add a table heading showing the number of extracted cells.
            body.append(
                f"### TABLE · {cells} cells"
            )
            body.append("")

            # If the table structure looks wrong, make that obvious.
            if structure_problems:

                # Convert the individual problems into one readable string.
                joined = "; ".join(
                    structure_problems
                )

                # Record the structural problem.
                problems.append(
                    f"p{page}: table {index} structure "
                    f"looks wrong ({joined})"
                )

                # Tell the reader what fallback strategy will be used.
                body.append(
                    f"> **STRUCTURE SUSPECT** — {joined}. "
                    "The summary will be generated from the rendered "
                    "image instead."
                )
                body.append("")

            # If table serialization worked, include the actual table.
            if markdown:
                body.append(markdown)

            else:
                # Explicitly expose missing table structure.
                #
                # Without this message, someone reading the report might
                # incorrectly assume the source PDF contained no table.
                body.append(
                    "> **MISSING** — export_to_markdown() failed. "
                    "The rows will be indexed without structure."
                )

            body.append("")

            # Add table information to the JSON inventory.
            record.update(
                cells=cells,
                chars=len(markdown),
                markdown=markdown,
                structure_problems=structure_problems,
            )

        # ==============================================================
        # FIGURE / PICTURE
        # ==============================================================

        elif isinstance(item, PictureItem):

            # Get the generated semantic description.
            description = picture_description(item)

            # Get structured chart data, if available.
            series = chart_data(item)

            # Add figure heading.
            body.append("### FIGURE")
            body.append("")

            # ----------------------------------------------------------
            # FIGURE DESCRIPTION
            # ----------------------------------------------------------

            if description:

                # Include the generated figure description.
                #
                # This description may become the primary text
                # representation of the image for semantic retrieval.
                body.append(description)

            else:

                # Record the missing description.
                problems.append(
                    f"p{page}: figure {index} has no description"
                )

                # Make the problem visible in the Markdown report.
                body.append(
                    "> **MISSING** — no description was produced. "
                    "This figure is invisible to every query."
                )

            # ----------------------------------------------------------
            # CHART DATA
            # ----------------------------------------------------------

            if series is not None:

                body.append("")

                # Include only the first 2000 characters.
                #
                # This prevents extremely large chart-data structures
                # from making the report unnecessarily huge.
                body.append(
                    f"```\n"
                    f"chart data: {str(series)[:2000]}\n"
                    f"```"
                )

            body.append("")

            # Add figure information to the JSON inventory.
            record.update(
                description=description,
                has_chart_data=series is not None,
            )

        # ==============================================================
        # OTHER DOCUMENT ELEMENTS
        # ==============================================================

        else:

            # Most text-based Docling elements expose their content
            # through a `text` attribute.
            #
            # getattr() prevents an AttributeError if a particular
            # element type does not have a text property.
            text = getattr(
                item,
                "text",
                "",
            ) or ""

            # Ignore empty or whitespace-only elements.
            if not text.strip():
                continue

            # ----------------------------------------------------------
            # FORMULA CHECK
            # ----------------------------------------------------------

            # Detect Docling's formula placeholder.
            #
            # This means the original mathematical expression was not
            # successfully decoded.
            if "formula-not-decoded" in text:
                problems.append(
                    f"p{page}: formula {index} not decoded"
                )

            # Add element heading.
            body.append(
                f"### {kind}"
            )
            body.append("")

            # Add the actual extracted text.
            body.append(text)
            body.append("")

            # Store text information in the JSON inventory.
            record.update(
                text=text,
                chars=len(text),
            )

        # Add the element to the inventory.
        inventory.append(record)

    # ------------------------------------------------------------------
    # BUILD REPORT HEADER
    # ------------------------------------------------------------------

    # The header summarizes the document and extraction problems.
    #
    # This section appears before the full element dump.
    header = [
        f"# {doc_id}",
        "",

        # Source PDF filename.
        f"- source: `{pdf.name}`",

        # UTC timestamp.
        #
        # Using UTC avoids confusion when reports are generated on
        # different machines/time zones.
        f"- extracted: "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",

        # Vision model used for visual enrichment.
        f"- vision model: `{VISION_MODEL}`",

        # Total page count.
        f"- pages: {len(doc.pages)}",

        "",
        "## Extraction problems",
        "",

        # Show every extraction problem.
        #
        # If none were found, explicitly say so.
        *(
            [
                f"- {problem}"
                for problem in problems
            ]
            or ["- none detected"]
        ),

        "",
        "## Element counts",
        "",

        # Show element counts from most common to least common.
        *[
            f"- {kind}: {count}"
            for kind, count in counts.most_common()
        ],

        "",
        "---",
    ]

    # ------------------------------------------------------------------
    # WRITE MARKDOWN REPORT
    # ------------------------------------------------------------------

    # Combine the header and body and write them to disk.
    md_path.write_text(
        "\n".join(
            header + body
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # WRITE JSON REPORT
    # ------------------------------------------------------------------

    # Create the machine-readable version of the extraction report.
    #
    # default=str ensures objects that are not natively JSON serializable
    # do not cause the entire report generation to fail.
    json_path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "source": pdf.name,
                "pages": len(doc.pages),
                "vision_model": VISION_MODEL,
                "counts": dict(counts),
                "problems": problems,
                "elements": inventory,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Print the generated report path.
    print(
        f"  extraction report: {md_path}",
        flush=True,
    )

    # If problems were found, print the count.
    if problems:
        print(
            f"  {len(problems)} extraction problems — "
            "see the report",
            flush=True,
        )

    # Return the Markdown path.
    #
    # This is useful because the Markdown report is the primary
    # human-readable artifact.
    return md_path


def write_chunk_report(
    records: list[dict],
    doc_id: str,
) -> Path:
    """Write every final chunk exactly as it will be embedded.

    The extraction report answers:

        "Did the PDF parse correctly?"

    The chunk report answers:

        "Is the final text going into the embedding model correct?"

    This distinction is important because a document may be extracted
    correctly but then become problematic during chunking.

    The chunk report helps detect:

        - bad chunk boundaries
        - truncation
        - incorrect metadata
        - missing heading context
        - table-summary problems
        - unexpected content types

    Two files are created:

        <doc_id>.chunks.md
            Human-readable Markdown chunk report.

        <doc_id>.chunks.json
            Machine-readable chunk metadata.
    """

    # ------------------------------------------------------------------
    # CREATE REPORT DIRECTORY
    # ------------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Output paths.
    md_path = REPORT_DIR / f"{doc_id}.chunks.md"
    json_path = REPORT_DIR / f"{doc_id}.chunks.json"

    # ------------------------------------------------------------------
    # COUNT CHUNKS BY CONTENT TYPE
    # ------------------------------------------------------------------

    # Example output:
    #
    #     text: 120
    #     table_summary: 8
    #     table_fragment: 25
    #
    # This provides a quick overview of what is being sent toward
    # the embedding stage.
    counts = Counter(
        record["meta"]["content_type"]
        for record in records
    )

    # ------------------------------------------------------------------
    # COUNT TRUNCATED CHUNKS
    # ------------------------------------------------------------------

    # A truncated chunk means that some content was dropped because
    # the original element could not be reduced enough to fit within
    # the configured token budget.
    #
    # Calculate this BEFORE building the f-string list below.
    #
    # This is intentionally separated from the f-string because nested
    # expressions make the code harder to read and can cause syntax
    # problems depending on how quotes are nested.
    truncated_count = sum(
        1
        for record in records
        if record["meta"].get("truncated")
    )

    # ------------------------------------------------------------------
    # COUNT TABLE SUMMARIES GENERATED FROM IMAGES
    # ------------------------------------------------------------------

    # Some tables may have structurally incorrect parsed grids.
    #
    # For those tables, the pipeline can generate a semantic summary
    # from the rendered table image instead.
    #
    # Count how many records used that fallback.
    table_image_count = sum(
        1
        for record in records
        if record["meta"].get("summary_source") == "image"
    )

    # ------------------------------------------------------------------
    # BUILD MARKDOWN REPORT HEADER
    # ------------------------------------------------------------------

    # IMPORTANT:
    #
    # Keep the calculations above separate from this list.
    #
    # This avoids complicated nested expressions such as:
    #
    #     f"{sum(1 for record in records ...)}"
    #
    # and makes the report-building code easier to understand.
    lines = [
        f"# {doc_id} — chunks",
        "",

        # Total number of final chunk records.
        f"- total: {len(records)}",

        # Breakdown by content type.
        *[
            f"- {kind}: {count}"
            for kind, count in counts.most_common()
        ],

        # Number of chunks that were truncated.
        f"- truncated: {truncated_count}",

        # Number of chunks whose table summary was generated
        # from the rendered table image.
        f"- from table image: {table_image_count}",

        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # WRITE EVERY CHUNK
    # ------------------------------------------------------------------

    # Each record is shown exactly as it will be embedded.
    #
    # This is the most important purpose of this report:
    #
    #     extraction report
    #         ->
    #     What Docling extracted
    #
    #     chunk report
    #         ->
    #     What the embedding model receives
    #
    for record in records:

        # Metadata is stored separately from the actual chunk text.
        meta = record["meta"]

        # --------------------------------------------------------------
        # CHUNK HEADING
        # --------------------------------------------------------------

        # Show:
        #
        #     position
        #     content type
        #     page range
        #     token count
        #
        lines.append(
            f"## [{meta['position']:>3}] "
            f"{meta['content_type']}"
            f" · p{meta['page']}-{meta['page_end']}"
            f" · {meta['n_tokens']} tokens"
        )

        lines.append("")

        # --------------------------------------------------------------
        # CHUNK ID
        # --------------------------------------------------------------

        # Display the unique chunk identifier.
        lines.append(
            f"`{meta['chunk_id']}`"
        )

        # --------------------------------------------------------------
        # HEADING CONTEXT
        # --------------------------------------------------------------

        # If heading information is available, display it.
        #
        # Example:
        #
        #     headings: Introduction > Architecture > Retrieval
        #
        # This helps verify whether the chunk has the correct document
        # context before it is embedded.
        if meta["headings"]:
            lines.append(
                f"headings: {' > '.join(meta['headings'])}"
            )

        # --------------------------------------------------------------
        # TABLE ID
        # --------------------------------------------------------------

        # Show the table ID when this chunk belongs to a table.
        if meta["table_id"]:
            lines.append(
                f"table_id: `{meta['table_id']}`"
            )

        # --------------------------------------------------------------
        # TABLE IMAGE FALLBACK
        # --------------------------------------------------------------

        # Explicitly mark table summaries that were generated from
        # a rendered image instead of the parsed table grid.
        if meta.get("summary_source") == "image":
            lines.append(
                "> generated from the rendered table image, "
                "because the parsed grid was structurally unsound"
            )

        # --------------------------------------------------------------
        # IMAGE URI
        # --------------------------------------------------------------

        # If an image URI is associated with the chunk, show it.
        if meta.get("image_uri"):
            lines.append(
                f"image: {meta['image_uri']}"
            )

        # --------------------------------------------------------------
        # TRUNCATION WARNING
        # --------------------------------------------------------------

        # Explicitly flag chunks where content was dropped.
        if meta.get("truncated"):
            lines.append(
                "> **TRUNCATED** — this element could not be split "
                "below the token budget and its tail was dropped."
            )

        lines.append("")

        # --------------------------------------------------------------
        # ACTUAL EMBEDDING TEXT
        # --------------------------------------------------------------

        # IMPORTANT:
        #
        # Use record["text"] here rather than another copy of the text
        # stored inside metadata.
        #
        # record["text"] should represent the exact string that will
        # eventually be passed to the embedding model.
        lines.append("```")
        lines.append(record["text"])
        lines.append("```")
        lines.append("")

    # ------------------------------------------------------------------
    # WRITE MARKDOWN CHUNK REPORT
    # ------------------------------------------------------------------

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # WRITE JSON CHUNK REPORT
    # ------------------------------------------------------------------

    # The JSON report contains metadata only.
    #
    # The actual chunk text is excluded because the Markdown report
    # already contains the complete text.
    #
    # This keeps the JSON focused on chunk metadata and easier to
    # consume programmatically.
    json_path.write_text(
        json.dumps(
            [
                {
                    key: value
                    for key, value in record["meta"].items()
                    if key != "text"
                }
                for record in records
            ],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Print the generated report path.
    print(
        f"  chunk report: {md_path}",
        flush=True,
    )

    # Return the Markdown report path.
    return md_path