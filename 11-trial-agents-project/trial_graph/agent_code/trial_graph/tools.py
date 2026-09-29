"""trial_graph's result summarization.

This module controls the ONLY view of a Neo4j query result that is shown
back to the LLM.

The complete query result is captured server-side, but the LLM receives
only a compact summary.

Data flow:

    execute_cypher
         │
         │ Full Neo4j result
         ▼
    captured
         │
         │ summarize()
         ▼
    counts + shape + limited samples
         │
         ▼
       LLM


IMPORTANT DESIGN PRINCIPLE
--------------------------

The LLM should NOT receive the complete result set.

For example, Neo4j could return:

    5,000 rows
    30 columns

Sending everything to the model would:

    - consume a large number of tokens
    - increase cost
    - increase latency
    - make prompt injection through retrieved data easier
    - encourage the model to retype large amounts of data
    - make follow-up reasoning less reliable


Instead, the model receives something like:

    table: 50 rows x 5 columns
    columns: trial_id, phase, status, sponsor
    <untrusted_data>
      trial_id=NCT123, phase=Phase 3, status=Completed
      trial_id=NCT456, phase=Phase 2, status=Recruiting
    </untrusted_data>
    ... and 48 more rows (captured, not shown).


The actual complete result remains in application state.


NO TOOL DEFINITIONS
-------------------

This module does not define:

    - MCP tools
    - Neo4j connections
    - AgentCore Gateway
    - LangGraph nodes

The actual tools live in:

    lambda_tools/handler.py

The result comes back through MCP and is eventually processed by
CypherMiddleware in core.py.

Keeping summarize() as a pure function makes it easy to unit test
without starting MCP, Gateway, Neo4j, or LangGraph.


WHY THE GUARDS EXIST
--------------------

Each guard addresses a concrete failure mode.


1. IDENTIFIER TRUNCATION

A long identifier could be silently shortened.

Example:

    original:
        abcdefghijklmnopqrstuvwxyz1234567890...

    displayed:
        abcdefghijklmnopqrstuvwxyz123456...

If the model copies the shortened identifier into a follow-up query,
the query may return zero rows.

The model could then incorrectly conclude that the entity does not
exist.

Therefore ID-like columns receive a much larger character limit and,
when truncated, the model is explicitly told not to reuse the value.


2. COLUMN CAP MISMATCH

The displayed sample and the "columns:" line must describe the same
visible columns.

Otherwise the model may see:

    columns:
        trial_id, phase, status, sponsor, country

but the sample only shows:

    trial_id, phase, status

The model could incorrectly conclude that sponsor/country were not
returned.

Both the column list and sample rows therefore use _COL_CAP.


3. ZERO RELATIONSHIPS

A list of nodes is NOT a graph network.

For example:

    nodes = [Trial A, Sponsor B, Drug C]
    relationships = []

does not establish:

    Trial A --SPONSORED_BY--> Sponsor B

The summary explicitly tells the model that no connections were
returned.

This prevents the model from inventing relationships simply because
multiple nodes appear in the result.


4. UNTRUSTED DATA

The values returned from the graph can ultimately originate from
external clinical-trial documents.

Those values must therefore be treated as DATA, not instructions.

The sample is fenced with:

    <untrusted_data>
    ...
    </untrusted_data>

Newlines are flattened and literal fence tags inside values are removed
so a retrieved value cannot easily escape the intended data boundary.
"""

from __future__ import annotations


# Maximum number of sample rows shown to the LLM.
#
# The complete result remains captured server-side.
#
# We intentionally expose only a very small sample to:
#
#     - reduce token usage
#     - reduce latency
#     - reduce the amount of untrusted data reaching the model
#     - prevent the model from attempting to reproduce the full result
_SAMPLE_ROWS = 3


# Maximum number of columns displayed in the sample.
#
# If the query returns more columns, the remaining columns are explicitly
# mentioned rather than silently disappearing.
_COL_CAP = 12


# Character limits for values displayed to the LLM.
#
# IDs get a much larger limit because truncating an identifier can make
# it unusable for a follow-up query.
#
# Normal values are intentionally shorter to keep the model context small.
_ID_CHARS, _VALUE_CHARS = 200, 40


def _clean(value) -> str:
    """Convert a database value into safe, compact display text.

    This function is used only for the small sample that is exposed to
    the LLM.

    Three transformations are performed:

        1. Convert the value to a string.
        2. Flatten newlines/carriage returns.
        3. Remove literal untrusted-data fence tags.

    Finally, repeated whitespace is collapsed.

    Example:

        "Clinical\ntrial <untrusted_data> test"

    becomes approximately:

        "Clinical trial test"

    This prevents a database value from creating unexpected formatting
    or attempting to manipulate the surrounding untrusted-data boundary.
    """

    # Convert any Python/database value into a string.
    text = str(value)

    # Flatten newlines.
    #
    # Without this, one database field could create multiple apparent
    # lines in the model's context and make the sample harder to parse.
    text = text.replace("\n", " ")

    # Flatten carriage returns as well.
    text = text.replace("\r", " ")

    # Remove literal opening/closing untrusted-data markers from the
    # database value.
    #
    # This prevents a value from pretending to close or reopen the
    # surrounding data fence.
    text = text.replace(
        "<untrusted_data>",
        ""
    )

    text = text.replace(
        "</untrusted_data>",
        ""
    )

    # Collapse repeated whitespace and trim leading/trailing whitespace.
    return " ".join(
        text.split()
    )


def _is_id(column: str) -> bool:
    """Return True when a column looks like an identifier.

    Identifier-like columns receive the larger _ID_CHARS limit.

    This is important because truncating an identifier can make it
    unusable for a follow-up query.

    Examples treated as IDs:

        trialId
        patientId
        documentId
        nctid
        docid
        chunkid
        sectionkey
        key
    """

    # Normalize the column name so matching is case-insensitive.
    c = column.lower()

    # Generic rule:
    #
    # Any column ending in "id" is considered identifier-like.
    #
    # Example:
    #
    #     trialId -> trialid
    #     sponsorID -> sponsorid
    #
    # Also handle known identifier names that don't end with "id".
    return (
        c.endswith("id")
        or c in (
            "nctid",
            "docid",
            "chunkid",
            "sectionkey",
            "key",
        )
    )


def summarize(captured: dict, cfg) -> str:
    """Create the compact model-facing representation of a query result.

    `captured` contains the complete server-side query result.

    This function deliberately exposes only:

        - result shape
        - counts
        - limited metadata
        - at most three sample rows

    The complete result is NOT returned to the model.

    The result shape determines which specialized formatter is used:

        graph -> _graph()
        table -> _table()
        anything else -> empty/unmatched result message
    """

    # Determine the result shape captured by the execution layer.
    #
    # If no shape is present, treat the result as empty rather than
    # assuming that a query returned data.
    shape = captured.get(
        "result_shape",
        "empty"
    )

    # Graph result:
    #
    # Neo4j returned nodes and/or relationships.
    if shape == "graph":
        return _graph(
            captured,
            cfg
        )

    # Table result:
    #
    # Neo4j returned rows/columns.
    if shape == "table":
        return _table(
            captured,
            cfg
        )

    # Anything other than graph/table is represented as an empty result
    # from the model's perspective.
    #
    # The wording intentionally tells the model NOT to invent data and
    # NOT to loosen filters unless the original question allows it.
    return (
        "empty: the query ran and matched nothing. This may well be the "
        "correct answer — do not invent data, and do not loosen the "
        "filters unless the question allows it."
    )


def _graph(captured: dict, cfg) -> str:
    """Create the compact model-facing representation of a graph result.

    A graph result contains:

        nodes
        relationships

    We expose:

        - total node count
        - total relationship count
        - up to eight distinct labels
        - explicit warning when there are zero relationships
        - truncation status

    Notice that actual node properties and relationship properties are
    NOT dumped into the model context here.
    """

    # Get the complete node list captured by the execution layer.
    #
    # This list may contain significantly more nodes than the model
    # ever sees.
    nodes = captured.get(
        "nodes",
        []
    )

    # Get the complete relationship list.
    rels = captured.get(
        "relationships",
        []
    )

    # Start with high-level counts.
    #
    # Example:
    #
    #     graph: 12 nodes, 15 relationships
    out = [
        f"graph: {len(nodes)} nodes, {len(rels)} relationships"
    ]

    # Extract all labels from all nodes.
    #
    # A set removes duplicates.
    #
    # Example:
    #
    #     Trial
    #     Trial
    #     Sponsor
    #
    # becomes:
    #
    #     {"Trial", "Sponsor"}
    labels = sorted(
        {
            label
            for node in nodes
            for label in node.get(
                "labels",
                []
            )
        }
    )

    # Only expose the first eight labels.
    #
    # The complete graph remains in server-side state.
    if labels:
        out.append(
            f"labels: {', '.join(labels[:8])}"
        )

    # A particularly important guard:
    #
    # Nodes without relationships do NOT establish a network.
    #
    # The model must not infer relationships simply because several
    # nodes were returned.
    if nodes and not rels:

        out.append(
            "NOTE: 0 relationships. This result establishes NO connections "
            "between these nodes — it is a node list, not a network. The usual "
            "cause is RETURNing bare node variables; return the path or the "
            "relationship variable instead. Do not describe these nodes as "
            "connected, and do not blame a cap that did not fire."
        )

    # The execution layer may cap the number of nodes returned.
    #
    # If that happened, explicitly tell the model so it does not interpret
    # the visible node count as the complete graph.
    if captured.get("truncated"):

        out.append(
            f"NOTE: truncated at the {cfg.graph_node_cap}-node cap."
        )

    # Convert the individual lines into the final compact summary.
    return "\n".join(
        out
    )


def _table(captured: dict, cfg) -> str:
    """Create the compact model-facing representation of a table result.

    The model receives:

        - total row count
        - total column count
        - first _COL_CAP column names
        - at most _SAMPLE_ROWS rows
        - explicit information about hidden columns
        - explicit information about additional rows
        - truncation information

    The complete table stays in application state.
    """

    # Complete row collection captured from Neo4j.
    rows = captured.get(
        "rows",
        []
    )

    # Normalize column names to strings.
    columns = [
        str(column)
        for column in captured.get(
            "columns",
            []
        )
    ]

    # Start the summary with overall table dimensions.
    #
    # Example:
    #
    #     table: 100 rows x 5 columns
    out = [
        f"table: {len(rows)} rows x {len(columns)} columns",

        # IMPORTANT:
        #
        # Use the same _COL_CAP here that is used when rendering
        # individual rows.
        #
        # This prevents the "column list says five, sample shows three"
        # mismatch described in the module documentation.
        f"columns: {', '.join(columns[:_COL_CAP])}"
    ]

    # Identify columns that are not shown because of the display cap.
    #
    # These columns still exist in the actual query result.
    hidden = columns[
        _COL_CAP:
    ]

    # Only create an untrusted-data block when there are actual rows.
    #
    # This keeps the boundary explicit:
    #
    #     <untrusted_data>
    #         database values
    #     </untrusted_data>
    if rows:
        out.append(
            "<untrusted_data>"
        )

    # Only expose the first three rows.
    #
    # The complete set remains available to the application.
    for row in rows[
        :_SAMPLE_ROWS
    ]:

        # Accumulate formatted cells for the current sample row.
        cells = []

        # Pair each visible column with its corresponding value.
        #
        # _COL_CAP is applied here as well so the sample and column list
        # have exactly the same visible width.
        for column, value in list(
            zip(columns, row)
        )[
            :_COL_CAP
        ]:

            # Convert and sanitize the database value.
            text = _clean(
                value
            )

            # IDs receive a larger limit.
            #
            # Normal values are aggressively compacted.
            cap = (
                _ID_CHARS
                if _is_id(column)
                else _VALUE_CHARS
            )

            # Check whether the sanitized value exceeds its display limit.
            if len(text) > cap:

                # Preserve the beginning of the value, but explicitly tell
                # the model that it was truncated.
                #
                # Most importantly, the model is told NOT to reuse this
                # partial value in a follow-up query.
                text = (
                    f"{text[:cap]}..."
                    f"[TRUNCATED, {len(_clean(value))} chars — "
                    "do not reuse this value; re-query for it]"
                )

            # Format the column/value pair.
            #
            # Example:
            #
            #     phase=Phase 3
            cells.append(
                f"{column}={text}"
            )

        # If there are hidden columns, explicitly tell the model that
        # additional columns exist.
        #
        # This prevents the model from assuming the visible sample is
        # the complete schema of the result.
        if hidden:
            cells.append(
                f"[+{len(hidden)} more columns not shown]"
            )

        # Add the formatted row to the output.
        out.append(
            "  " + ", ".join(cells)
        )

    # Close the untrusted-data fence if rows were present.
    if rows:
        out.append(
            "</untrusted_data>"
        )

    # If the table contained more columns than the display limit,
    # explicitly enumerate the hidden columns.
    #
    # This is important because the LLM must understand:
    #
    #     hidden != missing
    #
    # The columns WERE returned; they simply aren't displayed in the
    # compact model-facing representation.
    if hidden:

        out.append(
            f"NOTE: samples show the first {_COL_CAP} of {len(columns)} "
            f"columns. NOT SHOWN: {', '.join(hidden)}. These WERE returned "
            "and ARE captured — their absence here is a display limit."
        )

    # If more rows exist than the sample limit, tell the model how many
    # additional rows exist.
    #
    # Again, the complete data is still captured server-side.
    if len(rows) > _SAMPLE_ROWS:

        out.append(
            f"... and {len(rows) - _SAMPLE_ROWS} more rows "
            "(captured, not shown). "
            "Do not retype rows into your answer."
        )

    # The execution layer may have stopped collecting rows because the
    # configured safety cap was reached.
    #
    # Make that explicit so the model doesn't treat the result as
    # necessarily complete.
    if captured.get("truncated"):

        out.append(
            f"NOTE: truncated at the {cfg.row_cap}-row cap."
        )

    # Return the complete compact representation.
    return "\n".join(
        out
    )