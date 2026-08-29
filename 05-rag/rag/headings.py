"""Cleaning up heading labels before chunking.

    parsed document
          |
          v
    clean_headings()
          |
          |-- find SectionHeaderItems that are not section headings
          |-- REPLACE each with a TextItem at the same index
          |-- verify none survived
          |
          v
    HybridChunker
          |
          v
      chunks


WHY THIS EXISTS

HybridChunker never merges across a heading boundary, and in HybridChunker the
heading path is the ENTIRE merge predicate:

    if headings == current_headings and fits(candidate):
        merge

So every heading the layout model gets wrong is two faults at once: a boundary
that should not be there, and a wrong string prepended into the embedded text
of every chunk beneath it.

Measured on a 7-page research report: 16 detected SECTION_HEADERs, about half
of them real. `through 2024.` is a heading. So is `Exhibit 6:`. So is a
hundred-character sentence beginning "Please click here for the full excel
database".


WHY CHANGING THE LABEL DOES NOT WORK — READ THIS BEFORE EDITING

The obvious demotion is `item.label = DocItemLabel.TEXT`. It does nothing, and
it does nothing SILENTLY. From HierarchicalChunker.chunk():

    for item, level in dl_doc.iterate_items(with_groups=True, ...):
        if isinstance(item, TitleItem | SectionHeaderItem):
            level = item.level if isinstance(item, SectionHeaderItem) else 0
            ...
            heading_by_level[level] = item
            continue

The test is `isinstance`. The `label` field is never consulted for this
decision. A SectionHeaderItem whose label says TEXT is still a
SectionHeaderItem, so it is still captured as a heading, and the `continue`
means its text is not emitted as content either.

Pydantic permits the assignment because validate_assignment is off. Nothing
raises. A version of this file that logged "demoted 4 false heading(s)" ran
for two full ingestions and changed not one chunk boundary.

    HOW TO DEMOTE, ACTUALLY

        self_ref "#/texts/99"  ->  index 99 in doc.texts
                                        |
                                        v
                      construct a TextItem from the same fields
                                        |
                                        v
                              doc.texts[99] = new item

    self_ref is preserved, so every RefItem pointing at it still resolves and
    the parent's children list needs no edit.


WHAT THIS FILE DOES NOT DO

    It does NOT delete anything. The words stay in the document and become
    part of the chunk around them instead of splitting one.

    It does NOT fix OCR. `Al Agent Vendors` keeps its misread capital I here;
    that is an extraction problem.

    It does NOT touch TitleItem. A document title is a real heading.
"""

import re

# ---------------------------------------------------------------------------
# WHAT A HEADING IS NOT
#
# Each pattern below was derived from a real mislabelled heading, not
# invented. Anything more aggressive starts demoting real headings, and a real
# heading demoted is worse than a false one kept: it merges two genuine
# sections into one chunk, and nothing downstream can tell.
# ---------------------------------------------------------------------------

# A bare exhibit or figure number. The caption text belongs to the exhibit,
# not to a section.
#     "Exhibit 6:"    "Figure 12"    "Table 3:"
NUMBERED_LABEL = re.compile(r"^(exhibit|figure|table|chart|panel)\s*\d+\s*:?\s*$",
                            re.IGNORECASE)

# Ends like a sentence. A heading does not.
#     "through 2024."
SENTENCE_END = re.compile(r"[.,;]\s*$")

# Starts mid-thought. A fragment of the paragraph above, cut by a line break
# the layout model read as a boundary.
#     "through 2024."   "and the rest of the market"
CONTINUATION = re.compile(
    r"^(and|but|or|through|which|that|while|with|from|to|for|of|in|on|at)\s",
    re.IGNORECASE)

# Long enough to be a sentence rather than a title.
#     "Please click here for the full excel database of >3,700 stocks..."
#
# 120, not 80. Measured on a 113-page clinical protocol: exactly 3 headings
# exceeded 80 characters, TWO OF THEM REAL —
#
#     '8.2 Participating Study Sites, Investigators and Staff, Laboratories,
#      and Sponsor Information'                                      93 chars
#     '8.3.2 Durvalumab and Tremelimumab Dose Modification Not Due to
#      Treatment-related Toxicities'                                 90 chars
#
# The third was a sentence fragment which SENTENCE_END catches anyway. So at
# 80 this rule contributed zero correct demotions and two wrong ones on that
# document, and a real heading demoted is the expensive direction: it merges
# two genuine sections and nothing downstream can tell.
MAX_HEADING_CHARS = 120

# A numbered section heading, exempt from the length rule entirely.
#     "8.3.2 Durvalumab and Tremelimumab Dose Modification..."
#     "7.1 Inclusion Criteria"
#     "A3a.  Financial Conflict of Interest"
#
# Regulatory and technical documents number their sections, and those titles
# are long by convention — they are written to be unambiguous in a table of
# contents, not short. A number prefix is strong evidence of a real heading,
# so length stops being evidence of anything.
#
# The other rules still apply. A numbered line that ends like a sentence is
# still demoted, which is what catches a numbered list item the layout model
# mislabelled.
NUMBERED_SECTION = re.compile(r"^(\d+(\.\d+)*|[A-Z]\d*[a-z]?)[.)]?\s+\S")


def why_not_a_heading(text: str) -> str | None:
    """Reason this text is not a section heading, or None if it might be.

    Returns a reason string rather than a bool so the caller can report what
    it changed. A silent correction is one nobody checks.

    Examples:

        "Mapping AI's Rate of Change in Charts"  ->  None
        "through 2024."                          ->  "ends like a sentence"
        "Exhibit 6:"                             ->  "a numbered label"
    """
    stripped = text.strip()

    if not stripped:
        return "empty"
    if NUMBERED_LABEL.match(stripped):
        return "a numbered label, not a section"
    if SENTENCE_END.search(stripped):
        return "ends like a sentence"
    if CONTINUATION.match(stripped):
        return "starts mid-sentence"
    if len(stripped) > MAX_HEADING_CHARS and not NUMBERED_SECTION.match(stripped):
        return f"{len(stripped)} characters, too long for a heading"
    return None


# ---------------------------------------------------------------------------
# THE ONES TEXT CANNOT CATCH
#
# "Al Agent Infrastructure" and "Al Agent Vendors" are labels on two coloured
# boxes inside Exhibit 6. As strings they are indistinguishable from a real
# heading: short, capitalised, no punctuation, title case.
#
# So the signal has to be STRUCTURAL. Both share a shape:
#
#     SECTION_HEADER   "Exhibit 6:"              <- a numbered label
#     TEXT             "AI Agent Infrastructure and Vendor plays..."
#     SECTION_HEADER   "Al Agent Infrastructure" <- a box label
#     LIST_ITEM        MongoDB ...
#     LIST_ITEM        Elastic NV ...
#     SECTION_HEADER   "Al Agent Vendors"        <- a box label
#     LIST_ITEM        Salesforce ...
#
# A real section heading is followed by prose, or by a mix of things. These
# are followed by NOTHING BUT list items, and they sit inside an exhibit.
#
# Two conditions, both required:
#
#     1. the heading is immediately followed by a run of list items
#     2. an exhibit label appeared recently before it
#
# Condition 2 is what stops this demoting a legitimate heading that happens to
# introduce a bulleted list, which is common and perfectly normal.
# ---------------------------------------------------------------------------

# How many elements back to look for the exhibit label. Wide enough to cross a
# caption and a stray text line, narrow enough that the next real section does
# not still count as "inside the exhibit".
EXHIBIT_LOOKBACK = 6

# How many consecutive list items must follow before a heading is treated as a
# box label. Two is common in ordinary prose; three or more, directly under a
# heading, inside an exhibit, is the shape of a labelled box.
MIN_LIST_RUN = 3


def _is_group(item) -> bool:
    """True for a container rather than a leaf element.

    iterate_items(with_groups=True) yields a ListGroup BEFORE the ListItems it
    holds. That container sits between a heading and its list items, so a run
    count that treats it as a non-list-item stops at zero and _box_labels never
    fires.

    Counting groups as list items would be wrong the other way. They are
    skipped instead: transparent to the run, and not a run of their own.
    """
    try:
        from docling_core.types.doc.document import GroupItem
        return isinstance(item, GroupItem)
    except ImportError:
        return type(item).__name__.endswith("Group")


def _is_list_item(item) -> bool:
    """True for a list item, by class or by label.

    Checked both ways because ListItem is a subclass of TextItem in some
    docling versions and a sibling in others, and the label survives either.
    """
    from docling_core.types.doc import DocItemLabel
    try:
        from docling_core.types.doc import ListItem
        if isinstance(item, ListItem):
            return True
    except ImportError:
        pass
    return getattr(item, "label", None) == DocItemLabel.LIST_ITEM


def _box_labels(items: list) -> set[str]:
    """self_refs of headings that are labels inside an exhibit.

    Returns REFS, not indices. Indices into a walk that includes group
    containers do not line up with indices into one that does not, and both
    EXHIBIT_LOOKBACK and MIN_LIST_RUN are measured in elements — so a ListGroup
    sitting between a heading and its items both breaks the run AND pushes the
    next heading out of lookback range. Measured: with containers left in, the
    two box labels inside Exhibit 6 went from both caught to neither.

    Containers are therefore removed before anything is counted, and the
    result is keyed on something that survives that.

    Returns:
        The refs to demote. Empty when nothing matches, which is the common
        case for a document with ordinary headings.
    """
    from docling_core.types.doc.document import SectionHeaderItem

    # Leaves only. This restores the geometry the two constants below were
    # measured against.
    items = [item for item in items if not _is_group(item)]

    found: set[str] = set()
    exhibit_at = None

    for i, item in enumerate(items):
        text = (getattr(item, "text", "") or "").strip()

        # Remember where the last exhibit label was.
        if NUMBERED_LABEL.match(text):
            exhibit_at = i
            continue

        if not isinstance(item, SectionHeaderItem):
            continue

        # Condition 2 — are we still inside an exhibit?
        if exhibit_at is None or i - exhibit_at > EXHIBIT_LOOKBACK:
            continue

        # Condition 1 — is it immediately followed by a run of list items?
        #
        # COUNTED FROM THE FRONT, not checked across everything it owns. An
        # exhibit usually ends with a source line:
        #
        #     SECTION_HEADER   "Al Agent Vendors"
        #     LIST_ITEM        Salesforce ...
        #     ... 11 more ...
        #     TEXT             "Source: Morgan Stanley Research"
        #
        # Requiring EVERYTHING it owns to be a list item fails on that
        # trailing line, and the second box label survives while the first is
        # caught — which is worse than catching neither, because the exhibit
        # is then split in a way nobody would predict.
        run = 0
        for follower in items[i + 1:]:
            if not _is_list_item(follower):
                break
            run += 1

        if run >= MIN_LIST_RUN:
            ref = getattr(item, "self_ref", None)
            if ref:
                found.add(ref)
            # A box label does not end the exhibit — the second box is still
            # inside it, so the lookback anchor moves forward rather than
            # being cleared.
            exhibit_at = i

    return found


def _texts_index(self_ref: str) -> int | None:
    """Position in doc.texts that a self_ref points at.

        "#/texts/99"  ->  99
        "#/pictures/3" -> None

    Returns None for anything that is not a text ref, so the caller skips it
    rather than corrupting a different collection.
    """
    match = re.fullmatch(r"#/texts/(\d+)", self_ref or "")
    return int(match.group(1)) if match else None


def _demote(doc, item) -> bool:
    """Replace one SectionHeaderItem with an equivalent TextItem, in place.

        doc.texts[N]   SectionHeaderItem   "through 2024."
              |
              v
        doc.texts[N]   TextItem            "through 2024."

    Everything is carried over except `level`, which TextItem does not have.
    self_ref is preserved, so refs held elsewhere — captions, chunk doc_items,
    the parent's children list — keep resolving to this position.

    Returns:
        True when the replacement happened. False means the item could not be
        located in doc.texts, and the caller must report it rather than
        assume success.
    """
    from docling_core.types.doc import DocItemLabel
    from docling_core.types.doc.document import TextItem

    index = _texts_index(getattr(item, "self_ref", ""))
    if index is None or index >= len(doc.texts) or doc.texts[index] is not item:
        return False

    data = item.model_dump()
    data.pop("level", None)
    data["label"] = DocItemLabel.TEXT

    doc.texts[index] = TextItem.model_validate(data)
    return True


def clean_headings(doc, verbose: bool = True) -> list[tuple[str, str]]:
    """Demote SectionHeaderItems that are not section headings.

        1  collect items in document order
        2  structural pass, on the ORIGINAL classes
        3  text pass
        4  replace each demoted item in doc.texts
        5  verify none survived

    Mutates the document, so the chunker sees a corrected version. Nothing is
    deleted.

    Returns:
        A list of (text, reason) for every demotion, so the caller can print
        what changed. Empty means the headings looked sound.
    """
    from docling_core.types.doc.document import SectionHeaderItem

    items = [item for item, _ in doc.iterate_items(with_groups=True)]

    # Structural pass first, because it needs the ORIGINAL classes. Running it
    # after the text pass would mean the exhibit label has already been
    # demoted, so nothing anchors the lookback.
    box_label_refs = _box_labels(items)

    demoted: list[tuple[str, str]] = []
    failed: list[str] = []

    for item in items:
        if not isinstance(item, SectionHeaderItem):
            continue

        text = getattr(item, "text", "") or ""
        reason = why_not_a_heading(text)
        if reason is None and getattr(item, "self_ref", None) in box_label_refs:
            reason = "a label inside an exhibit, not a section"
        if reason is None:
            continue

        if _demote(doc, item):
            demoted.append((text[:70], reason))
        else:
            failed.append(text[:70])

    # ── verification ────────────────────────────────────────────────────────
    # The previous version of this file changed item.label and reported
    # success while the chunker went on treating every one of these as a
    # heading. Reporting a demotion is not evidence that it happened, so the
    # document is re-read and the surviving headings are counted.
    survivors = [getattr(i, "text", "") or ""
                 for i, _ in doc.iterate_items(with_groups=True)
                 if isinstance(i, SectionHeaderItem)]
    still_false = [t for t in survivors if why_not_a_heading(t)]

    if verbose:
        if demoted:
            print(f"  demoted {len(demoted)} false heading(s):", flush=True)
            for text, reason in demoted:
                print(f"    {text!r} — {reason}", flush=True)
        print(f"  {len(survivors)} heading(s) remain", flush=True)
        if failed:
            print(f"  WARNING: {len(failed)} heading(s) matched a rule but could "
                  "not be located in doc.texts and were NOT demoted: "
                  f"{failed}", flush=True)
        if still_false:
            print(f"  WARNING: {len(still_false)} heading(s) still match a "
                  f"demotion rule after the pass: {still_false}. The "
                  "replacement did not take effect — check that _demote is "
                  "writing to doc.texts and not to a copy.", flush=True)

    return demoted
