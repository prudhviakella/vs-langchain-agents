"""Is the code you edited the code that is running?

    python check_wiring.py

A notebook keeps imported modules in memory. Editing a file on disk does not
change what is already loaded, and nothing warns you — the old function keeps
running and the change appears to have done nothing.

This prints where each module was loaded from and whether the current fixes are
present in the LOADED copy, not on disk.

WHY EACH CHECK IS HERE

Every one of these corresponds to a bug that shipped and was invisible:

    demotion by class        setting item.label = TEXT does nothing. The
                             chunker tests isinstance(SectionHeaderItem), so a
                             relabelled header is still a header. The old code
                             printed "demoted 4 false headings" and changed no
                             chunk boundary for two full ingestions.

    drop filter              without it a quarter of the index was attribution
                             lines and a logo glyph, six of them byte-identical
                             and therefore six copies of one vector.

    prose merge              HybridChunker has a ceiling and no floor. Nothing
                             in it ever merges prose across an exhibit.

    figure pass              a figure sharing a vector with four other figures
                             answers a question about none of them.

    cached descriptions      the vision model is the only non-deterministic
                             stage, and its output is inside chunk_id. Uncached,
                             an unchanged PDF re-embeds every figure.
"""

import inspect as _inspect

from rag import chunking, docling_io, headings

print(f"{'module':<14}{'loaded from'}")
print("-" * 78)
for mod in (chunking, headings, docling_io):
    print(f"{mod.__name__.split('.')[-1]:<14}{mod.__file__}")

# Look inside the LOADED functions, not the files.
# The passes are separate functions now, so the markers live in several
# places. Read the whole loaded module rather than one function.
build = _inspect.getsource(chunking)
demote = _inspect.getsource(headings._demote)
describe = _inspect.getsource(docling_io.describe_figures)

checks = [
    ("headings are demoted by CLASS, not by label",
     "TextItem.model_validate" in demote and "doc.texts[index]" in demote),
    ("clean_headings runs before chunking",
     "clean_headings(doc)" in build
     and build.index("clean_headings(doc)") < build.index("chunker.chunk(")),
    ("page furniture is dropped before records are built",
     "is_furniture(" in build),
    ("prose is merged after the chunker",
     "PROSE_TARGET_TOKENS" in build),
    ("small records can reach a floor across headings",
     "MIN_CHUNK_TOKENS" in build),
    ("figures are handled in their own pass",
     "figure_slot" in build),
    ("figure captions are read from the document",
     "caption_text(doc)" in build),
    ("figure descriptions are cached on the rendered bytes",
     "FIGURE_CACHE" in describe and "sha256" in describe),
    ("merge_peers is off (our merge replaces it)",
     "merge_peers=False" in build),
    ("the passes are separate functions",
     all(hasattr(chunking, f) for f in
         ("_to_entries", "_merge_prose", "_apply_floor",
          "_to_records", "_table_summaries", "_finalise"))),
]

print()
for label, ok in checks:
    print(f"  {'yes' if ok else 'NO ':<5}{label}")

if not all(ok for _, ok in checks):
    print("\nThe loaded module is missing a fix that IS on disk.")
    print("Restart the kernel. `importlib.reload(...)` is not enough here:")
    print("build_records imports docling inside the function body, and an")
    print("already-compiled function object keeps the old code.")
else:
    print("\nThe loaded code is current.")

# Settings that change behaviour but not source, so the checks above cannot
# see them.
print()
print(f"  LAYOUT_MODEL                {docling_io.LAYOUT_MODEL or '(default)'}")
print(f"  TABLE_CELL_MATCHING         {docling_io.TABLE_CELL_MATCHING}")
print(f"  CACHE_FIGURE_DESCRIPTIONS   {docling_io.CACHE_FIGURE_DESCRIPTIONS}")
print(f"  MERGE_ACROSS_EXHIBITS       {chunking.MERGE_ACROSS_EXHIBITS}")
print(f"  PROSE_TARGET_TOKENS         {chunking.PROSE_TARGET_TOKENS}")
