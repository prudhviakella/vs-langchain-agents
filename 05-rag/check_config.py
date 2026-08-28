"""What settings are actually in effect?

    python check_config.py

Prints the environment variable, the value the package is using, and whether
they agree.

They disagree more often than you would think, because `rag.config` reads the
environment ONCE, at import. Setting a variable in a notebook cell after
importing has no effect until the kernel restarts, and nothing warns you — the
package simply keeps the value it read.
"""

import os

from rag import config

FLAGS = [
    ("DO_OCR", config.DO_OCR, "1",
     "CONDITIONAL. Reads text baked into graphics. Leave ON."),
    ("DO_CHART_EXTRACTION", config.DO_CHART_EXTRACTION, "0",
     "unconditional. 0/17 on vector charts — the real saving"),
    ("DO_CLASSIFICATION", config.DO_CLASSIFICATION, "1",
     "tags pictures chart/photo/logo"),
    ("DO_FORMULA", config.DO_FORMULA, "1",
     "equations -> LaTeX. Off means `formula-not-decoded`"),
    ("DO_CODE", config.DO_CODE, "1", "code blocks"),
    ("TABLE_MODE_ACCURATE", config.TABLE_MODE_ACCURATE, "1",
     "ACCURATE vs FAST table structure"),
    ("FIGURE_RENDER_SCALE", config.FIGURE_RENDER_SCALE, "2.0",
     "2x is legible to the vision model; 1x is not"),
    ("FIGURE_AREA_THRESHOLD", config.FIGURE_AREA_THRESHOLD, "0.01",
     "minimum share of a page before a figure is described"),
    ("CHUNK_TOKEN_TARGET", config.CHUNK_TOKENS, "1024", "chunk size"),
    ("EMBED_MODEL", config.EMBED_MODEL, "text-embedding-3-small", ""),
    ("VISION_MODEL", config.VISION_MODEL, "gpt-4o-mini", ""),
]

print(f"{'setting':<24}{'in env':<12}{'in effect':<12}{'':4}what it does")
print("-" * 96)

mismatched = []
for name, effective, default, note in FLAGS:
    in_env = os.getenv(name)
    shown_env = in_env if in_env is not None else "(unset)"

    # Compare as strings, because "0" and False are the same intent here.
    agrees = (in_env is None) or (str(effective).lower() in
                                  (in_env.lower(), str(in_env == "1").lower()))
    if not agrees:
        mismatched.append(name)

    print(f"{name:<24}{shown_env:<12}{str(effective):<12}"
          f"{'  ' if agrees else ' !!':<4}{note}")

if mismatched:
    print(f"\nMISMATCH: {', '.join(mismatched)}")
    print("The package is not using what the environment says. Almost always this")
    print("is because rag.config was imported before the variable was set — it")
    print("reads the environment once, at import.")
    print("\nRestart the kernel, then set them in a cell BEFORE any `from rag`"
          " import.")
else:
    print("\nEnvironment and package agree.")

# The three that break things when off, and what breaks.
if not config.DO_OCR:
    print("\nDO_OCR IS OFF.")
    print("  It is conditional — it only runs where there is no text layer, so it")
    print("  costs almost nothing on a digital PDF. But it is the only thing that")
    print("  reads text drawn inside a graphic. An exhibit made of coloured boxes")
    print("  with a list inside loses its entire contents: the caption and source")
    print("  line survive, everything in the boxes disappears.")
    print("  Switch off the unconditional models instead.")

print("\nIF FIGURES HAVE NO DESCRIPTIONS, CHECK THESE THREE:")
for name in ("do_picture_description", "generate_picture_images",
             "enable_remote_services"):
    print(f"  {name:<28} forced on in docling_io.py, not switchable")
print("\n  A figure with no description, with those three on, is either below")
print("  FIGURE_AREA_THRESHOLD or the model returned nothing. Run:")
print("      python diagnose_figure.py <parse.json> <index>")
