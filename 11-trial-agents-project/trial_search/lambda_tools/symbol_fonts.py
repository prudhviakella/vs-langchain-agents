"""Symbol-font codes in extracted protocol text -> the characters they are.

    PDF (Symbol font)      Docling text          what an agent reads
    ─────────────────      ────────────          ───────────────────
    BP ≥ 150 mmHg     ──►  "BP \\uf0b3 150"   ──►  "BP ≥ 150 mmHg"       normalize()
    Grade ≥ 3         ──►  "Grade \\uf0b3 3"
    ALT > 10 × ULN    ──►  "\\uf03e 10 \\uf0b4"

WHAT HAPPENS WITHOUT THIS

Protocols set comparison signs in the Symbol font. Text extraction keeps
the font's own byte, shifted into Unicode's Private Use Area (U+F000 plus
the byte), where it has no standard meaning and renders as blank space. In
this corpus that is 966 of 5,764 chunks, in 10 of 20 protocols — the
places where dosing and eligibility thresholds live. An agent reading
"systolic BP  150 mmHg" must guess whether that is ≥, >, ≤ or <.

WHY THE TABLE IS SMALL AND EXPLICIT

A private code does not say which font it came from, and fonts disagree:
U+F06C is λ in Symbol but a ● bullet in Wingdings. Every entry below was
decided from its real contexts in this corpus — all 308 uses of U+F06C
open list items, so it maps to a bullet, not λ. A code not in the table is
shown as "[?]": visibly a symbol that could not be read, never a guess.

WHERE THIS RUNS

At READ time, in the Lambda's _passage(), for text and headings. Fixing it
at ingestion instead changes each affected chunk's text, therefore its
hash, therefore its chunk_id — and Pinecone and the Neo4j Chunk/NEXT graph
would then have to be rebuilt together. The same table serves that
re-ingestion later.

WHAT THIS DOES NOT DO

    - It does not touch the stored vectors. They were embedded from text
      containing the private codes; that affects ranking slightly, not
      what the agents read.
    - It does not map private codes outside U+F020–U+F0FF, which are not
      symbol-font codes.
"""

# Decided from contexts in this corpus. Count = occurrences in the export.
SYMBOL_FONT = {
    # comparison and arithmetic — the thresholds (Symbol font)
    "\uf0b3": "≥",   # x282  "Grade ≥ 3", "BP ≥ 150 mmHg"
    "\uf0a3": "≤",   # x294  "reduced to ≤ 10 mg/day"
    "\uf03e": ">",   # x118  "Proteinuria > 3.5 g/24 hours"
    "\uf03c": "<",   # x128  "Serum albumin < 2.5 g/dL"
    "\uf03d": "=",   # x255  "ULN = upper limit of normal"
    "\uf0b1": "±",   # x227  "Cycle 2 Day 1 ± 1 week"
    "\uf0b4": "×",   # x86   "ALT > 10 × ULN"
    "\uf02b": "+",   # x59   "Cycle 3 +"
    "\uf02d": "-",   # x425  "Days 1 - 21", "Regorafenib - Associated"
    "\uf0b0": "°",   # x3    "Fever > 38.5 °C"
    # Greek letters used as units and names (Symbol font)
    "\uf06d": "µ",   # x9    "181,000/µL"
    "\uf061": "α",   # x44   "TNF-α inhibitors"
    "\uf062": "β",   # x4    "β-catenin"
    "\uf067": "γ",   # x6    "γ-glutamyl transferase"
    "\uf074": "τ",   # x2    "AUC 0-τ"
    # typography (Symbol font)
    "\uf0ae": "→",   # x25   "If yes, → How severe"
    "\uf0e2": "®",   # x24   "FluMist ®", "Stivarga ®"
    "\uf0be": "—",   # x2    "on a periodic basis — approximately"
    "\uf0bc": "…",   # x3    "Left Ventricular Dysfunction … Most"
    # list bullets (Symbol and Wingdings) — every context opens a list item
    "\uf0b7": "•",   # x3738
    "\uf06c": "•",   # x308  Wingdings ●, NOT Symbol λ
    "\uf0a7": "•",   # x185  Wingdings ▪
    "\uf071": "•",   # x25   Wingdings ❑, between questionnaire options
    "\uf06e": "•",   # x6    Wingdings ■
    "\uf0a8": "•",   # x1
    "\uf09f": "•",   # x1
}
UNREADABLE = "[?]"           # e.g. U+F0E0, which only ever appears alone in a table cell
_TABLE = {ord(k): v for k, v in SYMBOL_FONT.items()}
_TABLE.update({cp: UNREADABLE for cp in range(0xF020, 0xF100) if cp not in _TABLE})


def normalize(text: str) -> str:
    """Replace symbol-font private codes with the characters they encode."""
    return text.translate(_TABLE) if text else text
