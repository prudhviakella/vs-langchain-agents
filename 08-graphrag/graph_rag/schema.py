"""The extraction schema — layer 3 only.

This layer extracts what the registry cannot know: the content of the protocol
document itself. Sponsor, phase, conditions, interventions, sites and outcomes are
all in ClinicalTrials.gov, correct by definition and free, so extracting them with a
language model would mean paying to guess at facts already available.

What is left is the part that exists only in the PDF — the operational detail a
registry summary does not carry. That is a smaller schema, a cheaper prompt, and
fewer surfaces on which to hallucinate.

Everything written from this layer carries `source: "extracted"`. The registry
layer writes `source: "registry"`. A query can then demand facts or accept claims,
and extraction accuracy becomes measurable, because for every field both layers
cover the registry is ground truth.

The schema is closed. An open prompt returns "Study", "Clinical Trial", "Trial" and
"the trial" as four entities, and a traversal from any one of them misses three
quarters of the edges — silently, because fewer rows looks like all the rows.
"""

# ─────────────────────────────────────────────────────────────────────────────
# What the registry already owns
#
# Listed so the boundary is explicit rather than implied by absence. If a type
# appears here, extracting it is duplicated effort at best and a competing wrong
# answer at worst.
# ─────────────────────────────────────────────────────────────────────────────
REGISTRY_OWNED = {
    "Trial", "Sponsor", "CRO", "Disease", "Drug", "Site", "Country",
    "Outcome", "MeSHTerm", "TrialCategory", "PatientPopulation",
}

# ─────────────────────────────────────────────────────────────────────────────
# What only the document knows
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_TYPES = {
    "Procedure":   "a study procedure or visit activity, such as a biopsy, infusion "
                   "or screening visit",
    "Assessment":  "a named measurement, instrument or laboratory panel, such as "
                   "ECOG performance status, HbA1c or a quality-of-life scale",
    "Criterion":   "a specific inclusion or exclusion requirement as the protocol "
                   "words it",
    "SafetyRule":  "a dose modification, interruption, discontinuation or stopping "
                   "rule",
    "StatMethod":  "a statistical method, analysis population or testing strategy",
    "Timepoint":   "a scheduled visit, week, cycle or study day at which something "
                   "happens",
}

RELATION_TYPES = {
    "PERFORMED_AT":  ("Procedure", "Timepoint"),
    "MEASURED_AT":   ("Assessment", "Timepoint"),
    "ASSESSED_BY":   ("Procedure", "Assessment"),
    "TRIGGERS":      ("Assessment", "SafetyRule"),
    "APPLIES_TO":    ("SafetyRule", "Procedure"),
    "ANALYSED_BY":   ("Assessment", "StatMethod"),
    "REQUIRES":      ("Criterion", "Assessment"),
}

_TEMPLATE = """Extract entities and relationships from this passage of a clinical
trial protocol.

The trial's registry facts — sponsor, phase, conditions, drugs, sites, countries and
outcome measures — are already known from ClinicalTrials.gov. Do not extract them.
Extract only the operational detail that exists in the protocol document.

Use only these entity types:
{entities}

Use only these relationship types, with the source and target types shown:
{relations}

Rules:
- Name entities as the passage names them, dropping articles and possessives.
- A Timepoint is a schedule position: "Week 12", "Cycle 2 Day 1", "Screening",
  "End of Treatment". Not a calendar date.
- Only extract a relationship the passage states. Do not infer that a procedure
  happens at a timepoint merely because both appear in the same paragraph.
- If the passage contains no entity of these types, return empty lists. Most
  boilerplate, legal text and administrative sections do.

Return JSON only, with no prose and no code fences:

{{"entities": [{{"name": "...", "type": "...", "detail": "..."}}],
  "relations": [{{"source": "...", "target": "...", "type": "..."}}]}}

`detail` is one short phrase from the passage describing the entity, or "".
"""

# Words carrying no identity. Stripped from the node key but kept in the display
# name, so "the ECOG performance status" and "ECOG performance status" are one node
# that still prints correctly.
_NOISE = {
    "the", "a", "an", "of", "for", "and", "or",
    "dr", "prof", "professor", "mr", "ms", "mrs",
    "inc", "llc", "ltd", "plc", "corp", "corporation", "co", "gmbh",
}


def extraction_prompt() -> str:
    """The prompt with the schema rendered into it.

    Generated from the dictionaries above rather than written out, so adding a type
    changes the prompt and the validation together. Keeping the two apart is how a
    schema drifts out of agreement with itself.
    """
    entities = "\n".join(f"  {name}: {desc}" for name, desc in ENTITY_TYPES.items())
    relations = "\n".join(f"  {name}: {src} -> {dst}"
                          for name, (src, dst) in RELATION_TYPES.items())
    return _TEMPLATE.format(entities=entities, relations=relations)


def normalise(name: str) -> str:
    """Canonical form of an entity name, used as its node key.

    Handles mechanical variation only — case, punctuation, articles. It will not
    merge "ECOG performance status" with "performance status", because that is a
    judgement, and a wrong merge creates a false edge, which answers a question
    incorrectly rather than not at all.

    Note how much lighter this needs to be than it would in layer 2. The registry
    resolves diseases and drugs through MeSH, a controlled vocabulary maintained by
    NLM — so the hard resolution problem is solved there, for free, and never
    reaches this layer.
    """
    cleaned = " ".join(name.replace(",", " ").replace(".", " ").split()).lower()
    words = [word for word in cleaned.split() if word not in _NOISE]
    return " ".join(words) or cleaned


def validate(payload: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Keep only what matches the schema, and say what was dropped and why.

    Rejects are returned rather than discarded. A type the model keeps proposing is
    usually a gap in the schema rather than a mistake — and if it proposes something
    in REGISTRY_OWNED, that is the prompt failing to hold the boundary, which is
    worth seeing rather than silently dropping.
    """
    entities: list[dict] = []
    relations: list[dict] = []
    rejected: list[str] = []

    known = {normalise(e["name"]): e for e in payload.get("entities", [])
             if e.get("type") in ENTITY_TYPES and e.get("name")}

    for entity in payload.get("entities", []):
        kind = entity.get("type")
        if kind in REGISTRY_OWNED:
            # Not a schema error — the registry has this, authoritatively. Counted
            # separately so the boundary can be seen holding or slipping.
            rejected.append(f"{kind} (registry already owns this)")
            continue
        if kind not in ENTITY_TYPES:
            rejected.append(f"entity type {kind!r}")
            continue
        if not entity.get("name"):
            continue
        entities.append({
            "name": entity["name"].strip(),
            "key": normalise(entity["name"]),
            "type": kind,
            "detail": (entity.get("detail") or "")[:200],
        })

    for relation in payload.get("relations", []):
        kind = relation.get("type")
        if kind not in RELATION_TYPES:
            rejected.append(f"relation type {kind!r}")
            continue

        source = normalise(relation.get("source", ""))
        target = normalise(relation.get("target", ""))
        if source not in known or target not in known:
            rejected.append(f"relation {kind} with an unextracted endpoint")
            continue

        expected_source, expected_target = RELATION_TYPES[kind]
        if (known[source]["type"] != expected_source
                or known[target]["type"] != expected_target):
            rejected.append(
                f"relation {kind} between {known[source]['type']} and "
                f"{known[target]['type']}, expected {expected_source} -> "
                f"{expected_target}")
            continue

        relations.append({"source": source, "target": target, "type": kind})

    return entities, relations, rejected
