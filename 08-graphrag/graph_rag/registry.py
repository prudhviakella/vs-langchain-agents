"""Trial facts from ClinicalTrials.gov.

This is the authoritative layer. Sponsor, phase, conditions, interventions,
locations and outcomes are all recorded in the registry, so extracting them from a
PDF with a language model would mean paying to guess at facts that are available for
free and correct by definition.

Anything written here carries `source: "registry"`. Layer 3 writes
`source: "extracted"`. That distinction is what lets a query demand facts rather
than accept claims — and it is what makes extraction accuracy measurable, because
the registry is ground truth for every field it covers.

Structure follows the fetch → parse → load separation: the loaders never see the raw
API shape, only the canonical record `parse_trial` produces. Each concern can then be
tested and changed on its own.

The v2 API is public and needs no key.
    https://clinicaltrials.gov/api/v2/studies/NCT04368728

EVERY NODE GETS A .key

    store.find_chunks() and store.neighbours() — the functions that turn a question's
    words into graph traversal — both search on a `.key` property. Earlier, only the
    extraction layer's six entity types had one; every node written here had a
    domain-specific identity property instead (nctId, name, facility, measure,
    term), and none of them was named `.key`. The effect was silent, not an error:
    a question naming a sponsor, a drug, a disease, a country or a site matched
    nothing, because the property the search ran on simply did not exist on that
    node. The graph was correct; it was just invisible to term search.

    The fix is not to make the search function check five property names instead
    of one — that only holds until a sixth node type shows up. Every node below now
    gets `.key = normalise(...)` on its own display name, using the exact function
    `schema.py` already uses as the extraction layer's identity convention. One
    normalisation function, one identity property, applied everywhere. The
    `find_chunks`/`neighbours` queries in store.py do not change at all — they were
    always correct; they had nothing to find.

    `.key` is for SEARCH. The MERGE identity — what makes a re-run converge instead
    of duplicate — is still each node's own real-world identifier (nctId, name,
    facility, (measure, nctId)), unchanged below. Two different jobs, and conflating
    them would be its own mistake: a Trial's uniqueness is its nctId, not a
    lowercased, noise-stripped string that a differently-titled trial could collide
    with.
"""

import json
import time

from . import config
from .schema import normalise

CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"

REGISTRY_CACHE = config.CACHE_DIR.parent / "registry"
REGISTRY_CACHE.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Fetch
# ═════════════════════════════════════════════════════════════════════════════

def fetch_trial(nct_id: str, use_cache: bool = True) -> dict | None:
    """One study record from the registry, cached on disk.

    Cached because the registry is slow-moving: a completed trial's record does not
    change between runs, and re-fetching twenty of them on every notebook execution
    wastes time and is discourteous to a public service.

    Returns None rather than raising when a trial is not found. A corpus commonly
    contains a protocol whose NCT number was superseded or never registered, and one
    missing trial should not stop the other nineteen loading.
    """
    import requests

    cached = REGISTRY_CACHE / f"{nct_id}.json"
    if use_cache and cached.exists():
        return json.loads(cached.read_text())

    for attempt in range(3):
        try:
            response = requests.get(f"{CT_API_BASE}/{nct_id}",
                                    params={"format": "json"}, timeout=30)
            if response.status_code == 404:
                print(f"  {nct_id}: not found in the registry", flush=True)
                return None
            response.raise_for_status()
            payload = response.json()
            cached.write_text(json.dumps(payload))
            return payload
        except Exception as exc:
            if attempt == 2:
                print(f"  {nct_id}: fetch failed ({exc})", flush=True)
                return None
            time.sleep(2 ** attempt)
    return None


def dig(data: dict, *keys, default=None):
    """Walk nested dictionaries, returning `default` at the first missing key.

    The v2 response is deeply nested and many modules are optional — a phase 1 trial
    has no results section, an observational study no interventions. Guarding each
    access individually would triple the length of `parse_trial` and hide its shape.
    """
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


# ═════════════════════════════════════════════════════════════════════════════
# Parse
# ═════════════════════════════════════════════════════════════════════════════

def parse_trial(raw: dict) -> dict:
    """Turn the raw v2 JSON into a flat record ready to load.

    Separating this from loading means the Cypher never sees the API's shape. When
    the API changes — and v1 to v2 was a complete restructure — only this function
    moves.
    """
    protocol = dig(raw, "protocolSection", default={})
    ident = dig(protocol, "identificationModule", default={})
    status = dig(protocol, "statusModule", default={})
    design = dig(protocol, "designModule", default={})
    arms = dig(protocol, "armsInterventionsModule", default={})
    sponsors = dig(protocol, "sponsorCollaboratorsModule", default={})
    eligibility = dig(protocol, "eligibilityModule", default={})
    # v2 renamed this from locationsModule; reading the old key returns nothing.
    locations = dig(protocol, "contactsLocationsModule", default={})
    outcomes = dig(protocol, "outcomesModule", default={})
    conditions = dig(protocol, "conditionsModule", default={})

    # derivedSection is a SIBLING of protocolSection at the top level, not nested
    # inside it. Reading it from the protocol section silently returns nothing,
    # which is why MeSH terms are a common casualty.
    derived = dig(raw, "derivedSection", default={})

    # Condition and intervention MeSH terms together, so drug vocabulary
    # ("Antiviral Agents") sits alongside disease vocabulary ("COVID-19").
    mesh = (dig(derived, "conditionBrowseModule", "meshes", default=[])
            + dig(derived, "interventionBrowseModule", "meshes", default=[]))

    return {
        "trial": {
            "nctId": dig(ident, "nctId"),
            "briefTitle": dig(ident, "briefTitle"),
            "officialTitle": dig(ident, "officialTitle"),
            "acronym": dig(ident, "acronym"),
            "overallStatus": dig(status, "overallStatus"),
            "startDate": dig(status, "startDateStruct", "date"),
            "primaryCompletionDate": dig(status, "primaryCompletionDateStruct", "date"),
            "completionDate": dig(status, "completionDateStruct", "date"),
            "lastUpdateSubmitDate": dig(status, "lastUpdateSubmitDate"),
            # Phases is a list because a trial can straddle two, e.g. PHASE2/PHASE3.
            "phase": ", ".join(dig(design, "phases", default=[])),
            "studyType": dig(design, "studyType"),
            "enrollmentCount": dig(design, "enrollmentInfo", "count"),
            "enrollmentType": dig(design, "enrollmentInfo", "type"),
        },
        "conditions": dig(conditions, "conditions", default=[]),
        "interventions": [
            {"name": dig(iv, "interventionName"),
             "type": dig(iv, "interventionType"),
             "otherNames": dig(iv, "otherNames", default=[])}
            for iv in dig(arms, "interventions", default=[])
        ],
        "lead_sponsor": dig(sponsors, "leadSponsor", "name"),
        "collaborators": [dig(c, "name") for c in
                          dig(sponsors, "collaborators", default=[]) if dig(c, "name")],
        "locations": [
            {"facility": dig(loc, "facility"), "city": dig(loc, "city"),
             "country": dig(loc, "country"), "zip": dig(loc, "zip"),
             # The facility's own coordinates, not a country centroid — which
             # would be meaningless for a multi-site trial and makes
             # distance queries wrong rather than approximate.
             "lat": dig(loc, "geoPoint", "lat"), "lon": dig(loc, "geoPoint", "lon")}
            for loc in dig(locations, "locations", default=[])
        ],
        "primary_outcomes": [
            {"measure": dig(o, "measure"), "description": dig(o, "description", default=""),
             "timeFrame": dig(o, "timeFrame", default=""), "type": "primary"}
            for o in dig(outcomes, "primaryOutcomes", default=[])
        ],
        "secondary_outcomes": [
            {"measure": dig(o, "measure"), "description": dig(o, "description", default=""),
             "timeFrame": dig(o, "timeFrame", default=""), "type": "secondary"}
            for o in dig(outcomes, "secondaryOutcomes", default=[])
        ],
        "patient_population": {
            "eligibilityCriteria": dig(eligibility, "eligibilityCriteria"),
            "gender": dig(eligibility, "sex"),
            "minimumAge": dig(eligibility, "minimumAge"),
            "maximumAge": dig(eligibility, "maximumAge"),
            "stdAges": dig(eligibility, "stdAges", default=[]),
            "healthyVolunteers": str(dig(eligibility, "healthyVolunteers", default="")),
        },
        # NLM's controlled vocabulary, and the reason this layer needs no fuzzy
        # entity matching. Two trials studying "COVID-19" and "SARS-CoV-2 Infection"
        # connect through a shared MeSH node without anyone guessing they are the
        # same thing.
        "mesh_terms": [dig(m, "term") for m in mesh if dig(m, "term")],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Load
# ═════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    "CREATE CONSTRAINT trial_nct_id     IF NOT EXISTS FOR (t:Trial)           REQUIRE t.nctId IS UNIQUE",
    "CREATE CONSTRAINT disease_name     IF NOT EXISTS FOR (d:Disease)         REQUIRE d.name  IS UNIQUE",
    "CREATE CONSTRAINT drug_name        IF NOT EXISTS FOR (d:Drug)            REQUIRE d.name  IS UNIQUE",
    "CREATE CONSTRAINT sponsor_name     IF NOT EXISTS FOR (s:Sponsor)         REQUIRE s.name  IS UNIQUE",
    "CREATE CONSTRAINT cro_name         IF NOT EXISTS FOR (c:CRO)             REQUIRE c.name  IS UNIQUE",
    "CREATE CONSTRAINT country_name     IF NOT EXISTS FOR (c:Country)         REQUIRE c.name  IS UNIQUE",
    "CREATE CONSTRAINT site_facility    IF NOT EXISTS FOR (s:Site)            REQUIRE s.facility IS UNIQUE",
    "CREATE CONSTRAINT mesh_term        IF NOT EXISTS FOR (m:MeSHTerm)        REQUIRE m.term  IS UNIQUE",
    "CREATE CONSTRAINT category_name    IF NOT EXISTS FOR (c:TrialCategory)   REQUIRE c.name  IS UNIQUE",
    # Indexes on the properties the loaders MATCH on mid-transaction, and that
    # queries filter by.
    "CREATE INDEX trial_status IF NOT EXISTS FOR (t:Trial) ON (t.overallStatus)",
    "CREATE INDEX trial_phase  IF NOT EXISTS FOR (t:Trial) ON (t.phase)",
    "CREATE INDEX site_city    IF NOT EXISTS FOR (s:Site)  ON (s.city)",
    # .key carries the search burden now, on every label written below — a lookup
    # index makes find_chunks/neighbours a index seek instead of a label scan.
    "CREATE INDEX trial_key    IF NOT EXISTS FOR (t:Trial)         ON (t.key)",
    "CREATE INDEX disease_key  IF NOT EXISTS FOR (d:Disease)       ON (d.key)",
    "CREATE INDEX drug_key     IF NOT EXISTS FOR (d:Drug)          ON (d.key)",
    "CREATE INDEX sponsor_key  IF NOT EXISTS FOR (s:Sponsor)       ON (s.key)",
    "CREATE INDEX cro_key      IF NOT EXISTS FOR (c:CRO)           ON (c.key)",
    "CREATE INDEX country_key  IF NOT EXISTS FOR (c:Country)       ON (c.key)",
    "CREATE INDEX site_key     IF NOT EXISTS FOR (s:Site)          ON (s.key)",
    "CREATE INDEX outcome_key  IF NOT EXISTS FOR (o:Outcome)       ON (o.key)",
    "CREATE INDEX mesh_key     IF NOT EXISTS FOR (m:MeSHTerm)      ON (m.key)",
    "CREATE INDEX category_key IF NOT EXISTS FOR (c:TrialCategory) ON (c.key)",
]


def create_constraints(session) -> None:
    """Uniqueness constraints and lookup indexes, before any load.

    The constraints are what make MERGE idempotent — without one, two MERGE
    statements on the same key can both create a node, and re-running the loader
    silently duplicates the graph.
    """
    for statement in CONSTRAINTS:
        try:
            session.run(statement)
        except Exception as exc:
            # Already existing is the normal case on a re-run and not an error.
            print(f"  DDL skipped: {str(exc)[:80]}", flush=True)


# Rows per UNWIND call — same value and same reasoning as structure.py and
# the student-facing dump/load tools: small enough that one query's
# parameter payload is never the bottleneck, large enough that this stays
# a handful of round trips rather than hundreds.
BATCH_SIZE = 500


def _run_batched(session, query: str, rows: list[dict]) -> None:
    """UNWIND $batch through `query` in chunks of BATCH_SIZE. Used
    throughout below in place of one session.run() call per row."""
    for i in range(0, len(rows), BATCH_SIZE):
        session.run(query, batch=rows[i:i + BATCH_SIZE])


def load_trials(session, nct_ids: list[str], verbose: bool = True) -> dict:
    """Fetch, parse and load every trial. Missing ones are reported, not fatal.

    WHY THIS BATCHES ACROSS ALL TRIALS, NOT PER TRIAL

        load_trials(session, nct_ids)
            |
            |-- fetch_trial() + parse_trial(), one call per nct_id  (STEP 1)
            |   (network calls to ClinicalTrials.gov's own API —
            |    unavoidably one per trial, and a different bottleneck
            |    from the Neo4j writes below entirely)
            |
            |-- STEP 2   Trial nodes           one UNWIND, every trial
            |-- STEP 3   TrialCategory + edge  one UNWIND, every trial
            |-- STEP 4   Disease + edge        one UNWIND, every condition,
            |                                  every trial
            |-- STEP 5   Drug + edge           one UNWIND, every intervention
            |-- STEP 6   Sponsor + edge        one UNWIND, every trial with one
            |-- STEP 7   CRO + edge            one UNWIND, every collaborator
            |-- STEP 8   Country + edge        one UNWIND, every location
            |-- STEP 9   Site + edges          one UNWIND, every location
            |                                  with a facility
            |-- STEP 10  Outcome + edge        one UNWIND, every outcome
            |-- STEP 11  PatientPopulation     one UNWIND, every trial with data
            |-- STEP 12  MeSHTerm + edge       one UNWIND, every term
            |
            v
        {"loaded": [...], "missing": [...]}

        The previous version called load_trial() once per trial, and inside
        it, session.run() once per condition, per intervention, per
        collaborator, per location (TWICE — once for the country, once for
        the site), per outcome, per MeSH term. Measured directly against
        this exact 20-trial corpus: one trial alone had 172 locations, so
        just that trial's location loop was ~344 individual round trips
        before counting anything else. Summed across all 20 trials, that
        is easily 2,500-3,000+ round trips in one long-running Neo4j
        session — long enough that Aura's own connection actually died
        mid-run (SessionExpired: Failed to read from defunct connection),
        not merely ran slowly the way structure.py's 21-minute case did.

        The fix is the same one applied there: collect every row this
        function would have written one at a time, across ALL 20 trials
        at once, and send each node/relationship type through Cypher's
        UNWIND in batches of BATCH_SIZE instead of one row per round trip.
        Trial nodes are written first and everything else MATCHes them,
        preserving the same dependency the original per-trial version had
        (a trial's other facts always assumed load_trial() had already
        MERGEd the Trial node earlier in that same call).

    WHAT THIS DOES NOT DO

        - It does not change what gets written — same nodes, same
          relationships, same properties, same `key`/`source` conventions,
          same return shape. This is a performance rewrite only.
        - It does not deduplicate repeated countries within a trial before
          batching. The original called Country MERGE once per location
          even when several locations shared a country; batching preserves
          that same row-per-location shape. MERGE is idempotent regardless,
          so the result is identical either way — this only affects how
          many (harmless, no-op) rows are in the batch, not correctness.
        - It does not change fetch_trial()'s own per-trial network calls to
          ClinicalTrials.gov. Those are a different bottleneck (an external
          API, not this database) and were never the source of the timeout.
    """
    # STEP 1 — fetch and parse every trial first. Network calls to a
    # different service entirely; kept exactly as before.
    loaded, missing = [], []
    records = []
    for nct_id in nct_ids:
        raw = fetch_trial(nct_id)
        if raw is None:
            missing.append(nct_id)
            continue
        record = parse_trial(raw)
        records.append(record)
        loaded.append(nct_id)
        if verbose:
            trial = record["trial"]
            print(f"  {nct_id}  {trial['phase'] or 'n/a':<16} "
                  f"{(record['lead_sponsor'] or '')[:34]:<36} "
                  f"{len(record['locations'])} sites", flush=True)

    if verbose and missing:
        print(f"\n  not found: {', '.join(missing)}", flush=True)

    _load_records(session, records)
    return {"loaded": loaded, "missing": missing}


def _load_records(session, records: list[dict]) -> None:
    """Batch-write every node and relationship type, across every record
    in one call. See load_trials()'s own docstring for why.
    """
    # STEP 2 — Trial nodes. Written first: every later batch MATCHes a
    # Trial by nctId, exactly as the original per-trial code assumed the
    # Trial already existed from its own first statement.
    trial_rows = [{"nctId": r["trial"]["nctId"], "key": normalise(r["trial"]["nctId"]),
                  "props": {k: v for k, v in r["trial"].items() if k != "nctId"}}
                 for r in records if r["trial"]["nctId"]]
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (t:Trial {nctId: row.nctId})
        SET t += row.props, t.key = row.key, t.source = 'registry'
    """, trial_rows)

    valid_nct_ids = {row["nctId"] for row in trial_rows}
    records = [r for r in records if r["trial"]["nctId"] in valid_nct_ids]

    # STEP 3 — TrialCategory, one per trial from its first listed condition.
    category_rows = [{"nctId": r["trial"]["nctId"],
                      "category": r["conditions"][0] if r["conditions"] else "Unknown"}
                     for r in records]
    for row in category_rows:
        row["key"] = normalise(row["category"])
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (c:TrialCategory {name: row.category})
        SET c.key = row.key, c.source = 'registry'
        WITH c, row MATCH (t:Trial {nctId: row.nctId})
        MERGE (t)-[:BELONGS_TO]->(c)
    """, category_rows)

    # STEP 4 — Disease, one row per (trial, condition).
    disease_rows = [{"nctId": r["trial"]["nctId"], "name": c.strip(),
                     "key": normalise(c.strip())}
                    for r in records for c in r["conditions"] if c and c.strip()]
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (d:Disease {name: row.name})
        SET d.key = row.key, d.source = 'registry'
        WITH d, row MATCH (t:Trial {nctId: row.nctId})
        MERGE (t)-[:TARGETS]->(d)
    """, disease_rows)

    # STEP 5 — Drug, one row per (trial, intervention).
    drug_rows = []
    for r in records:
        for interv in r["interventions"]:
            name = (interv.get("name") or "").strip()
            if name:
                drug_rows.append({
                    "nctId": r["trial"]["nctId"], "name": name,
                    "type": interv.get("type"),
                    "otherNames": interv.get("otherNames", []),
                    "key": normalise(name),
                })
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (d:Drug {name: row.name})
        SET d.type = row.type, d.otherNames = row.otherNames,
            d.key = row.key, d.source = 'registry'
        WITH d, row MATCH (t:Trial {nctId: row.nctId})
        MERGE (t)-[:TESTS]->(d)
    """, drug_rows)

    # STEP 6 — Sponsor, one row per trial that has a lead sponsor.
    sponsor_rows = [{"nctId": r["trial"]["nctId"], "name": r["lead_sponsor"].strip(),
                     "key": normalise(r["lead_sponsor"].strip())}
                    for r in records if r["lead_sponsor"]]
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (s:Sponsor {name: row.name})
        SET s.key = row.key, s.source = 'registry'
        WITH s, row MATCH (t:Trial {nctId: row.nctId})
        MERGE (t)-[:SPONSORED_BY]->(s)
    """, sponsor_rows)

    # STEP 7 — CRO, one row per (trial, collaborator).
    cro_rows = [{"nctId": r["trial"]["nctId"], "name": c.strip(), "key": normalise(c.strip())}
               for r in records for c in r["collaborators"] if c and c.strip()]
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (c:CRO {name: row.name})
        SET c.key = row.key, c.source = 'registry'
        WITH c, row MATCH (t:Trial {nctId: row.nctId})
        MERGE (t)-[:MANAGED_BY]->(c)
    """, cro_rows)

    # STEP 8 — Country, one row per location (not deduplicated within a
    # trial — see the module-level docstring for why that is fine).
    country_rows = [{"nctId": r["trial"]["nctId"],
                     "country": (loc.get("country") or "").strip(),
                     "key": normalise((loc.get("country") or "").strip())}
                    for r in records for loc in r["locations"]
                    if (loc.get("country") or "").strip()]
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (c:Country {name: row.country})
        SET c.key = row.key, c.source = 'registry'
        WITH c, row MATCH (t:Trial {nctId: row.nctId})
        MERGE (t)-[:CONDUCTED_IN]->(c)
    """, country_rows)

    # STEP 9 — Site, the batch that mattered most: one row per location
    # that has a facility name. This is the loop that was 172 rows deep
    # for a single trial in this corpus.
    site_rows = []
    for r in records:
        for loc in r["locations"]:
            country = (loc.get("country") or "").strip()
            facility = (loc.get("facility") or "").strip()
            if country and facility:
                site_rows.append({
                    "nctId": r["trial"]["nctId"], "facility": facility,
                    "country": country, "city": loc.get("city"),
                    "zip": loc.get("zip"), "lat": loc.get("lat"),
                    "lon": loc.get("lon"), "key": normalise(facility),
                })
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (s:Site {facility: row.facility})
        SET s.city = row.city, s.zip = row.zip, s.lat = row.lat, s.lon = row.lon,
            s.key = row.key, s.source = 'registry'
        WITH s, row
        MATCH (t:Trial {nctId: row.nctId})
        MATCH (c:Country {name: row.country})
        MERGE (t)-[:LOCATED_AT]->(s)
        MERGE (s)-[:IN_COUNTRY]->(c)
    """, site_rows)

    # STEP 10 — Outcome, primary and secondary combined, one row each.
    # MERGEd on (measure, nctId) exactly as before: the same measure name
    # means something different per trial, so the trial-scoped key stays.
    outcome_rows = []
    for r in records:
        for outcome in r["primary_outcomes"] + r["secondary_outcomes"]:
            measure = (outcome.get("measure") or "").strip()
            if measure:
                outcome_rows.append({
                    "nctId": r["trial"]["nctId"], "measure": measure,
                    "description": outcome.get("description", ""),
                    "timeFrame": outcome.get("timeFrame", ""),
                    "type": outcome.get("type"), "key": normalise(measure),
                })
    _run_batched(session, """
        UNWIND $batch AS row
        MATCH (t:Trial {nctId: row.nctId})
        MERGE (o:Outcome {measure: row.measure, nctId: row.nctId})
        SET o.description = row.description, o.timeFrame = row.timeFrame,
            o.type = row.type, o.key = row.key, o.source = 'registry'
        MERGE (t)-[:MEASURES]->(o)
    """, outcome_rows)

    # STEP 11 — PatientPopulation. Deliberately no `.key` — see the
    # original module's own note: it has no display name a question would
    # name, and giving it one risks colliding with Trial's own key (both
    # would normalise from the same nctId).
    population_rows = [{"nctId": r["trial"]["nctId"],
                        "props": {k: v for k, v in r["patient_population"].items() if v}}
                       for r in records if any(r["patient_population"].values())]
    _run_batched(session, """
        UNWIND $batch AS row
        MATCH (t:Trial {nctId: row.nctId})
        MERGE (p:PatientPopulation {nctId: row.nctId})
        SET p += row.props, p.source = 'registry'
        MERGE (t)-[:ENROLLS]->(p)
    """, population_rows)

    # STEP 12 — MeSHTerm, the controlled vocabulary connecting trials that
    # describe the same condition differently, without fuzzy matching.
    mesh_rows = [{"nctId": r["trial"]["nctId"], "term": t.strip(), "key": normalise(t.strip())}
                for r in records for t in r["mesh_terms"] if t and t.strip()]
    _run_batched(session, """
        UNWIND $batch AS row
        MERGE (m:MeSHTerm {term: row.term})
        SET m.key = row.key, m.source = 'registry'
        WITH m, row MATCH (t:Trial {nctId: row.nctId})
        MERGE (t)-[:INDEXED_AS]->(m)
    """, mesh_rows)
