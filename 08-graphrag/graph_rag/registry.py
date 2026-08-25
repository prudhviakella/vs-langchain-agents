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
"""

import json
import time
from pathlib import Path

from . import config

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


def load_trial(session, record: dict) -> None:
    """Write one parsed trial and everything it connects to.

    Every node carries `source: "registry"`, which is what separates these facts
    from the claims layer 3 extracts. A query can then ask for one, the other, or
    both — and extraction accuracy can be measured against these.
    """
    trial = record["trial"]
    nct_id = trial["nctId"]
    if not nct_id:
        return

    session.run("""
        MERGE (t:Trial {nctId: $nctId})
        SET t += $props, t.source = 'registry'
    """, nctId=nct_id, props={k: v for k, v in trial.items() if k != "nctId"})

    # Coarse grouping from the first listed condition, for top-level browsing.
    # Granular targeting is the Disease nodes below.
    category = record["conditions"][0] if record["conditions"] else "Unknown"
    session.run("""
        MERGE (c:TrialCategory {name: $category}) SET c.source = 'registry'
        WITH c MATCH (t:Trial {nctId: $nctId})
        MERGE (t)-[:BELONGS_TO]->(c)
    """, category=category, nctId=nct_id)

    for condition in record["conditions"]:
        if condition and condition.strip():
            session.run("""
                MERGE (d:Disease {name: $name}) SET d.source = 'registry'
                WITH d MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:TARGETS]->(d)
            """, name=condition.strip(), nctId=nct_id)

    for intervention in record["interventions"]:
        name = (intervention.get("name") or "").strip()
        if name:
            session.run("""
                MERGE (d:Drug {name: $name})
                SET d.type = $type, d.otherNames = $otherNames, d.source = 'registry'
                WITH d MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:TESTS]->(d)
            """, name=name, type=intervention.get("type"),
                 otherNames=intervention.get("otherNames", []), nctId=nct_id)

    if record["lead_sponsor"]:
        session.run("""
            MERGE (s:Sponsor {name: $name}) SET s.source = 'registry'
            WITH s MATCH (t:Trial {nctId: $nctId})
            MERGE (t)-[:SPONSORED_BY]->(s)
        """, name=record["lead_sponsor"].strip(), nctId=nct_id)

    # Not every collaborator is a contract research organisation, but in registered
    # trials most are, and the separate label makes the operational-versus-financial
    # distinction queryable without a free-text search.
    for name in record["collaborators"]:
        if name and name.strip():
            session.run("""
                MERGE (c:CRO {name: $name}) SET c.source = 'registry'
                WITH c MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:MANAGED_BY]->(c)
            """, name=name.strip(), nctId=nct_id)

    for location in record["locations"]:
        country = (location.get("country") or "").strip()
        facility = (location.get("facility") or "").strip()
        if not country:
            continue
        session.run("""
            MERGE (c:Country {name: $country}) SET c.source = 'registry'
            WITH c MATCH (t:Trial {nctId: $nctId})
            MERGE (t)-[:CONDUCTED_IN]->(c)
        """, country=country, nctId=nct_id)
        if facility:
            session.run("""
                MERGE (s:Site {facility: $facility})
                SET s.city = $city, s.zip = $zip, s.lat = $lat, s.lon = $lon,
                    s.source = 'registry'
                WITH s
                MATCH (t:Trial {nctId: $nctId})
                MATCH (c:Country {name: $country})
                MERGE (t)-[:LOCATED_AT]->(s)
                MERGE (s)-[:IN_COUNTRY]->(c)
            """, facility=facility, city=location.get("city"), zip=location.get("zip"),
                 lat=location.get("lat"), lon=location.get("lon"),
                 nctId=nct_id, country=country)

    # Outcomes are MERGEd on (measure, trial) rather than measure alone: "Overall
    # Survival" means something different in each trial, so a shared node would
    # collapse unrelated endpoints into one.
    for outcome in record["primary_outcomes"] + record["secondary_outcomes"]:
        measure = (outcome.get("measure") or "").strip()
        if measure:
            session.run("""
                MATCH (t:Trial {nctId: $nctId})
                MERGE (o:Outcome {measure: $measure, nctId: $nctId})
                SET o.description = $description, o.timeFrame = $timeFrame,
                    o.type = $type, o.source = 'registry'
                MERGE (t)-[:MEASURES]->(o)
            """, measure=measure, nctId=nct_id,
                 description=outcome.get("description", ""),
                 timeFrame=outcome.get("timeFrame", ""), type=outcome.get("type"))

    population = record["patient_population"]
    if any(population.values()):
        session.run("""
            MATCH (t:Trial {nctId: $nctId})
            MERGE (p:PatientPopulation {nctId: $nctId})
            SET p += $props, p.source = 'registry'
            MERGE (t)-[:ENROLLS]->(p)
        """, nctId=nct_id, props={k: v for k, v in population.items() if v})

    # The controlled vocabulary. This is what connects trials that describe the same
    # condition differently, without any fuzzy matching.
    for term in record["mesh_terms"]:
        if term and term.strip():
            session.run("""
                MERGE (m:MeSHTerm {term: $term}) SET m.source = 'registry'
                WITH m MATCH (t:Trial {nctId: $nctId})
                MERGE (t)-[:INDEXED_AS]->(m)
            """, term=term.strip(), nctId=nct_id)


def load_trials(session, nct_ids: list[str], verbose: bool = True) -> dict:
    """Fetch, parse and load every trial. Missing ones are reported, not fatal."""
    loaded, missing = [], []
    for nct_id in nct_ids:
        raw = fetch_trial(nct_id)
        if raw is None:
            missing.append(nct_id)
            continue
        record = parse_trial(raw)
        load_trial(session, record)
        loaded.append(nct_id)
        if verbose:
            trial = record["trial"]
            print(f"  {nct_id}  {trial['phase'] or 'n/a':<16} "
                  f"{(record['lead_sponsor'] or '')[:34]:<36} "
                  f"{len(record['locations'])} sites", flush=True)

    if verbose and missing:
        print(f"\n  not found: {', '.join(missing)}", flush=True)
    return {"loaded": loaded, "missing": missing}
