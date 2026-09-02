"""Measuring extraction against the registry.

Having both layers makes something possible that most graph pipelines cannot do:
score the extractor. For every field the registry covers, it is ground truth — so
extracting that field from the PDF and comparing gives a real number for a step
nobody usually measures.

This is not used to build the graph. It exists to answer "how much should I trust
layer 3", and the answer is a percentage rather than an impression.

The comparison is deliberately run on fields the production schema does *not*
extract. That is the point: measure the extractor where the answer is known, then
carry the result across to the fields where it is not.
"""

import json
import os

from . import config, registry

# Fields the registry holds and a model could plausibly read off the document. The
# production schema excludes all of these — they are extracted here only to be
# scored against a known answer.
PROBE_FIELDS = {
    "sponsor":    "the organisation sponsoring or funding the trial",
    "phase":      "the trial phase, as Phase 1, Phase 2, Phase 3 or Phase 4",
    "condition":  "the primary disease or condition being studied",
    "enrollment": "the planned or actual number of participants, as a number",
}

PROBE_PROMPT = """Read this excerpt from a clinical trial protocol and report only
what it states.

Fields:
{fields}

Return JSON only, with no prose and no code fences. Use null for any field the
excerpt does not state. Do not infer, and do not use knowledge of the trial from
outside this text.

{{"sponsor": "...", "phase": "...", "condition": "...", "enrollment": null}}
"""


def _client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def probe_document(text: str) -> dict:
    """Extract the probe fields from a document's opening text.

    Given the opening rather than a single chunk, because that is where a protocol
    states its sponsor and phase — and giving the extractor its best chance is the
    honest way to measure it. A low score on favourable input means something.
    """
    fields = "\n".join(f"  {name}: {desc}" for name, desc in PROBE_FIELDS.items())
    response = _client().chat.completions.create(
        model=config.EXTRACT_MODEL, temperature=0, seed=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROBE_PROMPT.format(fields=fields)},
            {"role": "user", "content": text[:12000]},
        ],
    )
    return json.loads(response.choices[0].message.content)


def normalise_value(field: str, value) -> str:
    """Comparable form of a value, so scoring measures agreement not formatting.

    "Phase 3", "PHASE3" and "Phase III" are the same answer. Without this the score
    measures whether the model matched the registry's punctuation, which is not the
    question being asked.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    if field == "phase":
        # registry.parse_trial joins multiple phases with ", " (comma-space) — see
        # registry.py's `"phase": ", ".join(...)`. Without stripping the comma here
        # too, a registry value like "PHASE1, PHASE2" and a model's equally correct
        # "Phase 1/2" — literally how this corpus's own NSCLC trial titles itself —
        # normalise to "1, 2" and "1 2" respectively: different strings, scored as
        # a mismatch for a combination-phase trial that was extracted correctly.
        text = (text.replace("phase", "").replace("/", " ").replace("-", " ")
                    .replace(",", " ").replace(" and ", " ")
                    .replace("iv", "4").replace("iii", "3").replace("ii", "2")
                    .replace("i", "1"))
        return " ".join(sorted(text.split()))
    if field == "enrollment":
        digits = "".join(c for c in text if c.isdigit())
        return digits
    # Sponsor and condition: drop punctuation and corporate suffixes.
    for noise in (",", ".", " inc", " llc", " ltd", " plc", " corporation", " co"):
        text = text.replace(noise, " ")
    return " ".join(text.split())


def compare(extracted: dict, record: dict) -> dict:
    """One document's extracted fields against the registry record.

    Records `exact`, `partial` and `missing` rather than a bare true/false. A model
    that answers "Gilead" where the registry says "Gilead Sciences" is not wrong in
    the way that answering "Pfizer" is wrong, and collapsing the two hides the
    difference between a formatting gap and a factual error.
    """
    truth = {
        "sponsor": record.get("lead_sponsor"),
        "phase": record["trial"].get("phase"),
        "condition": (record["conditions"] or [None])[0],
        "enrollment": record["trial"].get("enrollmentCount"),
    }

    results = {}
    for field in PROBE_FIELDS:
        got = normalise_value(field, extracted.get(field))
        want = normalise_value(field, truth.get(field))
        if not want:
            verdict = "no ground truth"
        elif not got:
            verdict = "missing"
        elif got == want:
            verdict = "exact"
        elif got in want or want in got:
            verdict = "partial"
        else:
            verdict = "wrong"
        results[field] = {"extracted": extracted.get(field),
                          "registry": truth.get(field), "verdict": verdict}
    return results


def score(documents: list[dict], verbose: bool = True) -> dict:
    """Score extraction across documents, per field.

    `documents` is a list of {doc_id, nct_id, text}. Documents whose trial is not in
    the registry are skipped — there is nothing to score against.
    """
    from collections import Counter

    per_field: dict[str, Counter] = {field: Counter() for field in PROBE_FIELDS}
    rows = []

    for document in documents:
        raw = registry.fetch_trial(document["nct_id"])
        if raw is None:
            continue
        record = registry.parse_trial(raw)
        extracted = probe_document(document["text"])
        comparison = compare(extracted, record)

        for field, result in comparison.items():
            per_field[field][result["verdict"]] += 1
        rows.append({"doc_id": document["doc_id"], **{
            field: result["verdict"] for field, result in comparison.items()}})

        if verbose:
            print(f"  {document['nct_id']}  " + "  ".join(
                f"{field}={result['verdict']}" for field, result in comparison.items()),
                flush=True)

    summary = {}
    for field, counts in per_field.items():
        total = sum(counts.values())
        scored = total - counts["no ground truth"]
        summary[field] = {
            "exact": counts["exact"],
            "partial": counts["partial"],
            "wrong": counts["wrong"],
            "missing": counts["missing"],
            # Partial counts as half. A near-miss is not a correct answer, and it is
            # not the same failure as a wrong one.
            "accuracy": round((counts["exact"] + 0.5 * counts["partial"])
                              / scored, 3) if scored else None,
        }

    if verbose:
        print(f"\n{'field':<14}{'exact':>7}{'partial':>9}{'wrong':>7}"
              f"{'missing':>9}{'accuracy':>10}")
        for field, result in summary.items():
            print(f"{field:<14}{result['exact']:>7}{result['partial']:>9}"
                  f"{result['wrong']:>7}{result['missing']:>9}"
                  f"{(result['accuracy'] if result['accuracy'] is not None else '-'):>10}")

    return {"summary": summary, "rows": rows}
