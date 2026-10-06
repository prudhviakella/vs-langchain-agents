#!/usr/bin/env python3
"""Run resolve_trial against the REAL Neo4j graph, before deploying it.

    python check_resolve.py
    python check_resolve.py "IMbrave150" "the glaucoma trial"

    trial-graph/neo4j secret ──► lambda_tools/handler.resolve_trial(name)
                                    │  the exact Cypher the Lambda runs
                                    v
                                 candidates: nct_id · doc_id · title · score

WHY THIS EXISTS

The unit tests replay resolve_trial's graph walk in Python over the graph
export. They prove the shaping, not the Cypher. This runs the Cypher itself
against Aura, so a syntax error or a wrong label shows up here, not as an
"ERROR from resolve_trial" in an analyst's answer.

WHAT TO LOOK FOR

    every name     -> at least one candidate, the right trial first
    each candidate -> a doc_id (NONE means the trial has no protocol ingested)
    a nonsense name -> no candidates. If it still matches trials, the index
                       is on the old analyzer: run trial_graph/setup_neo4j.py
    an error naming trial_entity_names -> run trial_graph/setup_neo4j.py once

WHAT THIS DOES NOT DO

    It does not call the Lambda, the Gateway or the agent. It imports the
    handler module and calls the function directly, with your AWS credentials.
"""
import os
import sys
from pathlib import Path

# The handler imports its sibling modules by bare name, as it does in the
# deployed zip — the folder goes on sys.path first.
sys.path.insert(0, str(Path(__file__).parent / "lambda_tools"))
os.environ.setdefault("NEO4J_SECRET_ID", "trial-graph/neo4j")

import handler  # noqa: E402

DEFAULT_NAMES = ["IMbrave150", "NCT03434379", "the glaucoma trial", "STEP 1",
                 "COVID-19 vaccine", "semaglutide", "a name that matches nothing xyz"]


def main() -> None:
    names = sys.argv[1:] or DEFAULT_NAMES
    for name in names:
        result = handler.resolve_trial({"name": name})
        print(f"\n{name!r}")
        if result.get("error"):
            print(f"  ERROR: {result['detail']}")
            continue
        if not result["candidates"]:
            print("  no candidates")
        for c in result["candidates"]:
            via = f"  via {c['matched_conditions']}" if c["matched_conditions"] else ""
            print(f"  {c['score']:>6}  {c['nct_id']:<12} {c['doc_id'] or 'NONE':<45} "
                  f"{(c['acronym'] or '')[:12]:<12} {c['title'][:50]}{via}")
        if result.get("dropped_weak"):
            print(f"  ({result['dropped_weak']} weak match(es) under the relative floor dropped)")
        if result["truncated"]:
            print("  (more may match — the name is broad)")


if __name__ == "__main__":
    main()
