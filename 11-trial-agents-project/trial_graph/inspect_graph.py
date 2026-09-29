"""Profile the LIVE trial graph — what the trial_graph prompt must describe.

    python inspect_graph.py            writes graph_profile.md, prints a summary

    trial-graph/neo4j secret ──► connect ──► read-only queries ──► graph_profile.md
      1  labels and node counts
      2  per label: every property — how many nodes have it, its value types,
         and its values (all of them when few, samples when many)
      3  relationships: (from)-[TYPE]->(to), counts, and their own properties
      4  per trial: how many sites, countries, outcomes, diseases, documents
      5  indexes and constraints

WHY THIS EXISTS

The prompt must describe the graph that is actually loaded, not the one the
build code intends: the builder writes Drug nodes the live graph was found
not to have, and the live graph carries `key` properties the builder never
writes. A property the prompt names but the graph lacks comes back as null —
and a null is then reported as a fact ("stateOrRegion is null").

WHAT THIS DOES NOT DO

    It writes nothing. Every query is a MATCH, SHOW or CALL db.* read.
"""
import statistics
from collections import Counter

from neo4j import GraphDatabase

from setup_neo4j import credentials

MAX_LISTED = 25          # list every value when a property has at most this many
SAMPLES = 4
TRUNC = 90
OUT = "graph_profile.md"


def _short(value) -> str:
    text = repr(value) if not isinstance(value, str) else value
    return text if len(text) <= TRUNC else text[:TRUNC] + "…"


def _type(value) -> str:
    if isinstance(value, list):
        inner = {type(v).__name__ for v in value}
        return f"list[{'|'.join(sorted(inner)) or 'empty'}]"
    return type(value).__name__


def profile(session) -> list[str]:
    out = ["# Live trial graph profile", ""]

    # STEP 1 labels and counts
    labels = sorted(r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label"))
    counts = {l: session.run(f"MATCH (n:`{l}`) RETURN count(n) AS c").single()["c"] for l in labels}
    out += ["## 1 · Labels", "", "| Label | Nodes |", "|---|---|"]
    out += [f"| {l} | {counts[l]:,} |" for l in labels]

    # STEP 2 every property of every label
    out += ["", "## 2 · Properties per label", ""]
    for label in labels:
        values: dict[str, list] = {}
        for record in session.run(f"MATCH (n:`{label}`) RETURN properties(n) AS p"):
            for key, value in record["p"].items():
                values.setdefault(key, []).append(value)
        out += [f"### {label}  ({counts[label]:,} nodes)", "",
                "| Property | Present on | Types | Values |", "|---|---|---|---|"]
        for key in sorted(values):
            vals = values[key]
            types = ", ".join(sorted({_type(v) for v in vals}))
            hashable = [str(v) for v in vals]
            distinct = Counter(hashable)
            if len(distinct) <= MAX_LISTED:
                shown = "; ".join(f"{_short(v)} ×{n}" for v, n in distinct.most_common())
            else:
                nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                span = f"min {min(nums)} · median {statistics.median(nums)} · max {max(nums)} · " if nums else ""
                shown = (f"{len(distinct):,} distinct · {span}e.g. "
                         + "; ".join(_short(v) for v, _ in distinct.most_common(SAMPLES)))
            out.append(f"| `{key}` | {len(vals):,} / {counts[label]:,} | {types} | "
                       f"{shown.replace('|', '/')} |")
        out.append("")

    # STEP 3 relationships
    out += ["## 3 · Relationships", "", "| Pattern | Count | Relationship properties |", "|---|---|---|"]
    for r in session.run("""
            MATCH (a)-[r]->(b)
            WITH labels(a)[0] AS a, type(r) AS t, labels(b)[0] AS b, r
            RETURN a, t, b, count(*) AS c, collect(DISTINCT keys(r))[..5] AS ks
            ORDER BY a, t, b"""):
        keys = sorted({k for ks in r["ks"] for k in ks})
        out.append(f"| (:{r['a']})-[:{r['t']}]->(:{r['b']}) | {r['c']:,} | {', '.join(keys) or '—'} |")

    # STEP 4 per-trial fan-out
    out += ["", "## 4 · Per trial (min · median · max)", "", "| Neighbour | Min | Median | Max |", "|---|---|---|---|"]
    for rel, target in [("LOCATED_AT", "Site"), ("CONDUCTED_IN", "Country"), ("MEASURES", "Outcome"),
                        ("TARGETS", "Disease"), ("INDEXED_AS", "MeSHTerm"), ("SPONSORED_BY", "Sponsor"),
                        ("MANAGED_BY", "CRO"), ("TESTS", "Drug")]:
        # grouped by trial: without t.nctId in the RETURN, count() would total all trials
        rows = [r["c"] for r in session.run(
            f"MATCH (t:Trial) OPTIONAL MATCH (t)-[:{rel}]->(x:{target}) "
            f"RETURN t.nctId AS id, count(x) AS c")]
        if rows:
            out.append(f"| {rel} → {target} | {min(rows)} | {statistics.median(rows)} | {max(rows)} |")
    docs = [r["c"] for r in session.run(
        "MATCH (t:Trial) OPTIONAL MATCH (d:Document)-[:ABOUT]->(t) RETURN t.nctId AS id, count(d) AS c")]
    if docs:
        out.append(f"| Document ABOUT | {min(docs)} | {statistics.median(docs)} | {max(docs)} |")

    # STEP 5 indexes and constraints
    out += ["", "## 5 · Indexes and constraints", "", "| Name | Type | Labels | Properties |", "|---|---|---|---|"]
    for r in session.run("SHOW INDEXES YIELD name, type, labelsOrTypes, properties "
                         "RETURN name, type, labelsOrTypes, properties ORDER BY name"):
        out.append(f"| {r['name']} | {r['type']} | {r['labelsOrTypes']} | {r['properties']} |")
    return out


def main() -> None:
    s = credentials()
    with GraphDatabase.driver(s["uri"], auth=(s.get("user", "neo4j"), s["password"])) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            lines = profile(session)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith(("#", "| ")) and "Property" not in l)[:3000])
    print(f"\nwrote {OUT} — send it back to build the prompt from the live graph")


if __name__ == "__main__":
    main()
