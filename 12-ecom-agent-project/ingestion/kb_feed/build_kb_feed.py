"""Build kb_feed_ecom.xlsx — the e-commerce NLQ collection, in the kb_feed contract.

    kb_content.py   (tables, columns, joins — written meaning)  ─┐
    kb_glossary.py  (glossary, capability card)                  ├─► build ─► kb_feed_ecom.xlsx
    kb_examples.py  (SQL examples)                               │
    kb_build.duckdb     (cleaned data — measured facts, runs SQL)  ──┘
    HEADERS below   (the kb_feed sheets and columns, copied from the original workbook)

    cd ingestion
    python kb_feed/build_kb_feed.py --db data/kb_build.duckdb --out data/kb_feed_ecom.xlsx
    (--template <original kb_feed.xlsx> checks HEADERS against it; not needed to run)

    STEP 1  measure every column: type, enum values, samples, date format
    STEP 2  execute every SQL example; a failing or empty one stops the build
    STEP 3  write the sheets with the original's headers, dates as text
    STEP 4  check the result against the contract (keys, JSON, lists, dates)

WHAT THIS DOES NOT DO

    It does not mark examples verified = Y. The contract reserves Y for a
    human who has confirmed the query; examples are executed here and dated
    in last_validated, then await review.
"""
import json
import re
from pathlib import Path

import duckdb
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import kb_content as K
import kb_examples as X
import kb_glossary as G

# The kb_feed contract's sheets and columns, in order — copied from the original
# workbook so a local run does not need it. --template re-checks them.
HEADERS = {
    "nlc_node": ["action", "effective_date", "change_reason", "label", "description", "analyst_synonyms",
                 "answers_questions", "outbound_rels", "inbound_rels", "node_count", "source_systems", "embed_exempt"],
    "nlc_relationship": ["action", "effective_date", "change_reason", "rel_type", "source_labels", "target_labels",
                         "description", "analyst_synonyms", "answers_questions", "rel_count", "embed_exempt"],
    "nlc_property": ["action", "effective_date", "change_reason", "parent_name", "parent_kind", "property_name",
                     "data_type", "description", "analyst_synonyms", "answers_questions", "value_map", "sample_values",
                     "is_indexed", "is_unique", "is_required", "main_investigative_target", "list_element_type",
                     "median_list_size", "embed_exempt"],
    "nlq_table": ["action", "effective_date", "change_reason", "table", "description", "grain", "analyst_synonyms",
                  "answers_questions", "table_type", "row_count", "source_systems", "partition_columns",
                  "is_partitioned", "embed_exempt", "parent_kind", "parent_id", "active"],
    "nlq_column": ["action", "effective_date", "change_reason", "table", "column", "data_type", "description",
                   "analyst_synonyms", "answers_questions", "value_map", "sample_values", "main_investigative_target",
                   "embed_exempt", "active", "is_partition_col", "where_to_use", "is_enum", "enum_values", "is_date",
                   "date_format"],
    "nlq_join": ["action", "effective_date", "change_reason", "left_table", "right_table", "join_on", "cardinality",
                 "description"],
    "nlc_examples": ["action", "effective_date", "change_reason", "question", "paraphrases", "query", "pattern",
                     "fraud_scenario", "entities", "verified", "source", "owner", "last_validated"],
    "nlq_examples": ["action", "effective_date", "change_reason", "question", "paraphrases", "query", "pattern",
                     "fraud_scenario", "entities", "verified", "source", "owner", "last_validated", "tables_used",
                     "embed_exempt", "active", "dialect"],
    "examples_cross_source": ["action", "effective_date", "change_reason", "question", "paraphrases",
                              "fraud_scenario", "plan_json", "verified", "source", "owner", "last_validated"],
    "bridge": ["action", "effective_date", "change_reason", "bridge_key", "neo4j_label", "neo4j_key",
               "databricks_tables", "databricks_key", "directions", "description"],
    "glossary": ["action", "effective_date", "change_reason", "term", "category", "meaning", "synonyms", "maps_to"],
    "capability_card": ["action", "effective_date", "change_reason", "source", "description"],
}

TODAY = "2026-10-02"
REASON = "initial load — e-commerce NLQ collection"
BASE = dict(action="INSERT", effective_date=TODAY, change_reason=REASON)


def fq(table: str) -> str:
    return f"{K.SCHEMA}.{table}"


# ── STEP 1 — measured column facts ───────────────────────────────────────
def measure(con, table: str, column: str, role: str) -> dict:
    dtype = con.execute(f"SELECT data_type FROM information_schema.columns WHERE table_schema='{K.SCHEMA}' "
                        f"AND table_name='{table}' AND column_name='{column}'").fetchone()[0]
    kind = ("BOOLEAN" if dtype == "BOOLEAN" else "INTEGER" if dtype in ("BIGINT", "INTEGER") else
            "FLOAT" if dtype in ("DOUBLE", "FLOAT", "DECIMAL") else "DATE" if dtype in ("TIMESTAMP", "DATE") else "STRING")
    fmt = {"TIMESTAMP": "yyyy-MM-dd HH:mm:ss", "DATE": "yyyy-MM-dd"}.get(dtype, "")
    distinct = con.execute(f"SELECT COUNT(DISTINCT {column}) FROM {fq(table)}").fetchone()[0]
    values = [r[0] for r in con.execute(
        f"SELECT {column} FROM {fq(table)} WHERE {column} IS NOT NULL GROUP BY 1 ORDER BY COUNT(*) DESC, 1 LIMIT 60").fetchall()]
    is_enum = role in ("enum", "geo") and kind == "STRING" and distinct <= 60
    enum_values = " | ".join(sorted(str(v) for v in values)) if is_enum else ""
    # Numbers and dates: the spread (5th/25th/50th/75th/95th percentile), so the
    # scale is right — the most frequent amounts are arbitrary ties (20.28, 65.72)
    # when a typical order is ~2,400. Text and codes: the most frequent values.
    if is_enum or kind == "BOOLEAN":
        samples = ""
    elif kind in ("INTEGER", "FLOAT", "DATE"):
        q = con.execute(f"SELECT quantile_disc({column}, [0.05, 0.25, 0.5, 0.75, 0.95]) FROM {fq(table)}").fetchone()[0]
        samples = "; ".join(str(v)[:19] for v in q)
    else:
        samples = "; ".join(str(v) for v in values[:5])
    value_map = ""
    if role == "geo" and kind == "STRING":
        names = K.STATE_NAMES if column != "ip_country" else K.COUNTRY_NAMES
        present = {str(v): names[str(v)] for v in sorted(values) if str(v) in names}
        value_map = json.dumps(present, ensure_ascii=False) if present else ""
    return dict(data_type=kind, date_format=fmt, is_enum=is_enum, enum_values=enum_values,
                sample_values=samples, value_map=value_map, distinct=distinct)


def questions(title: str, label: str, role: str) -> str:
    t = title.lower()
    q = {"pk": [f"Show the {t} record for a given {label}", f"How many {t} are there?", f"Find {t} by {label}"],
         "fk": [f"Show {t} for a given {label}", f"How many {t} per {label}?", f"Which {label} has the most {t}?"],
         "measure": [f"What is the sum of {label} across {t}?", f"What is the average {label} in {t}?", f"Which {t} have the highest {label}?"],
         "count": [f"What is the {label} in {t}?", f"Which {t} have the highest {label}?", f"Show {t} with {label} above a threshold"],
         "enum": [f"What are the possible {label} values in {t}?", f"How many {t} per {label}?", f"Show {t} with a given {label}"],
         "geo": [f"How many {t} per {label}?", f"Show {t} in a given {label}", f"Which {label} has the most {t}?"],
         "date": [f"How many {t} by month of {label}?", f"Show {t} with {label} in a date range", f"What is the earliest and latest {label} in {t}?"],
         "flag": [f"How many {t} have {label} true?", f"What share of {t} have {label}?", f"Compare {t} with and without {label}"],
         "text": [f"What is the {label} of a given record in {t}?", f"Find {t} whose {label} contains a word", f"Show the {label} in {t}"],
         "code": [f"Find {t} by {label}", f"What is the {label} of a given record in {t}?", f"Show {t} with a given {label}"]}[role]
    return "; ".join(q)


def table_description(name: str, t: dict, rows: int) -> str:
    return (f"# {t['title']}\n**TABLE:** {fq(name)}\n**BUSINESS DOMAIN:** {K.DOMAIN}\n\n"
            f"## BUSINESS DESCRIPTION\n{t['description']}\n\n"
            f"## ANALYTICAL GRAIN\n{t['grain']} {rows:,} rows.\n\n"
            f"## KEY BUSINESS ENTITIES\n{t['entities']}\n\n"
            f"## KEY MEASURES AND ATTRIBUTES\n{t['measures']}\n\n"
            f"## QUERY GUIDANCE\n{t['guidance']}\n\n"
            f"## JOINS\n" + "\n".join(f"- {fq(l)} → {fq(r)} ON {on} ({card}): {d}" for l, r, on, card, d in K.JOINS if name in (l, r)))


def build_rows(con) -> dict:
    sheets = {}
    # nlq_table
    sheets["nlq_table"] = []
    for name, t in K.TABLES.items():
        n = con.execute(f"SELECT COUNT(*) FROM {fq(name)}").fetchone()[0]
        sheets["nlq_table"].append({**BASE, "table": fq(name), "description": table_description(name, t, n),
                                    "grain": t["grain"], "analyst_synonyms": t["synonyms"], "answers_questions": t["questions"],
                                    "table_type": t["type"], "row_count": n, "source_systems": t["sources"],
                                    "partition_columns": "", "is_partitioned": False, "embed_exempt": "N",
                                    "parent_kind": "", "parent_id": "", "active": True})
    # nlq_column
    sheets["nlq_column"] = []
    for name, cols in K.C.items():
        title = K.TABLES[name]["title"]
        for col, role, desc, syn, target in cols:
            m = measure(con, name, col, role)
            label = syn.split(";")[0].strip().lower()
            sheets["nlq_column"].append({
                **BASE, "table": fq(name), "column": col, "data_type": m["data_type"], "description": desc,
                "analyst_synonyms": syn, "answers_questions": questions(title, label, role),
                "value_map": m["value_map"], "sample_values": m["sample_values"], "main_investigative_target": target,
                "embed_exempt": "Y" if col in ("lat", "lon") or col.endswith(("_lat", "_lon")) else "N",
                "active": True, "is_partition_col": False, "where_to_use": K.WHERE_TO_USE[role],
                "is_enum": m["is_enum"], "enum_values": m["enum_values"], "is_date": role == "date" and m["data_type"] == "DATE",
                "date_format": m["date_format"]})
    # nlq_join
    sheets["nlq_join"] = [{**BASE, "left_table": fq(l), "right_table": fq(r), "join_on": on, "cardinality": card, "description": d}
                          for l, r, on, card, d in K.JOINS]
    # ── STEP 2 — every example executed before it is written ─────────────
    sheets["nlq_examples"] = []
    for q, para, sql, pattern, scenario, entities in X.E:
        rows = con.execute(sql).fetchall()
        if not rows:
            raise SystemExit(f"example returns no rows: {q}")
        tables = sorted(set(re.findall(rf"\b{K.SCHEMA}\.(\w+)", sql)))
        sheets["nlq_examples"].append({
            **BASE, "change_reason": f"{REASON} — executed {TODAY} against the cleaned data ({len(rows):,} rows)",
            "question": q, "paraphrases": "; ".join(para), "query": sql + ";", "pattern": pattern, "fraud_scenario": scenario,
            "entities": entities, "verified": "N", "source": "synthetic", "owner": "engineering-draft", "last_validated": TODAY,
            "tables_used": "; ".join(fq(t) for t in tables), "embed_exempt": False, "active": True, "dialect": "spark_sql"})
    sheets["glossary"] = [{**BASE, "term": t, "category": c, "meaning": m, "synonyms": s, "maps_to": mp} for t, c, m, s, mp in G.GLOSSARY]
    sheets["capability_card"] = [{**BASE, "source": src, "description": d} for src, d in G.CAPABILITY]
    return sheets


# ── Instructions ─────────────────────────────────────────────────────────
def instructions(sheets: dict) -> list:
    n = {k: len(v) for k, v in sheets.items()}
    rows = [("KB Feed — E-commerce NLQ collection", ""), ("", ""),
            ("What this file is", "THE contract for the e-commerce knowledge base, in the same format as the kb_feed workbook. The KB job ingests ONLY this format. Anything not in this file does not exist in the knowledge base."),
            ("This copy", f"INITIAL LOAD of the e-commerce NLQ collection: {n['nlq_table']} tables, {n['nlq_column']} columns, {n['nlq_join']} joins, {n['nlq_examples']} SQL examples, {n['glossary']} glossary terms, {n['capability_card']} capability entries. Every row action=INSERT, effective_date={TODAY}. Future changes are new files in the same format with INSERT/UPDATE/DELETE rows."),
            ("Collection scope", "NLQ only (SQL over the ecom schema). There is no graph: nlc_node, nlc_relationship, nlc_property, nlc_examples, bridge and examples_cross_source carry headers only. Table names are written ecom.<table> — replace 'ecom' with the real catalog.schema in one pass if it differs."),
            ("Source data", "The cleaned e-commerce dataset (14 CSVs) produced by fix_data.py and checked by validate_data.py (46 rules). Measured facts in nlq_column — data_type, enum_values, sample_values, date_format, value_map — are generated from that data."),
            ("Who produces rows", "Semantic columns (description, synonyms, questions, examples, glossary) are written; structural columns (types, counts, enums, samples, formats) are measured from the data. Same contract, multiple producers."),
            ("How to submit", "Export each changed sheet as CSV (sheet name = file name) and upload to: s3://<bucket>/kb-feed/ecom/<record_type>/dt=YYYY-MM-DD/<sheet>.csv — or upload the whole workbook; the job splits sheets itself."),
            ("Rules", "Lists use semicolons, never commas (enum_values uses ' | ', as in the kb_feed workbook). value_map is strict JSON with double quotes. Dates are TEXT 'YYYY-MM-DD' — never Excel dates. UPDATE rows carry the complete row for that key. DELETE rows need only key columns + action. Do not rename, reorder, or add header columns."),
            ("Record keys", "nlq_table: table · nlq_column: table+column · nlq_join: left_table+right_table+join_on · glossary: term · nlq_examples: id derived by pipeline from the question · capability_card: id derived from the description. Every key in this file is unique."),
            ("Examples", "Every SQL example was EXECUTED against the cleaned data on the date in last_validated and returned rows; the row count is in change_reason. verified = N until a human confirms each query is correct — then set Y. SQL uses only constructs common to Spark SQL and DuckDB; dialect = spark_sql."),
            ("Field dictionary (* = required)", "")]
    fields = {
        "nlq_table": [("*table", "Fully qualified table name, e.g. ecom.orders (KEY)."), ("*description", "Markdown: business description, grain, entities, measures, query guidance, joins. Main embedded text."),
                      ("grain", "What one row represents."), ("analyst_synonyms", "Semicolon-separated."), ("answers_questions", "Semicolon-separated example questions."),
                      ("table_type", "transactional | dimension | snapshot | bridge."), ("row_count", "Rows in the cleaned data (informational)."),
                      ("source_systems", "Semicolon-separated."), ("partition_columns", "Comma-free list of partition columns; empty = not partitioned."),
                      ("is_partitioned", "TRUE | FALSE."), ("embed_exempt", "Y = metadata only, no search vector."),
                      ("parent_kind", "Unused in this collection."), ("parent_id", "Unused in this collection."), ("active", "TRUE | FALSE.")],
        "nlq_column": [("*table", "Fully qualified table (KEY part)."), ("*column", "Exact column name (KEY part)."), ("data_type", "STRING | INTEGER | FLOAT | BOOLEAN | DATE (timestamps are DATE with date_format)."),
                       ("*description", "What it is, how to use it, what it is NOT. Main embedded text."), ("analyst_synonyms", "Semicolon-separated."), ("answers_questions", "Semicolon-separated."),
                       ("value_map", "Strict JSON decode of codes, e.g. {\"TX\": \"Texas\"}."), ("sample_values", "Semicolon-separated most frequent values (non-enum columns)."),
                       ("main_investigative_target", "Y = analysts query this directly; boosts retrieval."), ("embed_exempt", "Y = metadata only (coordinates)."), ("active", "TRUE | FALSE."),
                       ("is_partition_col", "TRUE | FALSE."), ("where_to_use", "Which SQL clauses the column belongs in."), ("is_enum", "TRUE when the column has a closed list of values."),
                       ("enum_values", "All values, ' | '-separated (enum columns only)."), ("is_date", "TRUE for date/timestamp columns."), ("date_format", "yyyy-MM-dd HH:mm:ss or yyyy-MM-dd; empty for non-dates.")],
        "nlq_join": [("*left_table", "KEY part."), ("*right_table", "KEY part."), ("*join_on", "Join columns; 'a = b' when names differ (KEY part)."), ("cardinality", "one-to-one | many-to-one | one-to-many | many-to-many."), ("*description", "When and how to use the join.")],
        "nlq_examples": [("*question", "Natural-language question — embedded."), ("paraphrases", "Semicolon-separated; each embedded."), ("*query", "SQL payload, never embedded."),
                         ("*pattern", "aggregation | join | ranking | window | subquery | signal_threshold | ratio | temporal_compare."),
                         ("fraud_scenario", "Business scenario in this collection: revenue_analysis | product_performance | customer_analytics | payments_risk | fulfilment | returns | inventory | procurement | promotions | reviews."),
                         ("entities", "Semicolon-separated tables and columns the query touches."), ("*verified", "Y only after a human confirmed the query; N until then."),
                         ("*source", "playbook | synthetic | feedback | query_log."), ("*owner", "Team or person accountable."), ("*last_validated", "YYYY-MM-DD the query last ran against the data."),
                         ("tables_used", "Semicolon-separated fully qualified tables."), ("embed_exempt", "TRUE | FALSE."), ("active", "TRUE | FALSE."), ("dialect", "spark_sql.")],
        "glossary": [("*term", "Business term (KEY, unique)."), ("*category", "metric | status | code | identifier | term."), ("*meaning", "Definition and how it maps to the data — for metrics, the formula."),
                     ("synonyms", "Semicolon-separated."), ("*maps_to", "Semicolon-separated ecom.table.column elements.")],
        "capability_card": [("*source", "playbook."), ("*description", "What the agent answers, what it cannot, and the rules it follows.")],
    }
    # Every sheet lists ALL its columns, the three common ones first — the
    # same layout as the kb_feed workbook, so the field dictionary can be read
    # sheet by sheet by the ingestion job.
    common = [("*action", "INSERT | UPDATE | DELETE. UPDATE = full-row replacement for that key. DELETE = key columns + action only."),
              ("*effective_date", "YYYY-MM-DD as TEXT. Orders changes when multiple files land."),
              ("change_reason", "Free text, optional. Shows up in the audit trail.")]
    for sheet, items in fields.items():
        rows.append((sheet, ""))
        rows += common + [i for i in items if i[0].lstrip("*") not in ("action", "effective_date", "change_reason")]
    return rows


# ── STEP 3 — write the workbook with the original's sheets and headers ───
def check_template(template: Path) -> None:
    """--template: HEADERS must equal the original workbook's sheets and columns."""
    src = openpyxl.load_workbook(template, read_only=True)
    theirs = {s: [c.value for c in next(src[s].iter_rows(max_row=1)) if c.value is not None] for s in src.sheetnames[1:]}
    if theirs != HEADERS:
        diff = sorted(s for s in set(theirs) | set(HEADERS) if theirs.get(s) != HEADERS.get(s))
        raise SystemExit(f"HEADERS differ from {template.name} in: {diff}")
    print(f"  headers match {template.name}")


def write(sheets: dict, out: Path) -> None:
    headers = HEADERS
    wb = openpyxl.Workbook()
    ins = wb.active; ins.title = "Instructions"
    head_font, head_fill = Font(name="Arial", bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1E3A5F")
    body = Font(name="Arial", size=10)
    for r, (a, b) in enumerate(instructions(sheets), start=1):
        ins.cell(r, 1, a).font = Font(name="Arial", bold=bool(a) and not a.startswith("*") and b == "" or r == 1, size=14 if r == 1 else 10)
        ins.cell(r, 2, b).font = body
        ins.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
    ins.column_dimensions["A"].width, ins.column_dimensions["B"].width = 34, 140
    for name, cols in headers.items():
        ws = wb.create_sheet(name)
        for j, h in enumerate(cols, start=1):
            c = ws.cell(1, j, h); c.font, c.fill = head_font, head_fill
        for i, row in enumerate(sheets.get(name, []), start=2):
            for j, h in enumerate(cols, start=1):
                v = row.get(h, "")
                c = ws.cell(i, j, v)
                c.font = body
                c.alignment = Alignment(wrap_text=True, vertical="top")
                if h in ("effective_date", "last_validated"):
                    c.number_format = "@"          # text, never an Excel date
        ws.freeze_panes = "A2"
        for j, h in enumerate(cols, start=1):
            longest = max([len(str(h))] + [min(len(str(r.get(h, ""))), 80) for r in sheets.get(name, [])[:200]])
            ws.column_dimensions[get_column_letter(j)].width = max(12, min(60, longest + 2))
    wb.save(out)


# ── STEP 4 — contract check on the written file ──────────────────────────
def check(out: Path) -> None:
    wb = openpyxl.load_workbook(out)
    keys = {"nlq_table": ["table"], "nlq_column": ["table", "column"], "nlq_join": ["left_table", "right_table", "join_on"],
            "glossary": ["term"], "nlq_examples": ["question"]}
    for name in wb.sheetnames[1:]:
        ws = wb[name]; h = [c.value for c in ws[1]]
        rows = [dict(zip(h, r)) for r in ws.iter_rows(min_row=2, values_only=True)]
        problems = []
        if None in h: problems.append("blank header")
        for r in rows:
            for k in ("effective_date", "last_validated"):
                if k in r and r[k] is not None and not (isinstance(r[k], str) and re.fullmatch(r"\d{4}-\d\d-\d\d", r[k])):
                    problems.append(f"{k}={r[k]!r}")
            if r.get("value_map"):
                json.loads(r["value_map"])
            for k in ("analyst_synonyms", "synonyms", "paraphrases"):
                if r.get(k) and "," in r[k] and ";" not in r[k]:
                    problems.append(f"{k} uses commas")
        if name in keys and rows:
            ids = [tuple(r[k] for k in keys[name]) for r in rows]
            if len(set(ids)) != len(ids): problems.append("duplicate keys")
        print(f"  {name:22} {len(rows):>4} rows  {'OK' if not problems else problems[:3]}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build the e-commerce kb_feed workbook")
    ap.add_argument("--db", type=Path, default=Path("data/kb_build.duckdb"), help="DuckDB file from load_duckdb.py")
    ap.add_argument("--out", type=Path, default=Path("data/kb_feed_ecom.xlsx"))
    ap.add_argument("--template", type=Path, help="original kb_feed workbook, to check HEADERS against")
    args = ap.parse_args()
    if args.template:
        check_template(args.template)
    con = duckdb.connect(str(args.db), read_only=True)
    sheets = build_rows(con)
    write(sheets, args.out)
    check(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
