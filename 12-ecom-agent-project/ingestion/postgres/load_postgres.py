"""Load the cleaned e-commerce CSVs into RDS PostgreSQL.

    connection ─► STEP 1  apply schema.sql (CREATE … IF NOT EXISTS: safe to re-run)
                  STEP 2  every CSV header matches its table's columns (fail before writing)
                  STEP 3  ONE transaction: TRUNCATE all 14 tables, COPY all 14, check row counts,
                          COMMIT — deferred foreign keys are checked here, so one bad
                          reference rolls the whole load back and the old data stays
                  STEP 4  (optional) COMMENT ON tables and columns from the KB feed
                  STEP 5  ANALYZE, summary
                  STEP 6  (optional) run every nlq_examples query from the KB feed

    cd ingestion
    python postgres/load_postgres.py --data data/clean
    python postgres/load_postgres.py --data data/clean --kb-feed data/kb_feed_ecom.xlsx --check-examples

CONNECTION — either
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, PGSSLMODE (default "require"), or
    --secret-id <Secrets Manager id> for username and password — an RDS-managed
        secret (what deploy.py creates) — with PGHOST / PGPORT / PGDATABASE for the
        rest. A secret that carries host / port / dbname itself is used as is.

WHY COPY, NOT INSERT
    COPY streams each file in one command: 120,156 order lines load in about a
    second. Row-by-row INSERTs would take minutes and need batching.

WHY THE CONNECTION IS IN AUTOCOMMIT MODE
    psycopg 3 opens a transaction implicitly at the first query of a
    non-autocommit connection; a later `with conn.transaction()` then becomes
    a SAVEPOINT inside it, and nothing commits until the connection closes.
    In autocommit mode every `with conn.transaction()` block is a real
    transaction: it commits when the block ends — which is where the deferred
    foreign keys are checked — and rolls back if anything inside raises.

WHY TRUNCATE + LOAD, NOT UPSERT
    The cleaned CSVs are a complete snapshot, not a change set. Replacing the
    contents inside one transaction is the simplest correct sync: no rows left
    behind from an older snapshot, and readers see old data until the commit.

WHAT THIS DOES NOT DO
    - It does not drop or alter tables. A changed column type needs a migration.
    - It does not create the database or the login role — the RDS master user
      (or your IaC) does that once; this script needs CREATE on the database.
    - It does not load the raw CSVs: run fix_data.py first; validate_data.py
      must pass on what you load.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg
from psycopg import sql

SCHEMA = "ecom"
TABLES = ["categories", "warehouses", "suppliers", "promotions", "customers", "customer_addresses",
          "products", "product_suppliers", "inventory", "orders", "order_items",
          "payment_transactions", "shipments", "reviews"]
HERE = Path(__file__).resolve().parent


def connect(secret_id: str | None) -> psycopg.Connection:
    """Open an autocommit connection.

    With --secret-id: username and password come from the secret, read fresh on
    every run (an RDS-managed secret rotates — every 7 days by default — so a
    copied password goes stale). An RDS-managed secret holds ONLY username and
    password; host, port and database then come from PGHOST / PGPORT /
    PGDATABASE, which deploy.py sets from the instance it created. A secret
    that does carry host/port/dbname (one you wrote yourself) wins.

    Without --secret-id: everything from the PG* variables (libpq's defaults).
    """
    sslmode = os.environ.get("PGSSLMODE", "require")
    if not secret_id:
        return psycopg.connect(sslmode=sslmode, connect_timeout=15, autocommit=True)
    import boto3
    secret = json.loads(boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)["SecretString"])
    host = secret.get("host") or os.environ.get("PGHOST")
    if not host:
        raise SystemExit(f"secret {secret_id} has no host and PGHOST is not set — "
                         "an RDS-managed secret holds only username and password")
    return psycopg.connect(host=host, port=int(secret.get("port") or os.environ.get("PGPORT", 5432)),
                           dbname=secret.get("dbname") or os.environ.get("PGDATABASE", "postgres"),
                           user=secret["username"], password=secret["password"],
                           sslmode=sslmode, connect_timeout=15, autocommit=True)


def table_columns(conn, table: str) -> list[str]:
    rows = conn.execute("SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                        (SCHEMA, table)).fetchall()
    return [r[0] for r in rows]


def csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return f.readline().rstrip("\r\n").split(",")


def load(conn, data: Path) -> dict:
    # STEP 1 — schema (idempotent), in its own transaction
    with conn.transaction():
        conn.execute((HERE / "schema.sql").read_text(encoding="utf-8"))
    print(f"[1] schema {SCHEMA}: {len(TABLES)} tables ready")

    # STEP 2 — every CSV matches its table before anything is written
    problems = []
    for t in TABLES:
        path = data / f"{t}.csv"
        if not path.exists():
            problems.append(f"{t}: {path} not found")
            continue
        header, cols = csv_header(path), table_columns(conn, t)
        if set(header) != set(cols):
            problems.append(f"{t}: CSV has {sorted(set(header) - set(cols))} not in the table; "
                            f"table has {sorted(set(cols) - set(header))} not in the CSV")
    if problems:
        raise SystemExit("[2] CSV/table mismatch — nothing loaded:\n  " + "\n  ".join(problems))
    print(f"[2] all {len(TABLES)} CSV headers match their tables")

    # STEP 3 — one transaction: replace everything, verify, commit
    counts = {}
    t0 = time.time()
    with conn.transaction():
        conn.execute("SET CONSTRAINTS ALL DEFERRED")
        conn.execute(sql.SQL("TRUNCATE {} ").format(
            sql.SQL(", ").join(sql.Identifier(SCHEMA, t) for t in TABLES)))
        with conn.cursor() as cur:
            for t in TABLES:
                path = data / f"{t}.csv"
                columns = sql.SQL(", ").join(sql.Identifier(c) for c in csv_header(path))
                copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
                    sql.Identifier(SCHEMA, t), columns)
                with path.open("rb") as f, cur.copy(copy_sql) as copy:
                    while chunk := f.read(1 << 20):
                        copy.write(chunk)
                loaded = conn.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(SCHEMA, t))).fetchone()[0]
                expected = sum(1 for _ in path.open(encoding="utf-8")) - 1
                if loaded != expected:
                    raise RuntimeError(f"{t}: loaded {loaded} rows, CSV has {expected}")
                counts[t] = loaded
        # leaving this block COMMITs — the deferred foreign keys are checked here;
        # a violation raises and the whole load rolls back
    print(f"[3] loaded {sum(counts.values()):,} rows into {len(counts)} tables in {time.time() - t0:.1f}s "
          "(one transaction; foreign keys checked at commit)")
    return counts


def comment_from_feed(conn, feed: Path) -> None:
    """STEP 4 — the KB descriptions as Postgres comments, so any SQL tool (and
    an agent reading information_schema / pg_description) sees the meaning."""
    import openpyxl
    wb = openpyxl.load_workbook(feed, read_only=True)

    def rows(name):
        it = wb[name].iter_rows(values_only=True)
        head = next(it)
        return [dict(zip(head, r)) for r in it if any(r)]

    n = 0
    with conn.transaction():
        for r in rows("nlq_table"):
            schema, table = r["table"].split(".")
            body = re.search(r"## BUSINESS DESCRIPTION\n(.*?)\n\n", r["description"] or "", re.S)
            text = f"{body.group(1).strip() if body else ''} Grain: {r['grain']}".strip()
            conn.execute(sql.SQL("COMMENT ON TABLE {} IS {}").format(sql.Identifier(schema, table), sql.Literal(text)))
            n += 1
        for r in rows("nlq_column"):
            schema, table = r["table"].split(".")
            conn.execute(sql.SQL("COMMENT ON COLUMN {} IS {}").format(
                sql.Identifier(schema, table, r["column"]), sql.Literal(r["description"])))
            n += 1
    print(f"[4] {n} table and column comments written from {feed.name}")


def check_examples(conn, feed: Path) -> int:
    """STEP 6 — every nlq_examples query must run here and return rows."""
    import openpyxl
    ws = openpyxl.load_workbook(feed, read_only=True)["nlq_examples"]
    it = ws.iter_rows(values_only=True)
    head = next(it)
    examples = [dict(zip(head, r)) for r in it if any(r)]
    failed = 0
    for e in examples:
        try:
            with conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")
                conn.execute("SET LOCAL statement_timeout = '60s'")
                rows = conn.execute(e["query"]).fetchall()
            status = f"{len(rows):>6} rows" if rows else "  EMPTY   "
            failed += not rows
        except Exception as exc:
            failed += 1
            status = f"ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
        if not status.endswith("rows"):
            print(f"    {status}  {e['question'][:70]}")
    print(f"[6] examples: {len(examples) - failed} of {len(examples)} run on PostgreSQL and return rows")
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="folder with the 14 cleaned CSVs")
    ap.add_argument("--kb-feed", type=Path, help="kb_feed workbook: table/column comments, example checks")
    ap.add_argument("--check-examples", action="store_true", help="run every nlq_examples query after loading")
    ap.add_argument("--secret-id", help="Secrets Manager id of the RDS credentials")
    args = ap.parse_args()

    try:
        conn = connect(args.secret_id)
    except psycopg.OperationalError as exc:
        where = f"{os.environ.get('PGHOST', '?')}:{os.environ.get('PGPORT', '5432')}" if not args.secret_id else args.secret_id
        raise SystemExit(f"cannot connect to Postgres at {where}: {str(exc).splitlines()[0]}\n"
                         "  check: the host and port; for RDS, that the security group allows your IP on 5432 "
                         "and the instance is publicly accessible (or you are on its VPC/VPN); the user and password")
    with conn:
        counts = load(conn, args.data)
        if args.kb_feed:
            comment_from_feed(conn, args.kb_feed)
        # STEP 5 — fresh planner statistics after replacing every row
        for t in TABLES:
            conn.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(SCHEMA, t)))
        print("[5] ANALYZE done\n" + "\n".join(f"    {t:22} {n:>8,}" for t, n in counts.items()))
        failed = check_examples(conn, args.kb_feed) if args.check_examples and args.kb_feed else 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
