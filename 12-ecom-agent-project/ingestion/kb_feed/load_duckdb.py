"""Load the cleaned CSVs into a local DuckDB file — what build_kb_feed.py measures and runs SQL on.

    data/clean/*.csv (14) ──► kb_build.duckdb, schema "ecom", one table per CSV
                              ZIP codes and card digits as text; timestamps typed

    cd ingestion
    python kb_feed/load_duckdb.py data/clean data/kb_build.duckdb

WHY DUCKDB
    It runs in-process: no server, no install beyond `pip install duckdb`. The
    feed builder needs a SQL engine to measure every column and to execute the
    61 examples before writing them; this file is that engine, rebuilt from the
    CSVs on every run.

WHAT THIS DOES NOT DO
    It is not the agent's database — that is RDS Postgres (postgres/).
    The file is a throwaway build artifact, recreated from scratch each time.
"""
import sys
from pathlib import Path

import duckdb

TABLES = ["categories", "customer_addresses", "customers", "inventory", "order_items", "orders",
          "payment_transactions", "product_suppliers", "products", "promotions", "reviews",
          "shipments", "suppliers", "warehouses"]
TEXT = {"zip_code", "shipping_zip", "card_last_four"}       # keep leading zeros


def main(clean: str, db: str) -> None:
    clean_dir, db_path = Path(clean), Path(db)
    # DuckDB names the database after the file: "ecom.duckdb" would make "ecom"
    # both the database and the schema, and every ecom.<table> ambiguous.
    if db_path.stem == "ecom":
        raise SystemExit(f"{db_path.name}: the file must not be named after the schema 'ecom' — use e.g. kb_build.duckdb")
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA ecom")
    for t in TABLES:
        path = clean_dir / f"{t}.csv"
        header = path.open(encoding="utf-8").readline().rstrip("\r\n").split(",")
        types = {c: "VARCHAR" for c in header if c in TEXT}
        opt = f", types={types!r}" if types else ""
        con.execute(f"CREATE TABLE ecom.{t} AS SELECT * FROM read_csv('{path.as_posix()}', header=true{opt}, "
                    f"timestampformat='%Y-%m-%d %H:%M:%S')")
    rows = con.execute("SELECT SUM(estimated_size) FROM duckdb_tables() WHERE schema_name = 'ecom'").fetchone()[0]
    con.close()
    print(f"loaded {len(TABLES)} tables, {rows:,} rows into {db_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
