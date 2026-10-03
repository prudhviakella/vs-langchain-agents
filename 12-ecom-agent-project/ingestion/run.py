"""Run the e-commerce NLQ pipeline on your machine — one command per step.

    cd ingestion         every command below runs from this folder (the pipeline's root)

    data/raw/*.csv
        │  python run.py prepare            (local only — no cloud access needed)
        │     1  data_prep/fix_data.py        raw → data/clean (+ changes/, CHANGES log)
        │     2  data_prep/validate_data.py   48 rules must pass, or it stops
        │     3  data_prep/gen_dictionary.py  data/clean/DATA_DICTIONARY.md
        │     4  kb_feed/load_duckdb.py       data/kb_build.duckdb (local SQL engine)
        │     5  kb_feed/build_kb_feed.py     data/kb_feed_ecom.xlsx (every example executed)
        ▼
        │  python run.py deploy             create RDS PostgreSQL (public, your IP only) and load it
        │  python run.py postgres           reload Postgres — the deployed one (deployment.json) or PG* in .env
        │  python run.py pinecone           kb_feed → Pinecone (PINECONE_* + AWS credentials)
        │  python run.py pinecone --dry-run build and check the records, send nothing
        │  python run.py test               offline tests of the Pinecone pipeline
        │  python run.py all                prepare → postgres → pinecone

SETTINGS
    Read from .env in this folder (copy .env.example), then from the environment.
    A variable already set in the environment wins over .env.

WHAT THIS DOES NOT DO
    It does not install anything: run `pip install -r requirements.txt` once in
    a virtual environment. It does not create the Postgres database or the
    Pinecone account — see README.md.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW, CLEAN = ROOT / "data" / "raw", ROOT / "data" / "clean"
DUCKDB, FEED = ROOT / "data" / "kb_build.duckdb", ROOT / "data" / "kb_feed_ecom.xlsx"
DEPLOYMENT = ROOT / "postgres" / "deployment.json"


# import name -> pip name, per command; checked BEFORE any step runs
NEEDS = {
    "prepare":  {"pandas": "pandas", "numpy": "numpy", "duckdb": "duckdb", "openpyxl": "openpyxl"},
    "deploy":   {"boto3": "boto3", "psycopg": "psycopg[binary]", "openpyxl": "openpyxl"},
    "postgres": {"psycopg": "psycopg[binary]", "openpyxl": "openpyxl", "boto3": "boto3"},
    "pinecone": {"openpyxl": "openpyxl", "pinecone": "pinecone", "boto3": "boto3"},
    "test":     {"pytest": "pytest", "openpyxl": "openpyxl", "moto": "moto"},
}
NEEDS["all"] = {**NEEDS["prepare"], **NEEDS["postgres"], **NEEDS["pinecone"]}


def check_packages(command: str) -> None:
    """Stop before step 1 — not three steps in — if a package is missing."""
    import importlib.util
    missing = sorted({pip for mod, pip in NEEDS.get(command, {}).items() if importlib.util.find_spec(mod) is None})
    if missing:
        raise SystemExit(f"missing packages for `run.py {command}`: {', '.join(missing)}\n"
                         f"install them into this environment ({sys.executable}):\n"
                         f"  pip install -r requirements.txt")


def load_env() -> None:
    """KEY=VALUE lines from .env; the real environment wins."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def safe_console() -> None:
    """Never crash on a character the console cannot show.

    Windows writes redirected or piped output (`run.py prepare > log.txt`) in
    cp1252, which has no ━ ✓ ✗ — print() would raise UnicodeEncodeError. Keep
    the console's encoding, print '?' for what it lacks, and give every script
    this runs the same rule through PYTHONIOENCODING.
    """
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", f"{sys.stdout.encoding}:replace")


def step(title: str, script: str, *args, cwd: Path = ROOT) -> None:
    print(f"\n━━ {title} ━━")
    result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"\n✗ stopped: {title} failed (exit {result.returncode})")


def prepare() -> None:
    step("1 fix the raw data", "data_prep/fix_data.py", RAW, CLEAN)
    step("2 validate the clean data", "data_prep/validate_data.py", CLEAN)
    step("3 data dictionary", "data_prep/gen_dictionary.py", CLEAN, CLEAN / "DATA_DICTIONARY.md")
    step("4 local DuckDB", "kb_feed/load_duckdb.py", CLEAN, DUCKDB)
    step("5 kb_feed workbook", "kb_feed/build_kb_feed.py", "--db", DUCKDB, "--out", FEED)


def deploy(extra: list[str]) -> None:
    step("Deploy RDS PostgreSQL and load", "postgres/deploy.py", *extra)


def postgres() -> None:
    # After `run.py deploy`, reload into that instance: address from deployment.json,
    # credentials from its RDS-managed secret. PG* in .env still win if set.
    if DEPLOYMENT.exists() and not os.environ.get("PGHOST"):
        import json
        info = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
        os.environ.update(PGHOST=info["host"], PGPORT=str(info["port"]), PGDATABASE=info["database"],
                          AWS_REGION=info["region"], AWS_DEFAULT_REGION=info["region"])
        os.environ.setdefault("PG_SECRET_ID", info["secret_arn"])
        print(f"using the deployed instance {info['instance']} (deployment.json)")
    missing = [k for k in ("PGHOST", "PGDATABASE", "PGUSER") if not os.environ.get(k)] if not os.environ.get("PG_SECRET_ID") else []
    if missing:
        raise SystemExit(f"Postgres settings missing: {missing} — set them in .env (or PG_SECRET_ID)")
    args = ["--data", CLEAN, "--kb-feed", FEED, "--check-examples"]
    if os.environ.get("PG_SECRET_ID"):
        args += ["--secret-id", os.environ["PG_SECRET_ID"]]
    step("Postgres load", "postgres/load_postgres.py", *args)


def pinecone(dry_run: bool) -> None:
    args = [FEED]
    if dry_run:
        args.append("--dry-run")
    else:
        if not (os.environ.get("PINECONE_API_KEY") or os.environ.get("PINECONE_SECRET_ID")):
            raise SystemExit("Pinecone key missing: set PINECONE_API_KEY or PINECONE_SECRET_ID in .env")
        args += ["--index", os.environ.get("PINECONE_INDEX", "ecom-kb"),
                 "--bedrock-region", os.environ.get("AWS_REGION", "us-east-1"),
                 "--pinecone-region", os.environ.get("PINECONE_REGION", "us-east-1")]
        if os.environ.get("PINECONE_SECRET_ID"):
            args += ["--pinecone-secret-id", os.environ["PINECONE_SECRET_ID"]]
        if os.environ.get("KB_MODE"):
            args += ["--mode", os.environ["KB_MODE"]]
    step("Pinecone publish" + (" (dry run)" if dry_run else ""), "pinecone_kb/kb_to_pinecone.py", *args)


def test() -> None:
    print("\n━━ Pinecone pipeline tests (offline) ━━")
    env = {**os.environ, "FEED": str(FEED)}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_kb_to_pinecone.py"],
                            cwd=ROOT / "pinecone_kb", env=env)
    if result.returncode != 0:
        raise SystemExit("✗ tests failed")


def main() -> None:
    safe_console()
    load_env()
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    check_packages(command)
    dry = "--dry-run" in sys.argv
    if command == "prepare":
        prepare()
    elif command == "deploy":
        deploy(sys.argv[2:])
    elif command == "postgres":
        postgres()
    elif command == "pinecone":
        pinecone(dry)
    elif command == "test":
        test()
    elif command == "all":
        prepare(); postgres(); pinecone(dry)
    else:
        print(__doc__)
        return
    print("\n✓ done")


if __name__ == "__main__":
    main()
