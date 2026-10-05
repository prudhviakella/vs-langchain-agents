"""SQL guard for the NLQ agent — every statement passes here before PostgreSQL sees it.

    model's SQL ─► mask literals + comments ─► one statement? ─► SELECT / WITH? ─► no write keyword?
                                                                                       │
              GuardResult(sql with LIMIT) ◄── add LIMIT if missing ◄── no dangerous function?

Ported from the ACT NLQ guard (Databricks/Spark) to PostgreSQL. The checks run
on a MASKED copy: string literals, quoted identifiers and comments are blanked
first, so

    WHERE note = 'please DELETE me'     is not rejected (the word is data), and
    SELECT 1; /* */ DROP TABLE orders   is rejected (the comment hides nothing).

Correctness-critical: versioned with the code, never runtime-editable. The
same guard runs again in the Lambda that executes the query — the agent
cannot be the only thing standing between a model and the database.

WHAT THIS DOES NOT DO
    - It is not the only protection: the query also runs as a read-only
      database user, inside a READ ONLY transaction, with a statement timeout.
    - It does not check that tables exist — Postgres does, and its error goes
      back to the model to correct.
    - It does not rewrite a query beyond appending a LIMIT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "TRUNCATE", "DROP", "ALTER", "CREATE",
    "GRANT", "REVOKE", "COPY", "CALL", "DO", "EXECUTE", "PREPARE", "DEALLOCATE", "VACUUM",
    "ANALYZE", "REINDEX", "CLUSTER", "COMMENT", "LOCK", "SET", "RESET", "LISTEN", "NOTIFY",
    "UNLISTEN", "REFRESH", "IMPORT", "SECURITY", "DISCARD", "CHECKPOINT", "LOAD", "INTO",
)
# INTO: `SELECT … INTO new_table` creates a table. `INSERT INTO` is already caught.

FORBIDDEN_FUNCTIONS = (
    "pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "lo_get", "dblink", "pg_terminate_backend", "pg_cancel_backend",
    "set_config", "pg_reload_conf", "pg_rotate_logfile", "query_to_xml", "pg_file_write",
)

_LITERAL = re.compile(r"""
      '(?:[^']|'')*'            # 'string' with '' escapes
    | E'(?:[^'\\]|\\.|'')*'     # E'escape string'
    | "(?:[^"]|"")*"            # "quoted identifier"
    | \$(\w*)\$.*?\$\1\$        # $tag$ dollar-quoted $tag$
    | --[^\n]*                  # -- line comment
    | /\*.*?\*/                 # /* block comment */
""", re.S | re.X)


class GuardError(ValueError):
    """Query rejected. The message is written for the model to correct itself."""


@dataclass
class GuardResult:
    sql: str                                   # what will run (LIMIT appended if needed)
    limit_added: bool = False
    warnings: list[str] = field(default_factory=list)


def mask(sql: str) -> str:
    """The SQL with every literal, quoted identifier and comment replaced by a blank
    placeholder of the same kind — keywords inside them can no longer match."""
    return _LITERAL.sub(lambda m: "''" if m.group(0)[:1] in ("'", "E", "$") else " ", sql)


def _outer_limit(masked: str) -> bool:
    """True when the statement's OUTERMOST level ends with LIMIT n [OFFSET m].
    A LIMIT inside a sub-query or CTE does not bound the result."""
    depth, outer = 0, []
    for ch in masked:
        depth += ch == "("
        depth -= ch == ")"
        outer.append(ch if depth == 0 else " ")
    tail = "".join(outer).strip().rstrip(";").strip()
    return bool(re.search(r"\bLIMIT\s+(\d+|ALL)(\s+OFFSET\s+\d+)?\s*$", tail, re.I) or
                re.search(r"\bFETCH\s+(FIRST|NEXT)\s+\d+\s+ROWS?\s+ONLY\s*$", tail, re.I))


def guard(sql: str, row_cap: int = 1000) -> GuardResult:
    """Validate a statement; append a LIMIT when the outer query has none.

    STEP 1  non-empty, ONE statement (a trailing ; is fine)
    STEP 2  starts with SELECT or WITH
    STEP 3  no write / admin keyword anywhere outside literals and comments
    STEP 4  no dangerous function
    STEP 5  LIMIT row_cap appended if the outer query has none
    """
    text = (sql or "").strip()
    if not text:
        raise GuardError("Empty query. Write one SELECT statement.")
    masked = mask(text)

    # STEP 1
    body = masked.strip().rstrip(";").strip()
    if ";" in body:
        raise GuardError("Only ONE statement per call. Remove the extra statement(s); "
                         "run them as separate execute_sql calls if you need both.")
    # STEP 2
    first = re.match(r"\s*\(*\s*(\w+)", body)
    if not first or first.group(1).upper() not in ("SELECT", "WITH"):
        raise GuardError("Read-only access: the statement must start with SELECT or WITH.")
    # STEP 3
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", body, re.I):
            raise GuardError(f"'{kw}' is not allowed: this is a read-only analytics connection. "
                             "Rewrite the query as a plain SELECT.")
    # STEP 4
    for fn in FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{fn}\s*\(", body, re.I):
            raise GuardError(f"The function {fn}() is not allowed. Rewrite the query without it.")
    # STEP 5
    result = GuardResult(sql=text.rstrip().rstrip(";").rstrip())
    if not _outer_limit(masked):
        result.sql = f"{result.sql}\nLIMIT {row_cap}"
        result.limit_added = True
        result.warnings.append(f"LIMIT {row_cap} added: the query had no outer LIMIT")
    return result
