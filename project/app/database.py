"""PostgreSQL access layer for the `flights` table.

`execute_sql_query` is the utility the SQL agent calls as a tool. It opens a
**read-only** session (defence-in-depth on top of the SQL guardrail) so a query
that slips through and tries to mutate data will be rejected by the server.
"""
from __future__ import annotations

from typing import Any

import psycopg2
import psycopg2.extras

from app import config

# Column list + creation DDL mirror the schema in the project brief.
FLIGHTS_COLUMNS = [
    ("id", "BIGINT PRIMARY KEY"),
    ("flight_no", "TEXT"),
    ("airline_code", "TEXT"),
    ("airline_name", "TEXT"),
    ("origin", "TEXT"),
    ("destination", "TEXT"),
    ("departure_date", "DATE"),
    ("departure_time", "TIME"),
    ("arrival_date", "DATE"),
    ("arrival_time", "TIME"),
    ("status", "TEXT"),
    ("delay_minutes", "INTEGER"),
    ("delay_reason", "TEXT"),
    ("terminal", "TEXT"),
    ("gate", "TEXT"),
    ("aircraft_type", "TEXT"),
    ("seats_total", "INTEGER"),
    ("seats_booked", "INTEGER"),
    ("fare_inr", "INTEGER"),
]

CREATE_TABLE_SQL = "CREATE TABLE IF NOT EXISTS flights (\n    " + ",\n    ".join(
    f"{name} {dtype}" for name, dtype in FLIGHTS_COLUMNS
) + "\n);"


def _connect(readonly: bool = True):
    conn = psycopg2.connect(**config.DB_PARAMS)
    # A read-only session makes accidental/hostile writes fail at the server.
    conn.set_session(readonly=readonly, autocommit=not readonly)
    return conn


def execute_sql_query(query: str) -> dict[str, Any]:
    """Execute a SELECT query and return a structured result.

    Returns a dict:
        {"ok": bool, "columns": [...], "rows": [ {col: val, ...}, ... ],
         "rowcount": int, "error": str | None}
    """
    conn = None
    try:
        conn = _connect(readonly=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall() if cur.description else []
            columns = [d.name for d in cur.description] if cur.description else []
        # Convert dates/times to str so the result is JSON-serialisable.
        clean = [{k: (str(v) if v is not None else None) for k, v in row.items()} for row in rows]
        return {"ok": True, "columns": columns, "rows": clean,
                "rowcount": len(clean), "error": None}
    except Exception as exc:  # noqa: BLE001 - surface any DB error to the caller
        return {"ok": False, "columns": [], "rows": [], "rowcount": 0,
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if conn is not None:
            conn.close()


def result_to_text(result: dict[str, Any], max_rows: int = 25) -> str:
    """Render a result dict as compact text for an LLM/agent to read."""
    if not result["ok"]:
        return f"SQL ERROR: {result['error']}"
    if result["rowcount"] == 0:
        return "No matching rows were found."
    rows = result["rows"][:max_rows]
    lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in rows]
    text = "\n".join(lines)
    if result["rowcount"] > max_rows:
        text += f"\n... ({result['rowcount'] - max_rows} more rows)"
    return text


# --------------------------------------------------------------------------
# Loader helpers (used by scripts/load_db.py — writable connection).
# --------------------------------------------------------------------------
def init_and_load(csv_path: str | None = None) -> int:
    """Create the flights table (if needed) and load the CSV. Returns row count."""
    import pandas as pd

    csv_path = str(csv_path or config.CSV_PATH)
    df = pd.read_csv(csv_path)
    # Normalise blank strings to NULL for nullable columns.
    df = df.where(df.notna(), None)

    col_names = [c for c, _ in FLIGHTS_COLUMNS]
    conn = psycopg2.connect(**config.DB_PARAMS)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute("TRUNCATE TABLE flights;")
            records = [tuple(None if (v is None or (isinstance(v, float) and pd.isna(v)))
                             else v for v in row)
                       for row in df[col_names].itertuples(index=False, name=None)]
            placeholders = "(" + ", ".join(["%s"] * len(col_names)) + ")"
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO flights ({', '.join(col_names)}) VALUES %s",
                records,
                template=placeholders,
            )
        conn.commit()
        return len(records)
    finally:
        conn.close()


def table_status() -> dict[str, Any]:
    """Lightweight status probe for /health and CLI checks."""
    try:
        conn = _connect(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM flights;")
            count = cur.fetchone()[0]
        conn.close()
        return {"ok": True, "flights": count, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "flights": 0, "error": f"{type(exc).__name__}: {exc}"}
