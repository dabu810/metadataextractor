"""
execute_node — Run each planned SQL query against the target database.

Uses a minimal inline connector (psycopg2/pyodbc/etc.) driven by the
DialogConfig.  Errors on individual queries are captured and do not abort
the whole pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..config import DialogConfig
from ..state import DialogState, QueryResult

logger = logging.getLogger(__name__)


# ── Thin DB runner ────────────────────────────────────────────────────────────

def _run_sql(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    """
    Execute *sql* and return {columns, rows, error}.
    Supports postgres/redshift, oracle, sqlserver, bigquery, sqlite, csv, excel.
    Falls back to a generic error for unsupported types.
    """
    db = cfg.db_type.lower()

    try:
        if db in ("postgres", "redshift"):
            return _run_postgres(cfg, sql)
        elif db == "oracle":
            return _run_oracle(cfg, sql)
        elif db == "sqlserver":
            return _run_sqlserver(cfg, sql)
        elif db == "bigquery":
            return _run_bigquery(cfg, sql)
        elif db in ("sqlite", "csv", "excel"):
            return _run_file_based(cfg, sql)
        else:
            return {"columns": [], "rows": [], "error": f"Unsupported db_type: {db}"}
    except Exception as exc:
        logger.exception("execute_node: SQL failed")
        return {"columns": [], "rows": [], "error": str(exc)}


def _cursor_to_result(cursor) -> Dict[str, Any]:
    columns = [desc[0] for desc in (cursor.description or [])]
    rows    = [list(r) for r in (cursor.fetchall() or [])]
    return {"columns": columns, "rows": rows, "error": None}


def _run_postgres(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    import psycopg2
    import psycopg2.extras

    if cfg.db_connection_string:
        conn = psycopg2.connect(cfg.db_connection_string)
    else:
        conn = psycopg2.connect(
            host=cfg.db_host, port=cfg.db_port or 5432,
            dbname=cfg.db_name, user=cfg.db_user, password=cfg.db_password,
            **cfg.db_extra,
        )
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Set search_path so unqualified table names resolve to the right schema.
            # This is a defence-in-depth measure: the LLM should already qualify names,
            # but this ensures the query works even if it doesn't.
            if cfg.db_schema:
                cur.execute(f"SET search_path TO {cfg.db_schema}, public")
            cur.execute(sql)
            columns = [desc[0] for desc in (cur.description or [])]
            rows    = [[row[c] for c in columns] for row in (cur.fetchall() or [])]
            return {"columns": columns, "rows": rows, "error": None}
    finally:
        conn.close()


def _run_oracle(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    import cx_Oracle

    if cfg.db_connection_string:
        conn = cx_Oracle.connect(cfg.db_connection_string)
    else:
        dsn  = cx_Oracle.makedsn(cfg.db_host, cfg.db_port or 1521,
                                  service_name=cfg.db_name)
        conn = cx_Oracle.connect(cfg.db_user, cfg.db_password, dsn)
    try:
        with conn.cursor() as cur:
            # Set current schema so unqualified names resolve correctly
            if cfg.db_schema:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {cfg.db_schema}")
            cur.execute(sql)
            return _cursor_to_result(cur)
    finally:
        conn.close()


def _run_sqlserver(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    import pyodbc

    if cfg.db_connection_string:
        conn = pyodbc.connect(cfg.db_connection_string)
    else:
        driver = cfg.db_extra.get("driver", "ODBC Driver 18 for SQL Server")
        conn = pyodbc.connect(
            f"DRIVER={{{driver}}};SERVER={cfg.db_host},{cfg.db_port or 1433};"
            f"DATABASE={cfg.db_name};UID={cfg.db_user};PWD={cfg.db_password}"
        )
    try:
        with conn.cursor() as cur:
            # Switch to the target schema so unqualified names resolve correctly
            if cfg.db_schema:
                cur.execute(f"USE [{cfg.db_name}]")
                cur.execute(f"SET SCHEMA [{cfg.db_schema}]")
            cur.execute(sql)
            return _cursor_to_result(cur)
    finally:
        conn.close()


_FILE_BASED_TYPES = {"sqlite", "csv", "excel"}


def _safe_col(name: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return ("col_" + s if s and s[0].isdigit() else s) or "col"


def _dedup_cols(df) -> None:
    """Deduplicate DataFrame column names in-place after sanitisation."""
    used: dict = {}
    new_cols = []
    for c in df.columns:
        sc = _safe_col(str(c))
        if sc in used:
            used[sc] += 1
            sc = f"{sc}_{used[sc]}"
        else:
            used[sc] = 1
        new_cols.append(sc)
    df.columns = new_cols


def _build_file_conn(cfg: DialogConfig):
    """
    Load a file-based source (SQLite / CSV / Excel) into a sqlite3 connection.
    The connection is returned open — caller is responsible for closing it.
    File is loaded ONCE so multiple SQL queries can reuse the same connection.

    CSV/Excel sources are loaded into a temp-file SQLite database (not :memory:)
    so that SQLite can spill sort/aggregation buffers to disk and never hits the
    "database or disk is full" error that `:memory:` raises on large datasets.
    """
    import re
    import sqlite3
    import tempfile
    from pathlib import Path

    db    = cfg.db_type.lower()
    fpath = cfg.db_file_path

    def _safe_table(name: str) -> str:
        """Must match understand_node._to_sql_table so table names are consistent."""
        s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
        return ("t_" + s if s and s[0].isdigit() else s) or "tbl"

    if db == "sqlite":
        conn = sqlite3.connect(fpath, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    import pandas as pd
    # Use a named temp file so SQLite can spill to disk; delete=False so it
    # persists while the connection is open, then we clean it up on close.
    _tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    _tmp.close()
    _tmp_path = _tmp.name
    conn = sqlite3.connect(_tmp_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Attach the temp path so caller can unlink it after closing
    conn._tmp_path = _tmp_path  # type: ignore[attr-defined]

    if db == "csv":
        dir_path = Path(fpath)
        for f in sorted(dir_path.glob("*.csv")):
            try:
                df = pd.read_csv(f)
                _dedup_cols(df)
                df.to_sql(f.stem, conn, if_exists="replace", index=False)
                logger.info("file_conn: loaded CSV %s (%d rows, %d cols)",
                            f.stem, len(df), len(df.columns))
            except Exception as exc:
                logger.warning("file_conn: skipping CSV %s — %s: %s",
                               f.name, type(exc).__name__, exc)
    else:  # excel
        xl = pd.ExcelFile(fpath)
        used_sheets: dict = {}
        for sheet in xl.sheet_names:
            base = _safe_table(sheet)
            if base in used_sheets:
                used_sheets[base] += 1
                safe = f"{base}_{used_sheets[base]}"
            else:
                used_sheets[base] = 1
                safe = base
            try:
                df = xl.parse(sheet)
                _dedup_cols(df)
                df.to_sql(safe, conn, if_exists="replace", index=False)
                logger.info("file_conn: loaded sheet %r → %r (%d rows, %d cols)",
                            sheet, safe, len(df), len(df.columns))
            except Exception as exc:
                logger.warning("file_conn: skipping sheet %r → %r — %s: %s",
                               sheet, safe, type(exc).__name__, exc)

    # Log what actually made it in so mismatches are visible
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    loaded = [r[0] for r in cur.fetchall()]
    cur.close()
    logger.info("file_conn: tables available: %s", loaded)
    return conn


def _exec_on_conn(conn, sql: str) -> Dict[str, Any]:
    """Run one SQL statement on an already-open sqlite3 connection."""
    import sqlite3
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in (cur.description or [])]
        rows = [list(r) for r in (cur.fetchall() or [])]
        cur.close()
        return {"columns": cols, "rows": rows, "error": None}
    except sqlite3.Error as exc:
        # Enrich "no such table" with the list of available tables
        msg = str(exc)
        if "no such table" in msg.lower():
            try:
                c2 = conn.cursor()
                c2.execute("SELECT name FROM sqlite_master WHERE type='table'")
                available = [r[0] for r in c2.fetchall()]
                c2.close()
                msg += f" (available tables: {available})"
            except Exception:
                pass
        return {"columns": [], "rows": [], "error": msg}
    except Exception as exc:
        return {"columns": [], "rows": [], "error": str(exc)}


def _run_bigquery(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    from google.cloud import bigquery

    kwargs: Dict[str, Any] = {}
    if cfg.db_extra.get("credentials_path"):
        from google.oauth2 import service_account
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            cfg.db_extra["credentials_path"]
        )
    client  = bigquery.Client(project=cfg.db_name or cfg.db_extra.get("project"), **kwargs)
    job     = client.query(sql)
    results = job.result()
    columns = [f.name for f in results.schema]
    rows    = [[r[c] for c in columns] for r in results]
    return {"columns": columns, "rows": rows, "error": None}


# ── node ──────────────────────────────────────────────────────────────────────

def execute_node(state: DialogState) -> DialogState:
    """Execute all planned SQL queries and store results."""
    logger.info("=== execute_node ===")

    config      = state["config"]
    sql_queries = state.get("sql_queries") or []

    if not sql_queries:
        logger.info("execute_node: no queries to run")
        state["query_results"] = []
        state["phase"] = "execute"
        return state

    # For file-based sources, load the file ONCE and reuse the connection for
    # every query — avoids re-parsing the entire Excel/CSV on each SQL call.
    file_conn = None
    if config.db_type.lower() in _FILE_BASED_TYPES:
        try:
            file_conn = _build_file_conn(config)
        except Exception as exc:
            logger.exception("execute_node: failed to build file connection")
            state["errors"].append(f"execute_node: could not load file — {exc}")
            state["query_results"] = []
            state["phase"] = "execute"
            return state

    results: List[QueryResult] = []

    try:
        for q in sql_queries:
            logger.info("  Running %s: %s", q["query_id"], q["description"])
            if file_conn is not None:
                outcome = _exec_on_conn(file_conn, q["sql"])
            else:
                outcome = _run_sql(config, q["sql"])

            rows      = outcome.get("rows") or []
            columns   = outcome.get("columns") or []
            error_msg: Optional[str] = outcome.get("error")

            results.append(
                QueryResult(
                    query_id    = q["query_id"],
                    description = q["description"],
                    sql         = q["sql"],
                    columns     = columns,
                    rows        = rows[: config.row_limit],
                    row_count   = len(rows),
                    error       = error_msg,
                )
            )

            if error_msg:
                logger.warning("  %s FAILED: %s", q["query_id"], error_msg)
                state["errors"].append(f"execute_node [{q['query_id']}]: {error_msg}")
            else:
                logger.info("  %s OK — %d rows returned", q["query_id"], len(rows))

    finally:
        if file_conn is not None:
            tmp_path = getattr(file_conn, "_tmp_path", None)
            try:
                file_conn.close()
            except Exception:
                pass
            # Remove the temp SQLite file created for CSV/Excel sources
            if tmp_path:
                try:
                    import os as _os
                    _os.unlink(tmp_path)
                except Exception:
                    pass

    state["query_results"] = results
    state["phase"] = "execute"
    return state
