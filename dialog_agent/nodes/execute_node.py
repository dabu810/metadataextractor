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
    Supports postgres/redshift, oracle, sqlserver, bigquery.
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

    results: List[QueryResult] = []

    for q in sql_queries:
        logger.info("  Running %s: %s", q["query_id"], q["description"])
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

    state["query_results"] = results
    state["phase"] = "execute"
    return state
