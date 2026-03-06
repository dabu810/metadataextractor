"""Amazon Redshift connector – inherits Postgres with Redshift-specific overrides."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .postgres import PostgresConnector
from ..config import DBConfig


class RedshiftConnector(PostgresConnector):
    """
    Redshift uses the PostgreSQL wire protocol so most logic is inherited.
    We override size/stats queries to use Redshift-specific system tables.
    """

    def connect(self) -> None:
        # Try redshift_connector first, fall back to psycopg2
        try:
            import redshift_connector
            if self._config.connection_string:
                raise ValueError("Use host/port/db for Redshift native connector")
            self._conn = redshift_connector.connect(
                host=self._config.host,
                port=self._config.port or 5439,
                database=self._config.database,
                user=self._config.username,
                password=self._config.password,
                **self._config.extra,
            )
            self._conn.autocommit = True
            self._cur = self._conn.cursor()
            # Monkey-patch fetchall to return dicts
            _orig = self._cur.fetchall

            def _dict_fetchall():
                cols = [d[0] for d in self._cur.description]
                return [dict(zip(cols, r)) for r in _orig()]

            self._cur.fetchall = _dict_fetchall  # type: ignore
        except ImportError:
            super().connect()

    def get_row_count(self, schema: str, table: str) -> int:
        schema = schema or "public"
        val = self.execute_scalar(
            "SELECT tbl_rows FROM svv_table_info "
            f"WHERE schema = '{schema}' AND \"table\" = '{table}'"
        )
        if val:
            return int(val)
        return int(self.execute_scalar(f"SELECT COUNT(*) FROM {self._fqn(schema, table)}") or 0)

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        schema = schema or "public"
        val = self.execute_scalar(
            "SELECT size * 1024 * 1024 FROM svv_table_info "
            f"WHERE schema = '{schema}' AND \"table\" = '{table}'"
        )
        return int(val) if val else None

    def get_partition_columns(self, schema: str, table: str) -> List[str]:
        schema = schema or "public"
        rows = self.execute(
            "SELECT \"column\" FROM svv_table_info "
            f"WHERE schema = '{schema}' AND \"table\" = '{table}'"
        )
        # Redshift doesn't expose sort/dist keys the same way; return dist key
        dist = self.execute(
            "SELECT column_name FROM svv_columns "
            "WHERE table_schema = %s AND table_name = %s "
            "AND distkey = 'true'",
            (schema, table),
        )
        return [r["column_name"] for r in dist]

    def _sample_clause(self, n: int) -> str:
        # Redshift supports LIMIT but not TABLESAMPLE
        return f"LIMIT {n}"
