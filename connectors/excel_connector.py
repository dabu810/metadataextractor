"""Excel connector — loads each sheet of an .xlsx/.xls file into in-memory SQLite."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from ..config import DBConfig

_VALID_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}


def _safe_name(name: str) -> str:
    """Convert a sheet name to a valid SQL table name."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "sheet_" + name
    return name or "sheet"


class ExcelConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._sheets: List[str] = []

    def _file_path(self) -> Path:
        return Path(self._config.file_path or self._config.database or "")

    def connect(self) -> None:
        import pandas as pd

        path = self._file_path()
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        if path.suffix.lower() not in _VALID_EXTENSIONS:
            raise ValueError(
                f"Not a supported Excel file (got '{path.suffix}'). "
                f"Expected one of: {', '.join(sorted(_VALID_EXTENSIONS))}"
            )

        xl = pd.ExcelFile(path)
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._sheets = []

        used: dict = {}
        for sheet in xl.sheet_names:
            base = _safe_name(sheet)
            # Deduplicate: if two sheets produce the same safe name, append _2, _3 …
            if base in used:
                used[base] += 1
                safe = f"{base}_{used[base]}"
            else:
                used[base] = 1
                safe = base
            try:
                df = xl.parse(sheet)
                # Sanitise column names
                df.columns = [_safe_name(str(c)) for c in df.columns]
                df.to_sql(safe, self._conn, if_exists="replace", index=False)
                self._sheets.append(safe)
            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "ExcelConnector: skipping sheet %r — %s", sheet, exc
                )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        self._sheets = []

    def execute(self, sql: str, params=None) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(sql, params or ())
        try:
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            cur.close()

    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        return [("", s) for s in self._sheets]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        rows = self.execute(f'PRAGMA table_info("{table}")')
        return [
            {
                "name":                     r["name"],
                "data_type":                r["type"] or "TEXT",
                "nullable":                 True,
                "column_default":           None,
                "character_maximum_length": None,
                "numeric_precision":        None,
                "numeric_scale":            None,
            }
            for r in rows
        ]

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        return []

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        return []

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        return []

    def get_row_count(self, schema: str, table: str) -> int:
        n = self.execute_scalar(f'SELECT COUNT(*) FROM "{table}"')
        return int(n) if n else 0

    # ── dialect helpers ────────────────────────────────────────────────────────
    def _quote(self, name: str) -> str:
        return f'"{name}"'

    def _sample_clause(self, n: int) -> str:
        return f"LIMIT {n}"

    def _concat(self, cols: List[str]) -> str:
        return " || '|' || ".join(
            f"COALESCE(CAST(\"{c}\" AS TEXT), '__NULL__')" for c in cols
        )
