"""
LangChain tool: collect table-level and column-level statistics.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..connectors.base import BaseConnector


_PATTERN_CHECKS = [
    ("EMAIL",         re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')),
    ("URL",           re.compile(r'^https?://')),
    ("UUID",          re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)),
    ("ISO_DATE",      re.compile(r'^\d{4}-\d{2}-\d{2}$')),
    ("DATETIME",      re.compile(r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}')),
    ("YEAR_4DIGIT",   re.compile(r'^(19|20)\d{2}$')),
    ("PHONE",         re.compile(r'^\+?[\d\s\-().]{7,20}$')),
    ("CURRENCY_CODE", re.compile(r'^[A-Z]{3}$')),
    ("COUNTRY_CODE2", re.compile(r'^[A-Z]{2}$')),
    ("NUMERIC_ID",    re.compile(r'^\d{3,18}$')),
    ("BOOLEAN_TEXT",  re.compile(r'^(yes|no|true|false|y|n|1|0)$', re.IGNORECASE)),
]


def _detect_patterns(top_values: List[Any], data_type: str) -> List[str]:
    """
    Detect value patterns from the top-N most frequent values.
    Returns a list of pattern labels (e.g. ["EMAIL", "ISO_DATE"]).
    Only labels a pattern if ≥ 60% of non-null sample values match it.
    """
    str_values = [str(v) for v in (top_values or []) if v is not None]
    if not str_values:
        return []

    detected = []
    for label, pattern in _PATTERN_CHECKS:
        matches = sum(1 for v in str_values if pattern.match(v))
        if matches / len(str_values) >= 0.6:
            detected.append(label)

    return detected


class MetadataCollectorInput(BaseModel):
    schema_name: str = Field(description="Database schema / dataset name")
    table_name: str = Field(description="Table name")
    columns: List[str] = Field(description="List of column names to collect stats for")
    sample_size: int = Field(default=10_000, description="Max rows to sample for stats")


class MetadataCollectorTool(BaseTool):
    """
    Collect row count, size, and per-column statistics (null rate, cardinality,
    min/max/avg/stddev, top-N values) for a table.
    """

    name: str = "metadata_collector"
    description: str = (
        "Collect quantitative metadata for a database table and its columns. "
        "Returns row count, table size, and per-column stats "
        "(null count, unique count, min, max, avg, stddev, top values)."
    )
    args_schema: Type[BaseModel] = MetadataCollectorInput
    connector: BaseConnector = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        schema_name: str,
        table_name: str,
        columns: List[str],
        sample_size: int = 10_000,
    ) -> str:
        try:
            row_count = self.connector.get_row_count(schema_name, table_name)
            size_bytes = self.connector.get_table_size_bytes(schema_name, table_name)

            column_stats = {}
            for col in columns:
                try:
                    stats = self.connector.get_column_stats(
                        schema_name, table_name, col, sample_size
                    )
                    # Compute null rate and uniqueness ratio
                    total = stats.get("row_count", row_count) or 1
                    null_count = stats.get("null_count", 0) or 0
                    unique_count = stats.get("unique_count", 0) or 0
                    stats["null_rate"] = round(null_count / total, 4)
                    stats["uniqueness_ratio"] = round(unique_count / total, 4)
                    stats["is_high_cardinality"] = unique_count / total > 0.95
                    stats["is_constant"] = unique_count <= 1
                    # Detect value patterns from top_values sample
                    stats["pattern_hints"] = _detect_patterns(
                        stats.get("top_values", []),
                        "",  # data_type not available here; patterns are value-based
                    )
                    column_stats[col] = stats
                except Exception as e:
                    column_stats[col] = {"error": str(e)}

            result = {
                "schema_name": schema_name,
                "table_name": table_name,
                "row_count": row_count,
                "size_bytes": size_bytes,
                "column_stats": column_stats,
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "table": table_name})

    async def _arun(self, schema_name: str, table_name: str,
                    columns: List[str], sample_size: int = 10_000) -> str:
        return self._run(schema_name, table_name, columns, sample_size)
