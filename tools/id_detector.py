"""
LangChain tool: detect inclusion dependencies (INDs) across tables.

An Inclusion Dependency R[A] ⊆ S[B] holds when every value appearing in
column A of table R also appears in column B of table S.

A high-coverage IND (≥ threshold) is a foreign-key candidate.

Strategy:
1. For each pair of tables (R, S), find column pairs with compatible data types.
2. Prioritise column pairs that share the same name or similar name.
3. Run LEFT JOIN to count how many R.A values appear in S.B.
4. Return all INDs above the coverage threshold.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..connectors.base import BaseConnector


class IDDetectorInput(BaseModel):
    schema_name: str = Field(description="Database schema name")
    left_table: str = Field(description="Left-hand side table (R)")
    right_table: str = Field(description="Right-hand side table (S)")
    left_columns: List[Dict[str, str]] = Field(
        description="Columns of left table: [{name, data_type}, ...]"
    )
    right_columns: List[Dict[str, str]] = Field(
        description="Columns of right table: [{name, data_type}, ...]"
    )
    sample_size: int = Field(default=10_000)
    threshold: float = Field(default=0.95, description="Min coverage fraction")
    max_pairs: int = Field(default=100)


class InclusionDependencyTool(BaseTool):
    """
    Detect inclusion dependencies (R[A] ⊆ S[B]) between two database tables.

    Uses SQL LEFT JOIN to measure coverage.  Column pairs with compatible types
    and similar names are tested first.  Returns INDs with coverage scores;
    high-coverage INDs are flagged as foreign-key candidates.
    """

    name: str = "inclusion_dependency_detector"
    description: str = (
        "Detect inclusion dependencies between two tables: "
        "checks if every value in left_table[col] exists in right_table[col]. "
        "High-coverage INDs (≥ threshold) are flagged as FK candidates."
    )
    args_schema: Type[BaseModel] = IDDetectorInput
    connector: BaseConnector = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    _NUMERIC_TYPES = {"int", "integer", "bigint", "smallint", "numeric",
                      "decimal", "float", "double", "real", "number"}
    _STRING_TYPES  = {"varchar", "char", "text", "string", "nvarchar",
                      "character varying", "clob"}
    _DATE_TYPES    = {"date", "timestamp", "datetime", "time"}

    def _type_family(self, dtype: str) -> str:
        dt = dtype.lower().split("(")[0].strip()
        if any(t in dt for t in self._NUMERIC_TYPES):
            return "numeric"
        if any(t in dt for t in self._STRING_TYPES):
            return "string"
        if any(t in dt for t in self._DATE_TYPES):
            return "date"
        return "other"

    def _name_similarity(self, a: str, b: str) -> float:
        """Rough similarity score between two column names."""
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        # Remove common suffixes/prefixes like _id, _key, id_
        def normalise(s):
            return re.sub(r"(_id|_key|_code|id_|key_)$", "", s)
        if normalise(a) == normalise(b):
            return 0.9
        if a in b or b in a:
            return 0.7
        return 0.0

    def _generate_column_pairs(
        self,
        left_columns: List[Dict[str, str]],
        right_columns: List[Dict[str, str]],
        max_pairs: int,
    ) -> List[Tuple[str, str]]:
        """
        Returns list of (left_col, right_col) pairs to test, sorted by
        name similarity (descending) so high-signal pairs are tested first.
        """
        candidates = []
        for lc in left_columns:
            for rc in right_columns:
                lf = self._type_family(lc["data_type"])
                rf = self._type_family(rc["data_type"])
                if lf != rf and lf != "other" and rf != "other":
                    continue  # incompatible types
                sim = self._name_similarity(lc["name"], rc["name"])
                candidates.append((lc["name"], rc["name"], sim))

        # Sort by similarity desc, then alphabetically for determinism
        candidates.sort(key=lambda x: (-x[2], x[0], x[1]))
        return [(a, b) for a, b, _ in candidates[:max_pairs]]

    # ------------------------------------------------------------------
    def _run(
        self,
        schema_name: str,
        left_table: str,
        right_table: str,
        left_columns: List[Dict[str, str]],
        right_columns: List[Dict[str, str]],
        sample_size: int = 10_000,
        threshold: float = 0.95,
        max_pairs: int = 100,
    ) -> str:
        try:
            pairs = self._generate_column_pairs(left_columns, right_columns, max_pairs)
            found_inds = []

            for left_col, right_col in pairs:
                coverage = self.connector.check_inclusion_dependency(
                    schema_name,
                    left_table, [left_col],
                    right_table, [right_col],
                    sample_size,
                )
                if coverage >= threshold:
                    found_inds.append({
                        "left_table": left_table,
                        "left_columns": [left_col],
                        "right_table": right_table,
                        "right_columns": [right_col],
                        "coverage": round(coverage, 4),
                        "is_foreign_key_candidate": coverage >= 0.99,
                    })

            return json.dumps({
                "left_table": left_table,
                "right_table": right_table,
                "inclusion_dependencies": found_inds,
                "pairs_tested": len(pairs),
            }, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
