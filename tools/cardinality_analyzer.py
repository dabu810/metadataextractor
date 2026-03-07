"""
LangChain tool: analyse join cardinality between table pairs.

Determines whether the relationship between two tables (via shared or
similarly-named columns) is 1:1, 1:N, N:1, or M:N.

Strategy:
1. Find candidate join columns: columns that appear in both tables (by name),
   or that are FK-referenced.
2. For each candidate, count DISTINCT values of the join column in each table.
3. Compare with total row count to determine uniqueness on each side.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..connectors.base import BaseConnector


class CardinalityAnalyzerInput(BaseModel):
    schema_name: str = Field(description="Database schema name")
    left_table: str = Field(description="First table")
    right_table: str = Field(description="Second table")
    left_columns: List[str] = Field(description="Column names of left table")
    right_columns: List[str] = Field(description="Column names of right table")
    foreign_keys: List[Dict[str, str]] = Field(
        default_factory=list,
        description="FK definitions: [{column, referenced_table, referenced_column}]"
    )


class CardinalityAnalyzerTool(BaseTool):
    """
    Determine the cardinality relationship (1:1, 1:N, N:1, M:N) between
    two database tables using shared column names and FK hints.
    """

    name: str = "cardinality_analyzer"
    description: str = (
        "Determine the join cardinality (1:1, 1:N, N:1, M:N) between two tables "
        "by finding common join columns and counting distinct values on each side."
    )
    args_schema: Type[BaseModel] = CardinalityAnalyzerInput
    connector: BaseConnector = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _find_join_columns(
        self,
        left_table: str,
        right_table: str,
        left_cols: List[str],
        right_cols: List[str],
        fks: List[Dict[str, str]],
    ) -> List[List[str]]:
        """
        Returns candidate join column sets (each set is a list of col names
        used as the join key).  Tries FK hints first, then name matching.
        """
        candidates: List[List[str]] = []

        # FK-based candidates
        for fk in fks:
            if fk.get("referenced_table") == right_table:
                candidates.append([fk["column"]])
            elif fk.get("referenced_table") == left_table:
                candidates.append([fk["column"]])

        # Name-based candidates: columns that share the same name
        right_col_set = set(rc.lower() for rc in right_cols)
        for lc in left_cols:
            if lc.lower() in right_col_set and [lc] not in candidates:
                candidates.append([lc])

        return candidates  # empty = no join columns found, skip this pair

    def _run(
        self,
        schema_name: str,
        left_table: str,
        right_table: str,
        left_columns: List[str],
        right_columns: List[str],
        foreign_keys: List[Dict[str, str]] = None,
    ) -> str:
        foreign_keys = foreign_keys or []
        try:
            join_col_sets = self._find_join_columns(
                left_table, right_table, left_columns, right_columns, foreign_keys
            )

            relationships = []
            for join_cols in join_col_sets[:5]:  # cap at 5 candidate sets
                # Ensure the join column exists in both tables
                right_col_set = set(rc.lower() for rc in right_columns)
                valid_join = [c for c in join_cols if c.lower() in right_col_set]
                if not valid_join:
                    continue

                result = self.connector.get_join_cardinality(
                    schema_name, left_table, right_table, valid_join
                )
                relationships.append({
                    "join_columns": valid_join,
                    "relationship_type": result["relationship_type"],
                    "left_unique_values": result["left_unique"],
                    "right_unique_values": result["right_unique"],
                    "left_table": left_table,
                    "right_table": right_table,
                })

            return json.dumps({
                "left_table": left_table,
                "right_table": right_table,
                "cardinality_results": relationships,
            }, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
