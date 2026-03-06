"""
LangChain tool: detect functional dependencies within a table.

Algorithm:
1. Generate candidate LHS-RHS column pairs (skip PKs as trivial determinants,
   skip high-cardinality columns as dependent).
2. For each candidate, run SQL GROUP BY to check if LHS → RHS holds.
3. Return all FDs above the configured confidence threshold.

Optimisations:
- Prune search space: exclude constant columns, binary columns, etc.
- Respect max_fd_column_pairs cap to avoid combinatorial explosion.
- Single SQL query per candidate pair.
"""
from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..connectors.base import BaseConnector


class FDDetectorInput(BaseModel):
    schema_name: str = Field(description="Database schema name")
    table_name: str = Field(description="Table to analyse")
    columns: List[str] = Field(description="All column names in the table")
    primary_keys: List[str] = Field(default_factory=list, description="Primary key columns")
    sample_size: int = Field(default=10_000)
    threshold: float = Field(default=1.0, description="Minimum FD confidence (0-1)")
    max_pairs: int = Field(default=200, description="Max LHS-RHS pairs to test")
    column_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pre-computed column stats dict (null_rate, uniqueness_ratio, etc.)"
    )


class FunctionalDependencyTool(BaseTool):
    """
    Detect functional dependencies (X → Y) within a single database table.

    Uses SQL GROUP BY to verify whether knowing the value of X uniquely
    determines the value of Y across all rows.  Returns a list of FDs with
    confidence scores.
    """

    name: str = "functional_dependency_detector"
    description: str = (
        "Detect functional dependencies (X → Y) in a database table using SQL-based "
        "GROUP BY analysis.  Returns FDs with confidence scores (1.0 = perfect FD). "
        "Handles single-column and composite determinants."
    )
    args_schema: Type[BaseModel] = FDDetectorInput
    connector: BaseConnector = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Candidate generation helpers
    # ------------------------------------------------------------------

    def _prune_columns(
        self,
        columns: List[str],
        primary_keys: List[str],
        stats: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        """
        Returns (determinant_candidates, dependent_candidates).

        Rules:
        - Constant columns (unique_count <= 1) are useless as either side.
        - PK columns can be determinants but are trivial – keep but mark.
        - Columns with uniqueness_ratio == 1.0 can be determinants (they
          trivially determine everything); we skip them as RHS candidates
          because no real FD insight is gained.
        """
        pk_set = set(primary_keys)
        det_cands: List[str] = []
        dep_cands: List[str] = []

        for col in columns:
            s = (stats or {}).get(col, {})
            unique_count = s.get("unique_count", 2)
            row_count = s.get("row_count", 2) or 1
            ratio = unique_count / row_count if row_count else 0

            if unique_count is not None and unique_count <= 1:
                continue  # constant column – skip both sides

            # Candidate for LHS: anything with reasonable cardinality
            det_cands.append(col)

            # Candidate for RHS: not a unique-key column (those trivially have
            # FDs from everything), not a PK column (known)
            if ratio < 0.99 and col not in pk_set:
                dep_cands.append(col)

        return det_cands, dep_cands

    def _generate_candidates(
        self,
        det_cols: List[str],
        dep_cols: List[str],
        max_pairs: int,
        primary_keys: List[str],
    ) -> List[Tuple[List[str], str]]:
        """
        Yield (lhs_list, rhs_col) candidate pairs.
        Generates single-column LHS first, then composite LHS (up to 3 cols).
        """
        pk_set = set(primary_keys)
        candidates: List[Tuple[List[str], str]] = []

        # Single-column determinants
        for det, dep in itertools.product(det_cols, dep_cols):
            if det == dep:
                continue
            candidates.append(([det], dep))
            if len(candidates) >= max_pairs:
                return candidates

        # Composite determinants (pairs)
        for combo in itertools.combinations(det_cols, 2):
            for dep in dep_cols:
                if dep in combo:
                    continue
                candidates.append((list(combo), dep))
                if len(candidates) >= max_pairs:
                    return candidates

        return candidates

    # ------------------------------------------------------------------
    def _run(
        self,
        schema_name: str,
        table_name: str,
        columns: List[str],
        primary_keys: List[str] = None,
        sample_size: int = 10_000,
        threshold: float = 1.0,
        max_pairs: int = 200,
        column_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        primary_keys = primary_keys or []
        try:
            det_cols, dep_cols = self._prune_columns(columns, primary_keys, column_stats)
            candidates = self._generate_candidates(det_cols, dep_cols, max_pairs, primary_keys)

            found_fds = []
            for lhs, rhs in candidates:
                conf, violations = self.connector.check_functional_dependency(
                    schema_name, table_name, lhs, [rhs], sample_size
                )
                if conf >= threshold:
                    # Skip trivial FDs implied by PKs
                    if set(lhs).issuperset(set(primary_keys)) and primary_keys:
                        continue
                    found_fds.append({
                        "table": table_name,
                        "determinant": lhs,
                        "dependent": [rhs],
                        "confidence": round(conf, 4),
                        "violations": violations,
                    })

            # Also record PK-based FDs explicitly
            if primary_keys:
                for dep in dep_cols:
                    if dep not in primary_keys:
                        found_fds.insert(0, {
                            "table": table_name,
                            "determinant": primary_keys,
                            "dependent": [dep],
                            "confidence": 1.0,
                            "violations": 0,
                            "source": "primary_key",
                        })

            return json.dumps({
                "table": table_name,
                "functional_dependencies": found_fds,
                "candidates_tested": len(candidates),
            }, default=str)

        except Exception as e:
            return json.dumps({"error": str(e), "table": table_name})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
