"""
LangChain tool: detect functional dependencies within a table.

Algorithm:
1. Generate candidate LHS-RHS column pairs.
   - Single-column LHS first, then composite (2-col, then 3-col) within budget.
2. Prune search space:
   - Skip constant columns (unique_count <= 1).
   - Skip columns with null_rate > 0.8 as determinants (weak signal).
   - Skip BLOB/CLOB/TEXT/JSON columns entirely.
   - Skip high-cardinality columns as RHS (near-unique → trivially determined by anything unique).
3. Verify each candidate with SQL GROUP BY via the connector.
4. Post-process:
   - Deduplicate FDs with identical LHS+RHS.
   - Classify FD type: primary_key / candidate_key / partial_key / non_key.
   - Detect transitively implied FDs (A→C when A→B and B→C both exist at conf=1.0).
   - Generate plain-English description for each FD.
5. Return all FDs above the configured confidence threshold.
"""
from __future__ import annotations

import itertools
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type

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


# Column data-type families to skip entirely for FD analysis
_SKIP_TYPE_PATTERNS = re.compile(
    r'\b(blob|clob|text|json|jsonb|xml|image|bytea|binary|varbinary|longtext|mediumtext|'
    r'tinytext|ntext|geography|geometry|hstore)\b',
    re.IGNORECASE,
)


def _is_skippable_type(dtype: str) -> bool:
    return bool(_SKIP_TYPE_PATTERNS.search(dtype or ""))


def _classify_fd_type(
    lhs: List[str],
    primary_keys: List[str],
    column_stats: Optional[Dict[str, Any]],
) -> str:
    """Classify a functional dependency by its LHS role."""
    pk_set = set(primary_keys)
    lhs_set = set(lhs)

    if pk_set and lhs_set == pk_set:
        return "primary_key"
    if pk_set and lhs_set < pk_set:
        return "partial_key"

    # Candidate key: single column or small set that is unique
    if column_stats:
        all_unique = all(
            (column_stats.get(col, {}).get("uniqueness_ratio", 0) or 0) >= 0.99
            for col in lhs
        )
        if all_unique:
            return "candidate_key"

    return "non_key"


def _describe_fd(
    lhs: List[str],
    rhs: List[str],
    confidence: float,
    violations: int,
    fd_type: str,
    is_transitively_implied: bool = False,
) -> str:
    """Generate a plain-English description grounded only in the FD metadata."""
    lhs_str = " + ".join(f"'{c}'" for c in lhs)
    rhs_str = f"'{rhs[0]}'" if rhs else "'?'"

    if is_transitively_implied:
        return (
            f"{lhs_str} determines {rhs_str} transitively through an intermediate column "
            f"(confidence {confidence * 100:.1f}%)"
        )

    if fd_type == "primary_key":
        return (
            f"Primary key {lhs_str} uniquely identifies {rhs_str} "
            f"— PK-implied dependency (0 violations)"
        )
    if fd_type == "candidate_key":
        return (
            f"Candidate key {lhs_str} uniquely determines {rhs_str} "
            f"(unique column, 0 violations)"
        )
    if fd_type == "partial_key":
        return (
            f"Partial key dependency: {lhs_str} → {rhs_str} "
            f"(LHS is a proper subset of the primary key; confidence {confidence * 100:.1f}%)"
        )

    if confidence == 1.0:
        return (
            f"{lhs_str} uniquely determines {rhs_str} "
            f"(exact FD, 0 violations in sampled rows)"
        )

    viol_str = f"{violations} violation{'s' if violations != 1 else ''}"
    return (
        f"{lhs_str} mostly determines {rhs_str} "
        f"({confidence * 100:.1f}% confidence, {viol_str})"
    )


class FunctionalDependencyTool(BaseTool):
    """
    Detect functional dependencies (X → Y) within a single database table.

    Uses SQL GROUP BY to verify whether knowing the value of X uniquely
    determines the value of Y across all rows.  Returns FDs with confidence
    scores, type classifications, and plain-English descriptions.
    """

    name: str = "functional_dependency_detector"
    description: str = (
        "Detect functional dependencies (X → Y) in a database table using SQL-based "
        "GROUP BY analysis.  Returns FDs with confidence scores (1.0 = perfect FD), "
        "type classifications (primary_key / candidate_key / partial_key / non_key), "
        "and plain-English descriptions of each dependency."
    )
    args_schema: Type[BaseModel] = FDDetectorInput
    connector: BaseConnector = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Pruning helpers
    # ------------------------------------------------------------------

    def _prune_columns(
        self,
        columns: List[str],
        primary_keys: List[str],
        stats: Optional[Dict[str, Any]],
        col_types: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Returns (determinant_candidates, dependent_candidates).

        Pruning rules:
        - Constant columns (unique_count <= 1): skip both sides.
        - BLOB/TEXT/JSON type columns: skip both sides (not meaningful for FDs).
        - High null rate (> 0.8) on LHS: skip as determinant (too many NULLs = weak signal).
        - Near-unique columns (uniqueness_ratio >= 0.99): skip as RHS — trivially
          determined by everything, so no business insight.
        """
        pk_set = set(primary_keys)
        det_cands: List[str] = []
        dep_cands: List[str] = []

        for col in columns:
            s = (stats or {}).get(col, {})
            dtype = (col_types or {}).get(col, "")

            # Skip unanalysable types entirely
            if _is_skippable_type(dtype):
                continue

            unique_count = s.get("unique_count")
            row_count = s.get("row_count", 2) or 1
            null_rate = s.get("null_rate", 0) or 0

            # Constant column — no discriminating power
            if unique_count is not None and unique_count <= 1:
                continue

            # Prefer pre-computed ratio from stats; fall back to computing from counts
            uniqueness_ratio = (
                s.get("uniqueness_ratio")
                if s.get("uniqueness_ratio") is not None
                else ((unique_count / row_count) if unique_count is not None else 0.5)
            )

            # LHS candidate: skip high-null columns (> 80% nulls = unreliable determinant)
            if null_rate <= 0.8:
                det_cands.append(col)

            # RHS candidate: skip near-unique columns (ratio >= 0.99) — they are trivially
            # determined by any unique key; skip PK columns too.
            if uniqueness_ratio < 0.99 and col not in pk_set:
                dep_cands.append(col)

        return det_cands, dep_cands

    def _generate_candidates(
        self,
        det_cols: List[str],
        dep_cols: List[str],
        max_pairs: int,
        primary_keys: List[str],
        column_stats: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[List[str], str]]:
        """
        Yield (lhs_list, rhs_col) candidate pairs.

        Budget allocation:
          - 60% of max_pairs for single-column LHS (highest quality signal).
          - 30% for 2-column composite LHS (catch partial keys and join keys).
          - 10% for 3-column composite LHS.

        Within single-column pairs, higher-uniqueness determinants are tested first
        (they produce more discriminating FDs).
        """
        pk_set = set(primary_keys)

        single_budget    = max(1, int(max_pairs * 0.60))
        composite2_budget = max(1, int(max_pairs * 0.30))
        composite3_budget = max_pairs - single_budget - composite2_budget

        # --- Sort determinant columns: PK cols first, then by uniqueness_ratio desc ---
        def _det_priority(col: str) -> float:
            if col in pk_set:
                return 2.0  # highest priority
            ratio = (column_stats or {}).get(col, {}).get("uniqueness_ratio") or 0.0
            return ratio or 0.0

        sorted_det = sorted(det_cols, key=_det_priority, reverse=True)

        candidates: List[Tuple[List[str], str]] = []

        # Single-column LHS
        for det, dep in itertools.product(sorted_det, dep_cols):
            if det == dep:
                continue
            candidates.append(([det], dep))
            if len(candidates) >= single_budget:
                break

        # 2-column composite LHS
        count2 = 0
        for combo in itertools.combinations(sorted_det, 2):
            if pk_set and set(combo).issuperset(pk_set):
                continue
            for dep in dep_cols:
                if dep in combo:
                    continue
                candidates.append((list(combo), dep))
                count2 += 1
                if count2 >= composite2_budget:
                    break
            if count2 >= composite2_budget:
                break

        # 3-column composite LHS
        count3 = 0
        if composite3_budget > 0:
            for combo in itertools.combinations(sorted_det, 3):
                if pk_set and set(combo).issuperset(pk_set):
                    continue
                for dep in dep_cols:
                    if dep in combo:
                        continue
                    candidates.append((list(combo), dep))
                    count3 += 1
                    if count3 >= composite3_budget:
                        break
                if count3 >= composite3_budget:
                    break

        return candidates

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(fds: List[Dict]) -> List[Dict]:
        """Remove FDs with identical (determinant, dependent) pairs, keeping highest confidence."""
        seen: Dict[tuple, Dict] = {}
        for fd in fds:
            key = (tuple(sorted(fd["determinant"])), tuple(sorted(fd["dependent"])))
            if key not in seen or fd["confidence"] > seen[key]["confidence"]:
                seen[key] = fd
        return list(seen.values())

    @staticmethod
    def _mark_transitive(fds: List[Dict]) -> List[Dict]:
        """
        Mark FDs as transitively_implied when a shorter derivation exists.

        A→C is transitively implied if there exists B such that A→B and B→C
        are both in the FD set at confidence 1.0.

        Only marks at confidence 1.0 to avoid false positives.
        """
        # Build lookup: determinant_tuple → set of dependent columns (at conf 1.0)
        det_to_deps: Dict[tuple, Set[str]] = {}
        for fd in fds:
            if fd["confidence"] < 1.0:
                continue
            key = tuple(sorted(fd["determinant"]))
            det_to_deps.setdefault(key, set()).update(fd["dependent"])

        for fd in fds:
            if fd.get("fd_type") == "primary_key":
                continue  # never mark PK FDs as transitive
            if fd["confidence"] < 1.0:
                continue
            lhs_key = tuple(sorted(fd["determinant"]))
            rhs = fd["dependent"][0] if fd["dependent"] else None
            if rhs is None:
                continue

            # Check: does LHS determine some B, and does B determine RHS?
            lhs_deps = det_to_deps.get(lhs_key, set())
            for intermediate in lhs_deps:
                if intermediate == rhs:
                    continue
                int_key = (intermediate,)
                int_deps = det_to_deps.get(int_key, set())
                if rhs in int_deps:
                    fd["fd_type"] = "transitively_implied"
                    fd["description"] = _describe_fd(
                        fd["determinant"], fd["dependent"],
                        fd["confidence"], fd["violations"],
                        fd["fd_type"], is_transitively_implied=True,
                    )
                    break

        return fds

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
            # Build type map from stats if available
            col_types: Dict[str, str] = {}

            det_cols, dep_cols = self._prune_columns(
                columns, primary_keys, column_stats, col_types
            )
            candidates = self._generate_candidates(det_cols, dep_cols, max_pairs, primary_keys, column_stats)

            found_fds: List[Dict] = []
            tested_keys: Set[tuple] = set()  # avoid duplicate SQL calls

            for lhs, rhs in candidates:
                pair_key = (tuple(sorted(lhs)), rhs)
                if pair_key in tested_keys:
                    continue
                tested_keys.add(pair_key)

                conf, violations = self.connector.check_functional_dependency(
                    schema_name, table_name, lhs, [rhs], sample_size
                )
                if conf < threshold:
                    continue

                # Skip FDs trivially implied by a PK superset (PK already determines everything)
                if primary_keys and set(lhs).issuperset(set(primary_keys)):
                    # Will be captured as PK FDs below; skip to avoid double-counting
                    continue

                fd_type = _classify_fd_type(lhs, primary_keys, column_stats)
                description = _describe_fd(lhs, [rhs], conf, violations, fd_type)

                found_fds.append({
                    "table": table_name,
                    "determinant": lhs,
                    "dependent": [rhs],
                    "confidence": round(conf, 4),
                    "violations": violations,
                    "fd_type": fd_type,
                    "description": description,
                })

            # Explicit PK-based FDs (PK → every non-PK column)
            if primary_keys:
                pk_rhs_set = {fd["dependent"][0] for fd in found_fds
                              if fd.get("fd_type") == "primary_key"}
                for dep in dep_cols:
                    if dep in primary_keys or dep in pk_rhs_set:
                        continue
                    description = _describe_fd(
                        primary_keys, [dep], 1.0, 0, "primary_key"
                    )
                    found_fds.insert(0, {
                        "table": table_name,
                        "determinant": primary_keys,
                        "dependent": [dep],
                        "confidence": 1.0,
                        "violations": 0,
                        "fd_type": "primary_key",
                        "description": description,
                        "source": "primary_key",
                    })

            # Deduplicate, then mark transitive FDs
            found_fds = self._deduplicate(found_fds)
            found_fds = self._mark_transitive(found_fds)

            # Sort: PK FDs first, then by confidence desc, then LHS length asc
            _type_order = {
                "primary_key": 0, "candidate_key": 1, "partial_key": 2,
                "non_key": 3, "transitively_implied": 4,
            }
            found_fds.sort(key=lambda f: (
                _type_order.get(f.get("fd_type", "non_key"), 9),
                -f["confidence"],
                len(f["determinant"]),
            ))

            return json.dumps({
                "table": table_name,
                "functional_dependencies": found_fds,
                "candidates_tested": len(candidates),
                "determinant_columns": len(det_cols),
                "dependent_columns": len(dep_cols),
            }, default=str)

        except Exception as e:
            return json.dumps({"error": str(e), "table": table_name})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
