"""
LangGraph node: run FD detection, IND detection, and cardinality analysis.

This is the most compute-intensive node.  It:
  1. Runs FunctionalDependencyTool on each table individually.
  2. Runs InclusionDependencyTool for every ordered pair of tables.
  3. Runs CardinalityAnalyzerTool for every unordered pair of tables.

Results are stored in state['func_deps'], state['incl_deps'],
and state['cardinalities'].
"""
from __future__ import annotations

import itertools
import json
import logging
from typing import Any, Dict, List

from ..state import (
    AgentState,
    CardinalityRelationship,
    FunctionalDependency,
    InclusionDependency,
    TableMeta,
)
from ..tools import (
    CardinalityAnalyzerTool,
    FunctionalDependencyTool,
    InclusionDependencyTool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_dicts(table_meta: TableMeta) -> List[Dict[str, str]]:
    return [{"name": c.name, "data_type": c.data_type} for c in table_meta.columns]


def _col_names(table_meta: TableMeta) -> List[str]:
    return [c.name for c in table_meta.columns]


def _col_stats(table_meta: TableMeta) -> Dict[str, Any]:
    out = {}
    for c in table_meta.columns:
        out[c.name] = {
            "unique_count": c.unique_count,
            "null_count": c.null_count,
            "row_count": c.row_count or table_meta.row_count,
        }
    return out


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def analysis_node(state: AgentState) -> AgentState:
    config = state["agent_config"]
    connector = state["connector"]
    table_metadata: Dict[str, TableMeta] = state["table_metadata"]

    if not table_metadata:
        state["phase"] = "analysed"
        return state

    fd_tool  = FunctionalDependencyTool(connector=connector)
    id_tool  = InclusionDependencyTool(connector=connector)
    car_tool = CardinalityAnalyzerTool(connector=connector)

    # ------------------------------------------------------------------
    # 1. Functional Dependencies (per table)
    # ------------------------------------------------------------------
    logger.info("=== Functional Dependency Detection ===")
    for table_name, meta in table_metadata.items():
        logger.info("  FD scan: %s", table_name)
        result_json = fd_tool._run(
            schema_name=meta.schema_name,
            table_name=table_name,
            columns=_col_names(meta),
            primary_keys=meta.primary_keys,
            sample_size=config.sample_size,
            threshold=config.fd_threshold,
            max_pairs=config.max_fd_column_pairs,
            column_stats=_col_stats(meta),
        )
        result = json.loads(result_json)
        if "error" in result:
            state["errors"].append(f"FD error [{table_name}]: {result['error']}")
            continue

        for fd in result.get("functional_dependencies", []):
            state["func_deps"].append(
                FunctionalDependency(
                    table_name=table_name,
                    determinant=fd["determinant"],
                    dependent=fd["dependent"],
                    confidence=fd["confidence"],
                    num_violations=fd.get("violations", 0),
                )
            )
        logger.info(
            "    → %d FDs found (%d pairs tested)",
            len(result.get("functional_dependencies", [])),
            result.get("candidates_tested", 0),
        )

    # ------------------------------------------------------------------
    # 2. Inclusion Dependencies (ordered table pairs)
    # ------------------------------------------------------------------
    logger.info("=== Inclusion Dependency Detection ===")
    table_names = list(table_metadata.keys())
    pair_count = 0

    for left_name, right_name in itertools.permutations(table_names, 2):
        if pair_count >= config.max_id_column_pairs:
            break
        left_meta  = table_metadata[left_name]
        right_meta = table_metadata[right_name]

        logger.info("  IND scan: %s → %s", left_name, right_name)
        result_json = id_tool._run(
            schema_name=left_meta.schema_name,
            left_table=left_name,
            right_table=right_name,
            left_columns=_col_dicts(left_meta),
            right_columns=_col_dicts(right_meta),
            sample_size=config.sample_size,
            threshold=config.id_threshold,
            max_pairs=50,
        )
        result = json.loads(result_json)
        if "error" in result:
            state["errors"].append(f"IND error [{left_name}→{right_name}]: {result['error']}")
            continue

        for ind in result.get("inclusion_dependencies", []):
            state["incl_deps"].append(
                InclusionDependency(
                    left_table=left_name,
                    left_columns=ind["left_columns"],
                    right_table=right_name,
                    right_columns=ind["right_columns"],
                    coverage=ind["coverage"],
                    is_foreign_key_candidate=ind["is_foreign_key_candidate"],
                )
            )
        pair_count += 1

    logger.info(
        "Total INDs found: %d (across %d ordered pairs)",
        len(state["incl_deps"]),
        pair_count,
    )

    # ------------------------------------------------------------------
    # 3. Cardinality Analysis (unordered table pairs)
    # ------------------------------------------------------------------
    logger.info("=== Cardinality Analysis ===")
    for left_name, right_name in itertools.combinations(table_names, 2):
        left_meta  = table_metadata[left_name]
        right_meta = table_metadata[right_name]

        # Collect all FK hints for both tables
        all_fks = left_meta.foreign_keys + right_meta.foreign_keys

        logger.info("  Cardinality: %s ↔ %s", left_name, right_name)
        result_json = car_tool._run(
            schema_name=left_meta.schema_name,
            left_table=left_name,
            right_table=right_name,
            left_columns=_col_names(left_meta),
            right_columns=_col_names(right_meta),
            foreign_keys=all_fks,
        )
        result = json.loads(result_json)
        if "error" in result:
            state["errors"].append(f"Cardinality error [{left_name}↔{right_name}]: {result['error']}")
            continue

        for cr in result.get("cardinality_results", []):
            state["cardinalities"].append(
                CardinalityRelationship(
                    left_table=left_name,
                    right_table=right_name,
                    join_columns=cr["join_columns"],
                    relationship_type=cr["relationship_type"],
                    left_unique=cr.get("left_unique_values", 0),
                    right_unique=cr.get("right_unique_values", 0),
                )
            )

    logger.info("Total cardinality relationships: %d", len(state["cardinalities"]))

    state["phase"] = "analysed"
    return state
