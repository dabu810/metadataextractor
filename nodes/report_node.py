"""
LangGraph node: aggregate all collected metadata into a final structured report
and optionally write it to disk as JSON.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..state import AgentState, TableMeta

logger = logging.getLogger(__name__)


def _meta_to_dict(meta: TableMeta) -> Dict[str, Any]:
    d = dataclasses.asdict(meta)
    return d


def report_node(state: AgentState) -> AgentState:
    config = state["agent_config"]
    table_metadata: Dict[str, TableMeta] = state["table_metadata"]

    # ------------------------------------------------------------------
    # Summarise discovered FKs from inclusion dependencies
    # ------------------------------------------------------------------
    fk_candidates = [
        {
            "left_table": ind.left_table,
            "left_columns": ind.left_columns,
            "right_table": ind.right_table,
            "right_columns": ind.right_columns,
            "coverage": ind.coverage,
            "ind_type": ind.ind_type,
            "description": ind.description,
        }
        for ind in state["incl_deps"]
        if ind.is_foreign_key_candidate
    ]

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database_type": config.db_config.db_type.value,
        "schema": config.db_config.schema or config.db_config.database or "unknown",
        "summary": {
            "total_tables": len(table_metadata),
            "total_columns": sum(len(m.columns) for m in table_metadata.values()),
            "total_functional_dependencies": len(state["func_deps"]),
            "total_inclusion_dependencies": len(state["incl_deps"]),
            "total_fk_candidates": len(fk_candidates),
            "total_cardinality_relationships": len(state["cardinalities"]),
            "errors": state["errors"],
        },
        "tables": {name: _meta_to_dict(meta) for name, meta in table_metadata.items()},
        "functional_dependencies": [
            {
                "table": fd.table_name,
                "determinant": fd.determinant,
                "dependent": fd.dependent,
                "confidence": fd.confidence,
                "violations": fd.num_violations,
                "fd_type": fd.fd_type,
                "description": fd.description,
            }
            for fd in state["func_deps"]
        ],
        "inclusion_dependencies": [
            {
                "left_table": ind.left_table,
                "left_columns": ind.left_columns,
                "right_table": ind.right_table,
                "right_columns": ind.right_columns,
                "coverage": ind.coverage,
                "is_fk_candidate": ind.is_foreign_key_candidate,
                "ind_type": ind.ind_type,
                "description": ind.description,
            }
            for ind in state["incl_deps"]
        ],
        "fk_candidates": fk_candidates,
        "cardinality_relationships": [
            {
                "left_table": cr.left_table,
                "right_table": cr.right_table,
                "join_columns": cr.join_columns,
                "type": cr.relationship_type,
                "left_unique_values": cr.left_unique,
                "right_unique_values": cr.right_unique,
            }
            for cr in state["cardinalities"]
        ],
    }

    state["final_report"] = report

    # ------------------------------------------------------------------
    # Write to disk if requested
    # ------------------------------------------------------------------
    if config.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
        with open(config.output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Report written to %s", config.output_path)
    else:
        logger.info("Report assembled (no output_path configured).")

    state["phase"] = "done"
    return state
