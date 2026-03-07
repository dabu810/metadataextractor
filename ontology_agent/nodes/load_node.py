"""
Ontology pipeline node: validate and summarise the incoming metadata report.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..state import OntologyState

logger = logging.getLogger(__name__)


def load_node(state: OntologyState) -> OntologyState:
    report: Dict = state.get("report", {})

    if not report:
        state["errors"].append("Empty report — nothing to build ontology from.")
        state["phase"] = "error"
        return state

    tables = report.get("tables") or {}
    if not tables:
        state["errors"].append("Report contains no tables.")
        state["phase"] = "error"
        return state

    logger.info(
        "Ontology load: %d tables, %d FDs, %d INDs, %d cardinality relationships",
        len(tables),
        len(report.get("functional_dependencies") or []),
        len(report.get("inclusion_dependencies") or []),
        len(report.get("cardinality_relationships") or []),
    )
    state["phase"] = "loaded"
    return state
