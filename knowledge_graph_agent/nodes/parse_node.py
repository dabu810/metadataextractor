"""
Knowledge Graph pipeline node: parse an OWL/RDF ontology string into an rdflib Graph.
"""
from __future__ import annotations

import logging

from rdflib import Graph

from ..state import KGState

logger = logging.getLogger(__name__)


def parse_node(state: KGState) -> KGState:
    text = (state.get("ontology_text") or "").strip()
    fmt  = state.get("ontology_format") or "turtle"

    if not text:
        state["errors"].append("Empty ontology text — nothing to convert.")
        state["phase"] = "error"
        return state

    g = Graph()
    try:
        g.parse(data=text, format=fmt)
    except Exception as exc:
        state["errors"].append(f"Failed to parse ontology ({fmt}): {exc}")
        state["phase"] = "error"
        return state

    logger.info("Parsed ontology: %d triples, format=%s", len(g), fmt)
    state["ontology_graph"] = g
    state["phase"] = "parsed"
    return state
