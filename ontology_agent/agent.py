"""
Ontology Agent — LangGraph pipeline.

Graph topology:
    START → load → build → serialize → END
                ↓ (error)
            error_end → END

Completely decoupled from the metadata_agent package.
The only input is a plain Python dict (the JSON report).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from .config import OntologyConfig
from .nodes import build_node, load_node, serialize_node
from .state import OntologyState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing + error
# ---------------------------------------------------------------------------

def _route_after_load(state: OntologyState) -> str:
    return "error_end" if state.get("phase") == "error" else "build"


def _error_end_node(state: OntologyState) -> OntologyState:
    logger.error("Ontology pipeline terminating: %s", state.get("errors"))
    state["phase"] = "error"
    return state


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    graph = StateGraph(OntologyState)
    graph.add_node("load",      load_node)
    graph.add_node("build",     build_node)
    graph.add_node("serialize", serialize_node)
    graph.add_node("error_end", _error_end_node)

    graph.add_edge(START, "load")
    graph.add_conditional_edges(
        "load",
        _route_after_load,
        {"build": "build", "error_end": "error_end"},
    )
    graph.add_edge("build",     "serialize")
    graph.add_edge("serialize", END)
    graph.add_edge("error_end", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OntologyAgent:
    """
    Convert a metadata report dict to an OWL/RDF ontology.

    Usage::

        agent  = OntologyAgent(OntologyConfig(output_path="schema.ttl"))
        result = agent.run(report_dict)
        # result["ontology_turtle"] — Turtle string
        # result["output_path"]     — path written (if configured)
        # result["class_count"]     — number of OWL classes
        # result["property_count"]  — number of OWL properties
    """

    def __init__(self, config: Optional[OntologyConfig] = None):
        self._config = config or OntologyConfig()
        self._graph  = _build_graph()

    def _initial_state(self, report: Dict) -> OntologyState:
        return {
            "config":          self._config,
            "report":          report,
            "ontology_graph":  None,
            "class_map":       {},
            "property_map":    {},
            "ontology_turtle": "",
            "output_path":     "",
            "triple_count":    0,
            "class_count":     0,
            "property_count":  0,
            "errors":          [],
            "phase":           "init",
        }

    def run(self, report: Dict) -> Dict:
        """Synchronous execution — returns a result summary dict."""
        final = self._graph.invoke(self._initial_state(report))
        if final.get("errors"):
            for e in final["errors"]:
                logger.warning("Ontology warning: %s", e)
        return {
            "ontology_turtle": final.get("ontology_turtle", ""),
            "output_path":     final.get("output_path", ""),
            "class_count":     final.get("class_count", 0),
            "property_count":  final.get("property_count", 0),
            "triple_count":    final.get("triple_count", 0),
            "errors":          final.get("errors", []),
            "phase":           final.get("phase", ""),
        }

    def stream_run(self, report: Dict):
        """Yield (node_name, state_update) as each pipeline node completes."""
        for event in self._graph.stream(self._initial_state(report)):
            for node_name, state_update in event.items():
                yield node_name, state_update
