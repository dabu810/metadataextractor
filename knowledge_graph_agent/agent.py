"""
Knowledge Graph Agent — LangGraph pipeline.

Graph topology:
    START → parse → [error] → error_end → END
                 → translate → execute → END

Completely decoupled from metadata_agent and ontology_agent.
The only inputs are a raw ontology string and a KGConfig.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from .config import KGConfig
from .nodes import execute_node, parse_node, translate_node
from .state import KGState

logger = logging.getLogger(__name__)


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_parse(state: KGState) -> str:
    return "error_end" if state.get("phase") == "error" else "translate"


def _route_after_translate(state: KGState) -> str:
    return "error_end" if state.get("phase") == "error" else "execute"


def _error_end_node(state: KGState) -> KGState:
    logger.error("KG pipeline terminating: %s", state.get("errors"))
    state["phase"] = "error"
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    graph = StateGraph(KGState)
    graph.add_node("parse",     parse_node)
    graph.add_node("translate", translate_node)
    graph.add_node("execute",   execute_node)
    graph.add_node("error_end", _error_end_node)

    graph.add_edge(START, "parse")
    graph.add_conditional_edges(
        "parse",
        _route_after_parse,
        {"translate": "translate", "error_end": "error_end"},
    )
    graph.add_conditional_edges(
        "translate",
        _route_after_translate,
        {"execute": "execute", "error_end": "error_end"},
    )
    graph.add_edge("execute",   END)
    graph.add_edge("error_end", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

class KGAgent:
    """
    Convert an OWL/RDF ontology to a Cypher or Gremlin knowledge graph.

    Usage::

        cfg    = KGConfig(graph_type="neo4j", neo4j_uri="bolt://localhost:7687",
                          neo4j_username="neo4j", neo4j_password="secret")
        agent  = KGAgent(cfg)
        result = agent.run(turtle_text)
        # result["queries"]    — list of Cypher/Gremlin statements
        # result["graph_data"] — {nodes, edges} for visualisation
        # result["node_count"] — number of OWL classes converted
        # result["edge_count"] — number of object properties converted
    """

    def __init__(self, config: Optional[KGConfig] = None):
        self._config = config or KGConfig()
        self._graph  = _build_graph()

    def _initial_state(self, ontology_text: str, ontology_format: str = "turtle") -> KGState:
        return {
            "config":            self._config,
            "ontology_text":     ontology_text,
            "ontology_format":   ontology_format,
            "ontology_graph":    None,
            "queries":           [],
            "graph_data":        {"nodes": [], "edges": []},
            "execution_results": [],
            "node_count":        0,
            "edge_count":        0,
            "executed_count":    0,
            "errors":            [],
            "phase":             "init",
        }

    def run(self, ontology_text: str, ontology_format: str = "turtle") -> Dict:
        """Synchronous execution — returns a result summary dict."""
        final = self._graph.invoke(self._initial_state(ontology_text, ontology_format))
        for err in final.get("errors", []):
            logger.warning("KG warning: %s", err)
        return {
            "queries":           final.get("queries", []),
            "graph_data":        final.get("graph_data", {"nodes": [], "edges": []}),
            "node_count":        final.get("node_count", 0),
            "edge_count":        final.get("edge_count", 0),
            "executed_count":    final.get("executed_count", 0),
            "execution_results": final.get("execution_results", []),
            "errors":            final.get("errors", []),
            "phase":             final.get("phase", ""),
        }

    def stream_run(self, ontology_text: str, ontology_format: str = "turtle"):
        """Yield (node_name, state_update) as each pipeline node completes."""
        for event in self._graph.stream(self._initial_state(ontology_text, ontology_format)):
            for node_name, state_update in event.items():
                yield node_name, state_update
