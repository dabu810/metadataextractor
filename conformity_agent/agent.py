"""
Conformity Agent — two LangGraph pipelines:

  Analyse pipeline:  START → analyse_node → recommend_node → END
  Stitch pipeline:   START → stitch_node  → END
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List

from langgraph.graph import END, START, StateGraph

from .config import ConformityConfig
from .nodes import analyse_node, recommend_node, stitch_node
from .state import ConformityState, KGSnapshot

logger = logging.getLogger(__name__)


def _build_analyse_graph() -> Any:
    g = StateGraph(ConformityState)
    g.add_node("analyse",   analyse_node)
    g.add_node("recommend", recommend_node)
    g.add_edge(START,       "analyse")
    g.add_edge("analyse",   "recommend")
    g.add_edge("recommend", END)
    return g.compile()


def _build_stitch_graph() -> Any:
    g = StateGraph(ConformityState)
    g.add_node("stitch", stitch_node)
    g.add_edge(START,    "stitch")
    g.add_edge("stitch", END)
    return g.compile()


_analyse_graph = _build_analyse_graph()
_stitch_graph  = _build_stitch_graph()


class ConformityAgent:
    """Thin wrapper that exposes run() and stream() for each pipeline."""

    def __init__(self, config: ConformityConfig | None = None) -> None:
        self.config = config or ConformityConfig()

    # ── Analyse ─────────────────────────────────────────────────────────────

    def analyse(
        self,
        kg_snapshots: List[KGSnapshot],
    ) -> ConformityState:
        """Run the full analyse → recommend pipeline synchronously."""
        initial: ConformityState = {
            "config":       self.config,
            "kg_snapshots": kg_snapshots,
            "errors":       [],
            "phase":        "init",
        }
        return _analyse_graph.invoke(initial)

    def stream_analyse(
        self,
        kg_snapshots: List[KGSnapshot],
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream events from the analyse pipeline."""
        initial: ConformityState = {
            "config":       self.config,
            "kg_snapshots": kg_snapshots,
            "errors":       [],
            "phase":        "init",
        }
        yield from _analyse_graph.stream(initial)

    # ── Stitch ──────────────────────────────────────────────────────────────

    def stitch(
        self,
        kg_snapshots: List[KGSnapshot],
        conformities: list,
        approved_indices: List[int],
    ) -> ConformityState:
        """Run the stitch pipeline synchronously."""
        initial: ConformityState = {
            "config":           self.config,
            "kg_snapshots":     kg_snapshots,
            "conformities":     conformities,
            "approved_indices": approved_indices,
            "errors":           [],
            "phase":            "init",
        }
        return _stitch_graph.invoke(initial)

    def stream_stitch(
        self,
        kg_snapshots: List[KGSnapshot],
        conformities: list,
        approved_indices: List[int],
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream events from the stitch pipeline."""
        initial: ConformityState = {
            "config":           self.config,
            "kg_snapshots":     kg_snapshots,
            "conformities":     conformities,
            "approved_indices": approved_indices,
            "errors":           [],
            "phase":            "init",
        }
        yield from _stitch_graph.stream(initial)
