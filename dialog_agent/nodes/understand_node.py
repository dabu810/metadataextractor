"""
understand_node — Build a structured schema context from KG nodes/edges.

No LLM call here; this node transforms the raw graph data (nodes = classes /
tables, edges = relationships) into a concise, human-readable schema description
that downstream nodes feed to the LLM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..state import DialogState

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _summarise_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> str:
    """
    Convert KG nodes/edges to a schema description.

    Node structure from KGAgent: {id, label, title, color, size}
    Edge structure:              {from, to, label, title}

    'title' in the KG nodes contains a tooltip that includes the class name
    and its datatype properties.  We parse that back into structured text.
    """
    if not nodes:
        return "(No schema context available — knowledge graph is empty.)"

    lines: List[str] = []
    lines.append("DATABASE SCHEMA (derived from knowledge graph)\n")
    lines.append("=" * 60)

    # Index edges by source for quick lookup
    edges_by_src: Dict[str, List[Dict]] = {}
    for e in edges:
        src = e.get("from", "")
        edges_by_src.setdefault(src, []).append(e)

    for node in nodes:
        node_id = node.get("id", "")
        label   = node.get("label", node_id)
        title   = node.get("title", "")  # tooltip text with property list

        lines.append(f"\nTable / Class: {label}  (id={node_id})")

        # Parse the tooltip — KG agent formats it as:
        #   "ClassName\nProperties:\n- prop1: type1\n- prop2: type2"
        if title:
            for tline in title.splitlines():
                stripped = tline.strip()
                if stripped and not stripped.startswith(label):
                    lines.append(f"  {stripped}")

        # Outgoing relationships (foreign-key-like edges)
        for edge in edges_by_src.get(node_id, []):
            tgt_node = next((n for n in nodes if n["id"] == edge.get("to")), None)
            tgt_lbl  = tgt_node["label"] if tgt_node else edge.get("to", "?")
            rel_lbl  = edge.get("label", "relatesTo")
            edge_tip = edge.get("title", "")
            lines.append(f"  [RELATION] {rel_lbl} -> {tgt_lbl}  ({edge_tip})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ── node ──────────────────────────────────────────────────────────────────────

def understand_node(state: DialogState) -> DialogState:
    """Build schema_context from KG graph data."""
    logger.info("=== understand_node ===")

    nodes = state.get("kg_nodes") or []
    edges = state.get("kg_edges") or []

    schema_context = _summarise_graph(nodes, edges)
    logger.info(
        "Schema context built: %d chars, %d tables, %d relationships",
        len(schema_context), len(nodes), len(edges),
    )

    state["schema_context"] = schema_context
    state["phase"] = "understand"
    return state
