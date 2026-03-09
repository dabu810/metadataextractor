"""
understand_node — Build a structured schema context from KG nodes/edges.

No LLM call here; this node transforms the raw graph data (nodes = classes /
tables, edges = relationships) into a concise, human-readable schema description
that downstream nodes feed to the LLM.

Key design:
- The KG node `label` = original table name (via rdfs:label in the ontology).
- The KG node `title` tooltip contains column names and types.
- We extract these and present them with the correct schema-qualified name
  (e.g. `public.orders`) so the LLM generates runnable SQL on the first try.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..config import DialogConfig
from ..state import DialogState

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_columns_from_title(title: str, table_label: str) -> List[str]:
    """
    Parse the KG node title tooltip to get a list of column definitions.

    KG translate_node formats the title as:
        Class: <name>
        <rdfs:comments...>

        Properties:
          col1: xsd_type1
          col2: xsd_type2
    """
    cols: List[str] = []
    in_props = False
    for line in title.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("properties:"):
            in_props = True
            continue
        if in_props:
            # Lines look like "  col_name: integer"
            # Stop if we hit something that looks like a new section
            if stripped.startswith("Class:") or stripped.startswith("["):
                in_props = False
            elif ":" in stripped:
                cols.append(stripped)
    return cols


def _extract_comments_from_title(title: str) -> List[str]:
    """
    Return the rdfs:comment lines from the node title (row_count, FD hints etc.)
    Skips the first 'Class: X' line and the 'Properties:' block.
    """
    comments: List[str] = []
    in_props = False
    for line in title.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("class:"):
            continue
        if stripped.lower().startswith("properties:"):
            in_props = True
            continue
        if in_props:
            break
        comments.append(stripped)
    return comments


def _qualified(schema: str, table: str) -> str:
    """Return `schema.table` if schema is set, else just `table`."""
    return f"{schema}.{table}" if schema else table


def _summarise_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    db_schema: str,
) -> str:
    """
    Convert KG nodes/edges to a schema description the SQL planner can use.

    Produces two sections:
      1. QUICK REFERENCE — a compact table list with exact qualified names
      2. DETAILED SCHEMA — column names, types, and relationships per table
    """
    if not nodes:
        return (
            "(No schema context available from the knowledge graph. "
            "Generate SQL based on the natural language question alone.)"
        )

    # Index edges by source node id
    edges_by_src: Dict[str, List[Dict]] = {}
    for e in edges:
        src = e.get("from", "")
        edges_by_src.setdefault(src, []).append(e)

    # Build an index of node id → label for edge target lookup
    node_label_by_id: Dict[str, str] = {
        n.get("id", ""): n.get("label", "") for n in nodes
    }

    lines: List[str] = []

    # ── Section 1: Quick reference ────────────────────────────────────────────
    lines.append("DATABASE SCHEMA CONTEXT")
    lines.append("=" * 60)
    if db_schema:
        lines.append(f"Schema: {db_schema}")
        lines.append(
            "IMPORTANT: Always write table names as "
            f"`{db_schema}.tablename` in your SQL queries."
        )
    lines.append("")

    lines.append("AVAILABLE TABLES (use these exact qualified names in SQL):")
    for node in nodes:
        label = node.get("label", "")
        if label:
            lines.append(f"  - {_qualified(db_schema, label)}")
    lines.append("")

    # ── Section 2: Detailed schema per table ──────────────────────────────────
    lines.append("DETAILED SCHEMA:")
    lines.append("-" * 60)

    for node in nodes:
        node_id   = node.get("id", "")
        label     = node.get("label", node_id)
        title     = node.get("title", "")

        qualified_name = _qualified(db_schema, label)
        lines.append(f"\nTable: {qualified_name}")

        # Metadata comments (row_count, FD hints)
        for comment in _extract_comments_from_title(title):
            lines.append(f"  -- {comment}")

        # Columns
        cols = _extract_columns_from_title(title, label)
        if cols:
            lines.append("  Columns:")
            for col in cols:
                lines.append(f"    {col}")

        # Foreign-key relationships (edges)
        for edge in edges_by_src.get(node_id, []):
            tgt_label = node_label_by_id.get(edge.get("to", ""), edge.get("to", "?"))
            tgt_qualified = _qualified(db_schema, tgt_label)
            edge_title = edge.get("title", "")
            rel_lbl    = edge.get("label", "")
            lines.append(
                f"  FK: {rel_lbl} -> {tgt_qualified}  ({edge_title})"
            )

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ── node ──────────────────────────────────────────────────────────────────────

def understand_node(state: DialogState) -> DialogState:
    """Build schema_context from KG graph data, qualified with the target schema."""
    logger.info("=== understand_node ===")

    nodes     = state.get("kg_nodes") or []
    edges     = state.get("kg_edges") or []
    config: DialogConfig = state["config"]
    db_schema = config.db_schema or ""

    schema_context = _summarise_graph(nodes, edges, db_schema)

    logger.info(
        "Schema context built: %d chars, %d tables, %d relationships, schema=%r",
        len(schema_context), len(nodes), len(edges), db_schema,
    )

    state["schema_context"] = schema_context
    state["phase"] = "understand"
    return state
