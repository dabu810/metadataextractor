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

_FILE_BASED_TYPES = {"sqlite", "csv", "excel"}

# Max distinct sample values shown per column in the schema context
_SAMPLE_LIMIT = 8
# Only sample columns whose distinct count is <= this (to skip IDs/timestamps)
_SAMPLE_DISTINCT_MAX = 200


def _to_sql_col(name: str) -> str:
    """
    Sanitize a column name to a valid SQLite identifier.
    Must match execute_node._safe_col exactly so the schema context
    uses the same column names that are actually stored in SQLite.
    """
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return ("col_" + s if s and s[0].isdigit() else s) or "col"


def _to_sql_table(name: str) -> str:
    """Sanitize a table/sheet name to a valid SQLite identifier."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return ("t_" + s if s and s[0].isdigit() else s) or "tbl"


def _sample_file_data(config: DialogConfig) -> Dict[str, Dict[str, List]]:
    """
    Load the file-based source and collect up to _SAMPLE_LIMIT distinct non-null
    values for every column in every table.  Returns:
        { sql_table_name: { sql_col_name: [val1, val2, ...] } }
    Keys use the sanitized (SQL-safe) names that execute_node actually stores.
    Returns empty dict on any failure.
    """
    import sqlite3

    db = config.db_type.lower()
    fpath = config.db_file_path
    if not fpath:
        return {}

    try:
        if db == "sqlite":
            conn = sqlite3.connect(fpath, check_same_thread=False)
        else:
            import pandas as pd
            conn = sqlite3.connect(":memory:", check_same_thread=False)

            if db == "csv":
                from pathlib import Path
                for f in sorted(Path(fpath).glob("*.csv")):
                    try:
                        df = pd.read_csv(f)
                        used: dict = {}
                        new_cols = []
                        for c in df.columns:
                            sc = _to_sql_col(str(c))
                            if sc in used:
                                used[sc] += 1
                                sc = f"{sc}_{used[sc]}"
                            else:
                                used[sc] = 1
                            new_cols.append(sc)
                        df.columns = new_cols
                        df.to_sql(f.stem, conn, if_exists="replace", index=False)
                    except Exception:
                        pass
            else:  # excel
                xl = pd.ExcelFile(fpath)
                used_sheets: dict = {}
                for sheet in xl.sheet_names:
                    base = _to_sql_table(sheet)
                    if base in used_sheets:
                        used_sheets[base] += 1
                        safe_sheet = f"{base}_{used_sheets[base]}"
                    else:
                        used_sheets[base] = 1
                        safe_sheet = base
                    try:
                        df = xl.parse(sheet)
                        used_cols: dict = {}
                        new_cols = []
                        for c in df.columns:
                            sc = _to_sql_col(str(c))
                            if sc in used_cols:
                                used_cols[sc] += 1
                                sc = f"{sc}_{used_cols[sc]}"
                            else:
                                used_cols[sc] = 1
                            new_cols.append(sc)
                        df.columns = new_cols
                        df.to_sql(safe_sheet, conn, if_exists="replace", index=False)
                    except Exception:
                        pass

        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]

        samples: Dict[str, Dict[str, List]] = {}
        for tbl in tables:
            try:
                cur.execute(f'PRAGMA table_info("{tbl}")')
                cols = [r[1] for r in cur.fetchall()]
                col_samples: Dict[str, List] = {}
                for col in cols:
                    try:
                        cur.execute(
                            f'SELECT COUNT(DISTINCT "{col}") FROM "{tbl}"'
                        )
                        distinct_count = (cur.fetchone() or [0])[0]
                        if distinct_count > _SAMPLE_DISTINCT_MAX:
                            continue  # skip high-cardinality columns (IDs etc.)
                        cur.execute(
                            f'SELECT DISTINCT "{col}" FROM "{tbl}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT {_SAMPLE_LIMIT}'
                        )
                        vals = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
                        if vals:
                            col_samples[col] = vals
                    except Exception:
                        pass
                if col_samples:
                    samples[tbl] = col_samples
            except Exception:
                pass

        conn.close()
        return samples

    except Exception as exc:
        logger.warning("understand_node: sample_file_data failed — %s", exc)
        return {}


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
    samples: Optional[Dict[str, Dict[str, List]]] = None,
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
            # For file-based sources, look up sample values using the sanitized
            # table name, since that is what is actually stored in SQLite.
            sql_table = _to_sql_table(label) if samples is not None else label
            tbl_samples = (samples or {}).get(sql_table, {})
            for col in cols:
                original_col = col.split(":")[0].strip()
                col_type     = col[len(original_col):].strip()  # e.g. ": integer"
                # Translate original KG column name → SQL-safe name used in SQLite
                sql_col = _to_sql_col(original_col) if samples is not None else original_col
                sample_vals = tbl_samples.get(sql_col)
                display = f"{sql_col}{col_type}" if samples is not None else col
                if sample_vals:
                    lines.append(f"    {display}  [sample values: {', '.join(repr(v) for v in sample_vals)}]")
                else:
                    lines.append(f"    {display}")

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

    # For file-based sources, inject sample values so the LLM sees actual data
    # values (e.g. exact company names) and generates accurate WHERE clauses.
    samples: Optional[Dict[str, Dict[str, List]]] = None
    if config.db_type.lower() in _FILE_BASED_TYPES and config.db_file_path:
        samples = _sample_file_data(config)
        if samples:
            logger.info("understand_node: sampled %d tables for value hints", len(samples))

    schema_context = _summarise_graph(nodes, edges, db_schema, samples)

    logger.info(
        "Schema context built: %d chars, %d tables, %d relationships, schema=%r",
        len(schema_context), len(nodes), len(edges), db_schema,
    )

    state["schema_context"] = schema_context
    state["phase"] = "understand"
    return state
