"""
LangGraph node: extract schema + statistics for every discovered table.

For each table this node:
  1. Calls SchemaExtractorTool  → columns, PKs, FKs, indexes
  2. Calls MetadataCollectorTool → row count, size, per-column stats
  3. Stores a TableMeta object in state['table_metadata']
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from ..state import AgentState, ColumnMeta, TableMeta
from ..tools import MetadataCollectorTool, SchemaExtractorTool

logger = logging.getLogger(__name__)


def extraction_node(state: AgentState) -> AgentState:
    config = state["agent_config"]
    connector = state["connector"]

    schema_tool = SchemaExtractorTool(connector=connector)
    meta_tool = MetadataCollectorTool(connector=connector)

    for schema_name, table_name in state["all_tables"]:
        if table_name in state["tables_done"]:
            continue

        logger.info("Extracting metadata for %s.%s …", schema_name, table_name)
        try:
            # --- Schema extraction ---
            schema_json = schema_tool._run(schema_name, table_name)
            schema_data: Dict[str, Any] = json.loads(schema_json)
            if "error" in schema_data:
                raise RuntimeError(schema_data["error"])

            col_names = [c["name"] for c in schema_data.get("columns", [])]

            # --- Statistics ---
            stats_json = meta_tool._run(
                schema_name, table_name, col_names, config.sample_size
            )
            stats_data: Dict[str, Any] = json.loads(stats_json)

            col_stats: Dict[str, Any] = stats_data.get("column_stats", {})

            # Build ColumnMeta objects
            col_metas: List[ColumnMeta] = []
            for col in schema_data.get("columns", []):
                cname = col["name"]
                cs = col_stats.get(cname, {})
                col_metas.append(
                    ColumnMeta(
                        name=cname,
                        data_type=col.get("data_type", ""),
                        nullable=col.get("nullable", True),
                        is_primary_key=col.get("is_primary_key", False),
                        is_foreign_key=col.get("is_foreign_key", False),
                        fk_references=col.get("fk_references"),
                        unique_count=cs.get("unique_count"),
                        null_count=cs.get("null_count"),
                        row_count=cs.get("row_count"),
                        min_value=cs.get("min_value"),
                        max_value=cs.get("max_value"),
                        avg_value=cs.get("avg_value"),
                        stddev_value=cs.get("stddev_value"),
                        top_values=cs.get("top_values"),
                    )
                )

            table_meta = TableMeta(
                schema_name=schema_name,
                table_name=table_name,
                row_count=stats_data.get("row_count"),
                size_bytes=stats_data.get("size_bytes"),
                columns=col_metas,
                primary_keys=schema_data.get("primary_keys", []),
                foreign_keys=schema_data.get("foreign_keys", []),
                indexes=schema_data.get("indexes", []),
                table_comment=schema_data.get("table_comment"),
                create_time=schema_data.get("create_time"),
                last_modified=schema_data.get("last_modified"),
                partitioned_by=schema_data.get("partitioned_by", []),
            )

            state["table_metadata"][table_name] = table_meta
            state["tables_done"].add(table_name)
            logger.info(
                "  → %d columns, %s rows",
                len(col_metas),
                f"{table_meta.row_count:,}" if table_meta.row_count else "unknown",
            )

        except Exception as exc:
            err = f"Extraction failed for {schema_name}.{table_name}: {exc}"
            logger.error(err)
            state["errors"].append(err)

    state["phase"] = "extracted"
    return state
