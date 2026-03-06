"""
LangGraph node: discover all tables in the target schema.
Filters to agent_config.target_tables if provided.
"""
from __future__ import annotations

import logging
from typing import Any

from ..state import AgentState

logger = logging.getLogger(__name__)


def discovery_node(state: AgentState) -> AgentState:
    """
    Populates state['all_tables'] with (schema, table) tuples.
    """
    config = state["agent_config"]
    connector = state["connector"]
    schema = config.db_config.schema or config.db_config.database or ""

    try:
        all_tables = connector.list_tables(schema)
        logger.info("Discovered %d tables in schema '%s'.", len(all_tables), schema)

        # Apply filter if caller specified target_tables
        if config.target_tables:
            target_set = {t.lower() for t in config.target_tables}
            all_tables = [
                (s, t) for s, t in all_tables if t.lower() in target_set
            ]
            logger.info("Filtered to %d target tables.", len(all_tables))

        if not all_tables:
            state["errors"].append(f"No tables found in schema '{schema}'.")
            state["phase"] = "error"
            return state

        state["all_tables"] = all_tables
        state["phase"] = "discovered"
    except Exception as exc:
        err = f"Discovery failed: {exc}"
        logger.error(err)
        state["errors"].append(err)
        state["phase"] = "error"

    return state
