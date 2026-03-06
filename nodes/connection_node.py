"""
LangGraph node: initialise the database connection and validate it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..connectors import get_connector
from ..state import AgentState

logger = logging.getLogger(__name__)


def connection_node(state: AgentState) -> AgentState:
    """
    Opens the database connection and stores the live connector in state.
    On failure, logs the error and marks phase = 'error'.
    """
    config = state["agent_config"]
    db_cfg = config.db_config

    logger.info(
        "Connecting to %s at %s/%s",
        db_cfg.db_type.value,
        db_cfg.host or "n/a",
        db_cfg.database or db_cfg.schema or "n/a",
    )

    try:
        connector = get_connector(db_cfg)
        connector.connect()

        # Quick smoke-test: execute a trivial query
        connector.execute("SELECT 1 AS ok")

        state["connector"] = connector
        state["phase"] = "connected"
        logger.info("Connection established successfully.")
    except Exception as exc:
        err = f"Connection failed: {exc}"
        logger.error(err)
        state["errors"].append(err)
        state["phase"] = "error"

    return state
