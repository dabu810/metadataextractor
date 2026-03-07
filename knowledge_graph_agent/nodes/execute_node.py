"""
Knowledge Graph pipeline node: execute generated Cypher or Gremlin statements
on the target graph database.

If the connection URI/URL is empty in the config, execution is skipped and
the node reports success — the graph schema and queries are still returned
to the caller for manual execution or preview.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..state import KGState

logger = logging.getLogger(__name__)


def execute_node(state: KGState) -> KGState:
    config  = state["config"]
    queries = state.get("queries") or []

    # Determine whether we should execute
    should_execute = (
        (config.graph_type == "neo4j"   and bool(config.neo4j_uri)) or
        (config.graph_type == "gremlin" and bool(config.gremlin_url))
    )

    if not should_execute:
        logger.info("No connection URI provided — skipping execution (preview mode).")
        state["execution_results"] = []
        state["executed_count"]    = 0
        state["phase"]             = "executed"
        return state

    try:
        if config.graph_type == "neo4j":
            results, count = _execute_neo4j(config, queries)
        else:
            results, count = _execute_gremlin(config, queries)

        state["execution_results"] = results
        state["executed_count"]    = count
        state["phase"]             = "executed"
        logger.info("Executed %d/%d queries successfully.", count, len(queries))

    except Exception as exc:
        logger.exception("Graph DB execution failed")
        state["errors"].append(f"Execution failed: {exc}")
        # Keep phase as "executed" so graph_data + queries remain accessible
        state["execution_results"] = []
        state["executed_count"]    = 0
        state["phase"]             = "executed"

    return state


# ── Neo4j ─────────────────────────────────────────────────────────────────────

def _execute_neo4j(config: Any, queries: List[str]):
    from neo4j import GraphDatabase  # noqa: PLC0415

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )
    results: List[Dict] = []
    count = 0
    try:
        with driver.session(database=config.neo4j_database) as session:
            for q in queries:
                try:
                    session.run(q)
                    results.append({"query": q[:120], "ok": True, "error": None})
                    count += 1
                except Exception as exc:
                    results.append({"query": q[:120], "ok": False, "error": str(exc)})
                    logger.warning("Cypher query failed: %s — %s", q[:80], exc)
    finally:
        driver.close()

    return results, count


# ── Gremlin / TinkerPop ───────────────────────────────────────────────────────

def _execute_gremlin(config: Any, queries: List[str]):
    from gremlin_python.driver import client as gremlin_client  # noqa: PLC0415

    gc = gremlin_client.Client(
        config.gremlin_url,
        config.gremlin_traversal_source,
    )
    results: List[Dict] = []
    count = 0
    try:
        for q in queries:
            try:
                gc.submit(q).all().result()
                results.append({"query": q[:120], "ok": True, "error": None})
                count += 1
            except Exception as exc:
                results.append({"query": q[:120], "ok": False, "error": str(exc)})
                logger.warning("Gremlin query failed: %s — %s", q[:80], exc)
    finally:
        gc.close()

    return results, count
