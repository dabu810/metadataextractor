"""
Knowledge Graph pipeline node: fetch an existing graph from the database.

Used in "load" mode — skips parse/translate and reads the graph that was
previously written to Neo4j or Gremlin, reconstructing the {nodes, edges}
visualisation data from the live database.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from ..state import KGState

logger = logging.getLogger(__name__)


# ── Neo4j fetch ───────────────────────────────────────────────────────────────

def _fetch_neo4j(config: Any) -> Tuple[Dict, int, int]:
    from neo4j import GraphDatabase  # noqa: PLC0415

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )
    nodes: List[Dict] = []
    edges: List[Dict] = []

    try:
        with driver.session(database=config.neo4j_database) as session:
            # Fetch all KGNode vertices
            for rec in session.run("MATCH (n:KGNode) RETURN n"):
                props = dict(rec["n"].items())
                uri   = props.get("uri", str(rec["n"].id))
                name  = props.get("name", uri)
                desc  = props.get("description", "")
                title = f"Class: {name}"
                if desc:
                    title += f"\n{desc}"

                # Reconstruct property lines from stored node attributes
                dt_lines = []
                skip = {"uri", "name", "type", "description"}
                for k, v in props.items():
                    if k not in skip:
                        dt_lines.append(f"  {k}: {v}")
                if dt_lines:
                    title += "\n\nProperties:\n" + "\n".join(dt_lines)

                nodes.append({
                    "id":    uri,
                    "label": name,
                    "title": title,
                    "color": "#63b3ed",
                    "size":  20 + min(len(dt_lines) * 2, 20),
                })

            # Fetch all relationships between KGNode vertices
            for rec in session.run(
                "MATCH (a:KGNode)-[r]->(b:KGNode) "
                "RETURN a.uri AS from_uri, b.uri AS to_uri, "
                "type(r) AS rel_type, r.name AS rel_name, r.cardinality AS cardinality"
            ):
                rel_name = rec["rel_name"] or rec["rel_type"]
                card     = rec["cardinality"] or "M:N"
                edges.append({
                    "from":  rec["from_uri"],
                    "to":    rec["to_uri"],
                    "label": rel_name,
                    "title": f"{rel_name} ({card})",
                })

    finally:
        driver.close()

    logger.info("Fetched %d nodes, %d edges from Neo4j", len(nodes), len(edges))
    return {"nodes": nodes, "edges": edges}, len(nodes), len(edges)


# ── Gremlin fetch ─────────────────────────────────────────────────────────────

def _fetch_gremlin(config: Any) -> Tuple[Dict, int, int]:
    from gremlin_python.driver import client as gremlin_client  # noqa: PLC0415

    gc = gremlin_client.Client(
        config.gremlin_url,
        config.gremlin_traversal_source,
    )
    nodes: List[Dict] = []
    edges: List[Dict] = []

    try:
        # Fetch all vertices with their properties
        vertices = gc.submit("g.V().valueMap(true)").all().result()
        for v in vertices:
            uri   = v.get("uri", [None])[0] or str(v.get("id", ""))
            name  = v.get("name", [uri])[0]
            desc  = (v.get("description", [None])[0] or "")
            title = f"Class: {name}"
            if desc:
                title += f"\n{desc}"

            skip = {"uri", "name", "type", "description", "T.id", "T.label"}
            dt_lines = []
            for k, vals in v.items():
                if str(k) not in skip and isinstance(vals, list):
                    for val in vals:
                        dt_lines.append(f"  {k}: {val}")
            if dt_lines:
                title += "\n\nProperties:\n" + "\n".join(dt_lines)

            nodes.append({
                "id":    uri,
                "label": name,
                "title": title,
                "color": "#63b3ed",
                "size":  20 + min(len(dt_lines) * 2, 20),
            })

        # Fetch all edges
        edge_data = gc.submit(
            "g.E().project('from_uri','to_uri','label','card')"
            ".by(outV().values('uri'))"
            ".by(inV().values('uri'))"
            ".by(label())"
            ".by(coalesce(values('cardinality'), constant('M:N')))"
        ).all().result()

        for e in edge_data:
            edges.append({
                "from":  e["from_uri"],
                "to":    e["to_uri"],
                "label": e["label"],
                "title": f"{e['label']} ({e['card']})",
            })

    finally:
        gc.close()

    logger.info("Fetched %d nodes, %d edges from Gremlin", len(nodes), len(edges))
    return {"nodes": nodes, "edges": edges}, len(nodes), len(edges)


# ── Node ──────────────────────────────────────────────────────────────────────

def fetch_node(state: KGState) -> KGState:
    """Retrieve an existing graph from the connected graph database."""
    logger.info("=== fetch_node (load mode) ===")
    config = state["config"]

    connected = (
        (config.graph_type == "neo4j"   and bool(config.neo4j_uri)) or
        (config.graph_type == "gremlin" and bool(config.gremlin_url))
    )
    if not connected:
        state["errors"].append(
            "fetch_node: no graph database URI provided — cannot load existing graph"
        )
        state["phase"] = "error"
        return state

    try:
        if config.graph_type == "neo4j":
            graph_data, nc, ec = _fetch_neo4j(config)
        else:
            graph_data, nc, ec = _fetch_gremlin(config)

        state["graph_data"]        = graph_data
        state["node_count"]        = nc
        state["edge_count"]        = ec
        state["queries"]           = []
        state["execution_results"] = []
        state["executed_count"]    = 0
        state["phase"]             = "fetched"

    except Exception as exc:
        logger.exception("fetch_node: failed to retrieve graph")
        state["errors"].append(f"fetch_node: {exc}")
        state["phase"] = "error"

    return state
