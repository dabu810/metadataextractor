"""
Configuration dataclass for the Knowledge Graph Agent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KGConfig:
    # Target graph database type
    graph_type: str = "neo4j"          # "neo4j" | "gremlin"

    # ── Neo4j settings ────────────────────────────────────────────────────────
    neo4j_uri:      str = ""           # e.g. "bolt://localhost:7687" — empty = skip execution
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # ── Gremlin / TinkerPop settings ──────────────────────────────────────────
    gremlin_url:               str = ""    # e.g. "ws://localhost:8182/gremlin" — empty = skip
    gremlin_traversal_source:  str = "g"

    # ── Behaviour ─────────────────────────────────────────────────────────────
    clear_existing: bool = False       # Drop all existing vertices/edges before loading
    batch_size:     int  = 50          # Queries executed per batch (for progress tracking)
