"""LangGraph state definition for the KG Conformity Agent."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class KGSnapshot(TypedDict):
    """One KG as submitted to the analyse endpoint."""
    kg_id:  str          # caller-supplied label, e.g. "kg_job_abc123"
    nodes:  List[Dict]   # graph_data["nodes"] from kg_api /jobs/{id}/graph
    edges:  List[Dict]   # graph_data["edges"]


class Conformity(TypedDict):
    """One detected match between a node in KG-A and a node in KG-B."""
    index:          int         # stable 0-based row index for checkbox approval
    kg_a_id:        str
    kg_b_id:        str
    node_a_id:      str         # URI/id of node in KG-A
    node_b_id:      str         # URI/id of node in KG-B
    node_a_label:   str
    node_b_label:   str
    match_type:     str         # "exact" | "fuzzy" | "property_jaccard"
    score:          float       # label similarity 0–1
    node_a_props:   List[str]   # property names from title tooltip
    node_b_props:   List[str]
    jaccard:        float       # property Jaccard (always computed)


class ConformityState(TypedDict, total=False):
    # Inputs
    config:           Any               # ConformityConfig
    kg_snapshots:     List[KGSnapshot]

    # analyse_node output
    conformities:     List[Conformity]
    exact_count:      int
    fuzzy_count:      int
    jaccard_count:    int

    # recommend_node output
    recommendations:  str               # LLM markdown narrative

    # stitch_node inputs (injected before stitch run)
    approved_indices: List[int]

    # stitch_node output
    super_graph:      Dict              # {nodes: [...], edges: [...]}
    stitch_log:       List[str]

    # Common
    errors:           List[str]
    phase:            str   # "init"|"analysed"|"recommended"|"stitched"|"error"
