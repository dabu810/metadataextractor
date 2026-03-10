"""
Conformity Agent node: merge approved conformities into a super knowledge graph.

Procedure:
  1. Build id_remap: source node ID → canonical ID in super-graph
     - Conformed node pairs → CONFORMED::<normalised_label>
     - All others         → <kg_id>::<orig_id>  (namespace to avoid collisions)
  2. Emit conformed (merged) nodes — amber colour, merged property list
  3. Emit non-conformed nodes     — original colour, namespaced IDs
  4. Rewire all edges through id_remap; deduplicate by (from, to, label)
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Set, Tuple

from ..state import Conformity, ConformityState, KGSnapshot

logger = logging.getLogger(__name__)


def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", label.lower()).strip("_")


def _parse_props_from_title(title: str) -> List[str]:
    import re as _re  # noqa: PLC0415
    m = _re.search(r"Properties:\n([\s\S]*?)(?:\n\n|$)", title or "")
    if not m:
        return []
    lines = []
    for line in m.group(1).splitlines():
        s = line.strip()
        if s:
            lines.append(s)
    return lines


def _build_merged_title(label: str, kg_ids: List[str], prop_lines: List[str]) -> str:
    src = ", ".join(kg_ids)
    title = f"Class: {label}\n[CONFORMED from: {src}]"
    if prop_lines:
        title += "\n\nProperties:\n" + "\n".join(f"  {p}" for p in prop_lines)
    return title


def stitch_node(state: ConformityState) -> ConformityState:
    logger.info("=== stitch_node ===")
    snapshots: List[KGSnapshot]  = state.get("kg_snapshots") or []
    conformities: List[Conformity] = state.get("conformities") or []
    approved_indices: List[int]  = state.get("approved_indices") or []
    errors: List[str]            = list(state.get("errors") or [])
    log: List[str]               = []

    if not snapshots:
        errors.append("stitch_node: no KG snapshots provided")
        state["errors"] = errors
        state["phase"]  = "error"
        return state

    approved = {c["index"]: c for c in conformities if c["index"] in approved_indices}

    # ── Step 1: Build id_remap ─────────────────────────────────────────────────
    # canonical_id → {canonical_label, kg_ids, prop_lines from each source}
    conformed_nodes: Dict[str, Dict] = {}
    id_remap: Dict[str, str] = {}   # (kg_id + "||" + orig_id) → canonical_id

    for c in approved.values():
        canon_label = c["node_a_label"]          # node_a is the "authority" label
        canon_id    = f"CONFORMED::{_norm(canon_label)}"

        for kg_id, node_id, label, props in [
            (c["kg_a_id"], c["node_a_id"], c["node_a_label"], c["node_a_props"]),
            (c["kg_b_id"], c["node_b_id"], c["node_b_label"], c["node_b_props"]),
        ]:
            remap_key = f"{kg_id}||{node_id}"
            id_remap[remap_key] = canon_id

        if canon_id not in conformed_nodes:
            conformed_nodes[canon_id] = {
                "canonical_label": canon_label,
                "kg_ids":          [],
                "prop_lines_seen": set(),
                "prop_lines":      [],
            }

        rec = conformed_nodes[canon_id]
        for kg_id in (c["kg_a_id"], c["kg_b_id"]):
            if kg_id not in rec["kg_ids"]:
                rec["kg_ids"].append(kg_id)

        # Merge properties (union, dedup, order-stable)
        all_props = c["node_a_props"] + c["node_b_props"]
        for p in all_props:
            if p not in rec["prop_lines_seen"]:
                rec["prop_lines_seen"].add(p)
                rec["prop_lines"].append(p)

        log.append(
            f"Merged '{c['node_a_label']}' ({c['kg_a_id']}) + "
            f"'{c['node_b_label']}' ({c['kg_b_id']}) → {canon_id}"
        )

    # ── Step 2: Namespace non-conformed node IDs ───────────────────────────────
    for snap in snapshots:
        for node in snap["nodes"]:
            key = f"{snap['kg_id']}||{node['id']}"
            if key not in id_remap:
                id_remap[key] = f"{snap['kg_id']}::{node['id']}"

    # Resolve helper
    def _resolve(kg_id: str, orig_id: str) -> str:
        return id_remap.get(f"{kg_id}||{orig_id}", f"{kg_id}::{orig_id}")

    # ── Step 3: Emit conformed (merged) nodes ──────────────────────────────────
    super_nodes: List[Dict] = []
    emitted_ids: Set[str]   = set()

    for canon_id, rec in conformed_nodes.items():
        title = _build_merged_title(
            rec["canonical_label"], rec["kg_ids"], rec["prop_lines"]
        )
        super_nodes.append({
            "id":    canon_id,
            "label": rec["canonical_label"],
            "title": title,
            "color": "#f6ad55",   # amber — visually distinct from regular nodes
            "size":  28,
        })
        emitted_ids.add(canon_id)

    # ── Step 4: Emit non-conformed nodes (namespaced) ─────────────────────────
    for snap in snapshots:
        for node in snap["nodes"]:
            canon = _resolve(snap["kg_id"], node["id"])
            if canon in emitted_ids:
                continue
            emitted_ids.add(canon)
            super_nodes.append({
                "id":    canon,
                "label": node.get("label", ""),
                "title": node.get("title", ""),
                "color": node.get("color", "#63b3ed"),
                "size":  node.get("size", 20),
            })

    # ── Step 5: Rewire and deduplicate edges ───────────────────────────────────
    seen_edges: Set[Tuple[str, str, str]] = set()
    super_edges: List[Dict] = []

    for snap in snapshots:
        for edge in snap["edges"]:
            new_from = _resolve(snap["kg_id"], edge.get("from", ""))
            new_to   = _resolve(snap["kg_id"], edge.get("to", ""))
            lbl      = edge.get("label", "")

            key = (new_from, new_to, lbl)
            if key in seen_edges:
                continue
            seen_edges.add(key)

            # Cross-KG stitched edges get a purple colour
            from_kg = snap["kg_id"] in new_from or new_from.startswith("CONFORMED")
            to_kg   = snap["kg_id"] in new_to   or new_to.startswith("CONFORMED")
            color   = "#b794f4" if new_from.startswith("CONFORMED") or new_to.startswith("CONFORMED") else "#68d391"

            super_edges.append({
                "from":  new_from,
                "to":    new_to,
                "label": lbl,
                "title": edge.get("title", lbl),
                "color": {"color": color, "highlight": color},
            })

    logger.info(
        "stitch_node: super-graph has %d nodes, %d edges; %d conformities applied",
        len(super_nodes), len(super_edges), len(approved),
    )

    state["super_graph"] = {"nodes": super_nodes, "edges": super_edges}
    state["stitch_log"]  = log
    state["errors"]      = errors
    state["phase"]       = "stitched"
    return state
