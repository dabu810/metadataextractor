"""
Conformity Agent node: detect conformed nodes across multiple KG snapshots.

Three detection strategies (in order of confidence):
  1. exact          — normalised labels are identical
  2. fuzzy          — rapidfuzz token_sort_ratio ≥ config.fuzzy_threshold
  3. property_jaccard — property-name Jaccard similarity ≥ config.jaccard_threshold

Only the highest-scoring match per (node_a, node_b KG pair) is kept.
"""
from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Dict, List, Set, Tuple

from ..config import ConformityConfig
from ..state import Conformity, ConformityState, KGSnapshot

logger = logging.getLogger(__name__)


# ── Property extraction from tooltip title ─────────────────────────────────────

_PROPS_RE = re.compile(r"Properties:\n([\s\S]*?)(?:\n\n|$)")


def _parse_props(title: str) -> List[str]:
    """Extract property names from a KG node's title tooltip."""
    m = _PROPS_RE.search(title or "")
    if not m:
        return []
    block = m.group(1)
    props = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        name = stripped.split(":")[0].strip()
        if name:
            props.append(name.lower())
    return props


# ── Label normalisation & similarity ─────────────────────────────────────────

def _norm(label: str) -> str:
    """Lowercase, strip punctuation — used for exact comparison."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _label_sim(a: str, b: str) -> Tuple[str, float]:
    """Return (match_type, score) for two node labels."""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return "exact", 1.0
    try:
        from rapidfuzz import fuzz  # noqa: PLC0415
        score = fuzz.token_sort_ratio(a.lower(), b.lower())  # 0-100
        return "fuzzy", score / 100.0
    except ImportError:
        # Graceful fallback: simple ratio using stdlib difflib
        from difflib import SequenceMatcher  # noqa: PLC0415
        score = SequenceMatcher(None, na, nb).ratio()
        return "fuzzy", score


def _jaccard(props_a: List[str], props_b: List[str]) -> float:
    sa: Set[str] = set(props_a)
    sb: Set[str] = set(props_b)
    union = sa | sb
    if not union:
        return 1.0  # both empty → structurally identical
    return len(sa & sb) / len(union)


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_node(state: ConformityState) -> ConformityState:
    logger.info("=== analyse_node ===")
    snapshots: List[KGSnapshot] = state.get("kg_snapshots") or []
    config: ConformityConfig = state["config"]

    if len(snapshots) < 2:
        state["errors"] = state.get("errors") or []
        state["errors"].append("analyse_node: at least 2 KG snapshots are required")
        state["conformities"] = []
        state["exact_count"] = 0
        state["fuzzy_count"] = 0
        state["jaccard_count"] = 0
        state["phase"] = "error"
        return state

    all_conformities: List[Conformity] = []
    idx = 0

    # Compare every pair of KGs
    for snap_a, snap_b in combinations(snapshots, 2):
        nodes_a = snap_a["nodes"]
        nodes_b = snap_b["nodes"]

        # Guard against O(N²) explosion
        if len(nodes_a) * len(nodes_b) > config.max_node_pairs:
            logger.warning(
                "Pair (%s, %s): %d × %d = %d pairs exceeds max_node_pairs=%d; skipping fuzzy/jaccard",
                snap_a["kg_id"], snap_b["kg_id"],
                len(nodes_a), len(nodes_b), len(nodes_a) * len(nodes_b),
                config.max_node_pairs,
            )
            # Only do exact matching when the pair is too large
            _compare_pair(
                snap_a, snap_b, nodes_a, nodes_b, config,
                all_conformities, idx, exact_only=True,
            )
        else:
            _compare_pair(
                snap_a, snap_b, nodes_a, nodes_b, config,
                all_conformities, idx, exact_only=False,
            )

        idx = len(all_conformities)

    # Re-index sequentially (index is used as checkbox key)
    for i, c in enumerate(all_conformities):
        c["index"] = i

    exact   = sum(1 for c in all_conformities if c["match_type"] == "exact")
    fuzzy   = sum(1 for c in all_conformities if c["match_type"] == "fuzzy")
    jaccard = sum(1 for c in all_conformities if c["match_type"] == "property_jaccard")

    logger.info(
        "analyse_node: %d conformities found (exact=%d, fuzzy=%d, jaccard=%d)",
        len(all_conformities), exact, fuzzy, jaccard,
    )

    state["conformities"]  = all_conformities
    state["exact_count"]   = exact
    state["fuzzy_count"]   = fuzzy
    state["jaccard_count"] = jaccard
    state["phase"]         = "analysed"
    return state


def _compare_pair(
    snap_a: KGSnapshot,
    snap_b: KGSnapshot,
    nodes_a: List[Dict],
    nodes_b: List[Dict],
    config: ConformityConfig,
    out: List[Conformity],
    start_idx: int,
    exact_only: bool,
) -> None:
    """Append conformity candidates for one (KG-A, KG-B) pair to `out`."""
    # Track which node_a IDs already have a match to avoid duplicates
    matched_a: Dict[str, float] = {}   # node_a_id → best score so far

    for na in nodes_a:
        a_id    = na.get("id", "")
        a_label = na.get("label", "")
        a_props = _parse_props(na.get("title", ""))

        for nb in nodes_b:
            b_id    = nb.get("id", "")
            b_label = nb.get("label", "")
            b_props = _parse_props(nb.get("title", ""))

            j = _jaccard(a_props, b_props)

            # ── Strategy 1: exact label ────────────────────────────────────
            if _norm(a_label) == _norm(b_label):
                match_type, score = "exact", 1.0

            # ── Strategy 2: fuzzy label ────────────────────────────────────
            elif not exact_only:
                _, raw_score = _label_sim(a_label, b_label)
                if raw_score * 100 >= config.fuzzy_threshold:
                    match_type, score = "fuzzy", raw_score
                # ── Strategy 3: property Jaccard ───────────────────────────
                elif j >= config.jaccard_threshold:
                    match_type, score = "property_jaccard", j
                else:
                    continue
            else:
                continue

            # Keep only the best match per node_a
            prev_best = matched_a.get(a_id, -1.0)
            if score <= prev_best:
                continue

            # Remove any previous lower-score entry for this node_a
            out[:] = [c for c in out if not (c["node_a_id"] == a_id
                                              and c["kg_a_id"] == snap_a["kg_id"]
                                              and c["kg_b_id"] == snap_b["kg_id"])]

            matched_a[a_id] = score
            out.append(Conformity(
                index        = start_idx + len(out),
                kg_a_id      = snap_a["kg_id"],
                kg_b_id      = snap_b["kg_id"],
                node_a_id    = a_id,
                node_b_id    = b_id,
                node_a_label = a_label,
                node_b_label = b_label,
                match_type   = match_type,
                score        = round(score, 4),
                node_a_props = a_props,
                node_b_props = b_props,
                jaccard      = round(j, 4),
            ))
