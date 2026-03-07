"""
Ontology pipeline node: serialise the rdflib Graph to a file.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ..state import OntologyState

logger = logging.getLogger(__name__)

_FORMAT_EXT = {
    "turtle": ".ttl",
    "xml":    ".owl",
    "n3":     ".n3",
}


def serialize_node(state: OntologyState) -> OntologyState:
    config = state["config"]
    g      = state["ontology_graph"]

    fmt = config.serialize_format.lower()
    if fmt not in _FORMAT_EXT:
        fmt = "turtle"

    turtle_str: str = g.serialize(format=fmt)
    state["ontology_turtle"] = turtle_str

    if config.output_path:
        path = Path(config.output_path)
        # Ensure correct extension
        if path.suffix not in _FORMAT_EXT.values():
            path = path.with_suffix(_FORMAT_EXT[fmt])
        os.makedirs(path.parent, exist_ok=True)
        path.write_text(turtle_str, encoding="utf-8")
        state["output_path"] = str(path)
        logger.info("Ontology written to %s (%d triples)", path, state.get("triple_count", 0))
    else:
        state["output_path"] = ""

    state["phase"] = "serialized"
    return state
