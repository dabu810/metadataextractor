"""
Configuration for the Ontology Agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OntologyConfig:
    """
    base_uri          : Base URI for the ontology namespace.
    ontology_name     : Human-readable name embedded in the OWL header.
    output_path       : Where to write the serialised ontology file.
                        If None the file is not written to disk.
    serialize_format  : "turtle" (.ttl) | "xml" (.owl) | "n3"
    include_statistics: Annotate datatype properties with column stats
                        (unique count, null count, min/max) as rdfs:comment.
    """
    base_uri:           str  = "http://metadata-agent.io/ontology/"
    ontology_name:      str  = "DatabaseOntology"
    output_path:        Optional[str] = None
    serialize_format:   str  = "turtle"
    include_statistics: bool = True
