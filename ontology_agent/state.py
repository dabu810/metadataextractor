"""
LangGraph state for the Ontology Agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class OntologyState(TypedDict, total=False):
    """
    config            : OntologyConfig instance
    report            : raw metadata report dict (from MetadataExtractionAgent)
    ontology_graph    : rdflib.Graph being built
    class_map         : table_name  -> rdflib.URIRef (owl:Class)
    property_map      : (table, col) -> rdflib.URIRef (owl:DatatypeProperty)
    ontology_turtle   : final serialised string (set by serialize_node)
    output_path       : path where the file was written
    triple_count      : number of triples in the graph
    class_count       : number of OWL classes created
    property_count    : total OWL properties created
    errors            : list of non-fatal error strings
    phase             : current pipeline phase
    """
    config:          Any
    report:          Dict
    ontology_graph:  Any
    class_map:       Dict
    property_map:    Dict
    ontology_turtle: str
    output_path:     str
    triple_count:    int
    class_count:     int
    property_count:  int
    errors:          List
    phase:           str
