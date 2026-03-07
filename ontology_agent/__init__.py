"""
Ontology Agent — converts metadata extraction reports to OWL/RDF ontologies.

Completely independent of the metadata_agent package.
Input : the JSON report dict produced by MetadataExtractionAgent.run()
Output: an OWL/RDF file in Turtle (default), RDF/XML, or N3 format.
"""
from .agent import OntologyAgent
from .config import OntologyConfig

__all__ = ["OntologyAgent", "OntologyConfig"]
