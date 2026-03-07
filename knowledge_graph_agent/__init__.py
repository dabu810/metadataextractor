"""
Knowledge Graph Agent — converts OWL/RDF ontologies to Cypher (Neo4j) or
Gremlin (Apache TinkerPop) and optionally executes them on a live graph DB.

Completely decoupled from metadata_agent and ontology_agent.
The only input is a raw ontology string (Turtle / RDF-XML / N3).
"""
from .agent import KGAgent
from .config import KGConfig

__all__ = ["KGAgent", "KGConfig"]
