"""
Dialog with Data Agent — natural-language-to-SQL via knowledge graph traversal.

Standalone package; zero imports from metadata_agent, ontology_agent, or
knowledge_graph_agent.
"""
from .agent import DialogAgent
from .config import DialogConfig

__all__ = ["DialogAgent", "DialogConfig"]
