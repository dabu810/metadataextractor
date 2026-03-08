"""
LangGraph state for the Dialog with Data Agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class SQLQuery(TypedDict):
    query_id: str
    description: str
    sql: str
    table_refs: List[str]


class QueryResult(TypedDict):
    query_id: str
    description: str
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    error: Optional[str]


class DialogState(TypedDict, total=False):
    # Inputs
    config: Any                        # DialogConfig
    natural_query: str                 # the user's NQL string
    schema_context: str                # graph/ontology summary fed to LLM
    kg_nodes: List[Dict[str, Any]]     # knowledge graph nodes (from KG agent)
    kg_edges: List[Dict[str, Any]]     # knowledge graph edges

    # Intermediate
    sql_queries: List[SQLQuery]        # planner output
    query_results: List[QueryResult]   # executor output

    # Output
    insights: str                      # LLM-derived narrative

    errors: List[str]
    phase: str                         # understand | plan | execute | synthesize | done | error
