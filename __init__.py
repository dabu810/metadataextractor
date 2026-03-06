"""
Metadata Extraction Agent
=========================
A LangGraph + LangChain agent that scans RDBMS schemas (Oracle, Teradata,
Delta Lake, Redshift, SQL Server, PostgreSQL, BigQuery) and extracts:
  - Table-level metadata (row count, size, timestamps, partitions)
  - Column-level statistics (nulls, cardinality, min/max/avg/stddev, top values)
  - Functional Dependencies (X → Y)
  - Inclusion Dependencies (R[A] ⊆ S[B])
  - Cardinality relationships (1:1, 1:N, M:N)
"""
from .agent import MetadataExtractionAgent, build_graph
from .config import AgentConfig, DBConfig, DBType
from .state import (
    AgentState,
    CardinalityRelationship,
    ColumnMeta,
    FunctionalDependency,
    InclusionDependency,
    TableMeta,
)

__all__ = [
    "MetadataExtractionAgent",
    "build_graph",
    "AgentConfig",
    "DBConfig",
    "DBType",
    "AgentState",
    "TableMeta",
    "ColumnMeta",
    "FunctionalDependency",
    "InclusionDependency",
    "CardinalityRelationship",
]
