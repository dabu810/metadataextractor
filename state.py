"""
LangGraph agent state definition for the metadata extraction agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Sub-structures stored in state
# ---------------------------------------------------------------------------

@dataclass
class ColumnMeta:
    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool = False
    is_foreign_key: bool = False
    fk_references: Optional[str] = None       # "table.column"
    unique_count: Optional[int] = None
    null_count: Optional[int] = None
    row_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_value: Optional[float] = None
    stddev_value: Optional[float] = None
    top_values: Optional[List[Any]] = None    # up to 10 most frequent


@dataclass
class TableMeta:
    schema_name: str
    table_name: str
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    columns: List[ColumnMeta] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    table_comment: Optional[str] = None
    create_time: Optional[str] = None
    last_modified: Optional[str] = None
    partitioned_by: Optional[List[str]] = None


@dataclass
class FunctionalDependency:
    """X -> Y holds in table 'table_name' with given confidence."""
    table_name: str
    determinant: List[str]     # X (left-hand side)
    dependent: List[str]       # Y (right-hand side)
    confidence: float          # fraction of tuples satisfying the FD
    num_violations: int = 0


@dataclass
class InclusionDependency:
    """left_table[left_cols] ⊆ right_table[right_cols]"""
    left_table: str
    left_columns: List[str]
    right_table: str
    right_columns: List[str]
    coverage: float            # fraction of left values found in right
    is_foreign_key_candidate: bool = False


@dataclass
class CardinalityRelationship:
    """Cardinality between two tables via shared/join columns."""
    left_table: str
    right_table: str
    join_columns: List[str]    # columns used to determine cardinality
    relationship_type: str     # "1:1" | "1:N" | "N:1" | "M:N"
    left_unique: int = 0
    right_unique: int = 0


# ---------------------------------------------------------------------------
# Agent state (passed between LangGraph nodes)
# ---------------------------------------------------------------------------

class AgentState(dict):
    """
    TypedDict-like state for LangGraph.  Keys:

    phase          : current phase name (string)
    db_config      : DBConfig instance
    agent_config   : AgentConfig instance
    connector      : live DB connector (injected at runtime)
    all_tables     : list of (schema, table) tuples discovered
    tables_done    : set of table names whose metadata is extracted
    table_metadata : dict[table_name -> TableMeta]
    func_deps      : list[FunctionalDependency]
    incl_deps      : list[InclusionDependency]
    cardinalities  : list[CardinalityRelationship]
    messages       : LangChain message history (for LLM node)
    errors         : list of error strings
    final_report   : dict with the full aggregated report
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Config / runtime objects — must be registered here so LangGraph
        # always recognises them as valid state keys even when it reconstructs
        # state from a merged dict rather than the original constructor kwargs.
        self.setdefault("agent_config", None)
        self.setdefault("db_config", None)
        self.setdefault("connector", None)
        # Pipeline data
        self.setdefault("phase", "init")
        self.setdefault("all_tables", [])
        self.setdefault("tables_done", set())
        self.setdefault("table_metadata", {})
        self.setdefault("func_deps", [])
        self.setdefault("incl_deps", [])
        self.setdefault("cardinalities", [])
        self.setdefault("messages", [])
        self.setdefault("errors", [])
        self.setdefault("final_report", {})
