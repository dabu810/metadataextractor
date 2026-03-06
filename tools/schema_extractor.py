"""
LangChain tool: extract schema (columns, PKs, FKs, indexes) for a table.
"""
from __future__ import annotations

import json
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..connectors.base import BaseConnector
from ..state import ColumnMeta, TableMeta


class SchemaExtractorInput(BaseModel):
    schema_name: str = Field(description="Database schema / dataset name")
    table_name: str = Field(description="Table name to extract schema for")


class SchemaExtractorTool(BaseTool):
    """Extract full schema metadata for a single table."""

    name: str = "schema_extractor"
    description: str = (
        "Extract schema information for a database table including columns "
        "(name, type, nullability), primary keys, foreign keys, and indexes. "
        "Returns a JSON dict with all schema details."
    )
    args_schema: Type[BaseModel] = SchemaExtractorInput
    connector: BaseConnector = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, schema_name: str, table_name: str) -> str:
        try:
            raw_cols = self.connector.get_columns(schema_name, table_name)
            pks = self.connector.get_primary_keys(schema_name, table_name)
            fks = self.connector.get_foreign_keys(schema_name, table_name)
            indexes = self.connector.get_indexes(schema_name, table_name)
            timestamps = self.connector.get_table_timestamps(schema_name, table_name)
            comment = self.connector.get_table_comment(schema_name, table_name)
            partitions = self.connector.get_partition_columns(schema_name, table_name)

            pk_set = set(pks)
            fk_map = {f["column"]: f for f in fks}

            columns = []
            for col in raw_cols:
                col_name = col.get("name") or col.get("column_name", "")
                columns.append({
                    "name": col_name,
                    "data_type": col.get("data_type", "unknown"),
                    "nullable": col.get("nullable") not in ("NO", "N", False),
                    "is_primary_key": col_name in pk_set,
                    "is_foreign_key": col_name in fk_map,
                    "fk_references": (
                        f"{fk_map[col_name]['referenced_table']}.{fk_map[col_name]['referenced_column']}"
                        if col_name in fk_map else None
                    ),
                    "column_default": col.get("column_default"),
                    "char_max_length": col.get("character_maximum_length"),
                    "numeric_precision": col.get("numeric_precision"),
                    "numeric_scale": col.get("numeric_scale"),
                })

            result = {
                "schema_name": schema_name,
                "table_name": table_name,
                "columns": columns,
                "primary_keys": pks,
                "foreign_keys": fks,
                "indexes": indexes,
                "create_time": timestamps.get("create_time"),
                "last_modified": timestamps.get("last_modified"),
                "table_comment": comment,
                "partitioned_by": partitions,
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "table": table_name})

    async def _arun(self, schema_name: str, table_name: str) -> str:
        return self._run(schema_name, table_name)
