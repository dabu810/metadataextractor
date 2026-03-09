"""
plan_node — Use LLM to decompose a natural language query into SQL statements.

The LLM receives:
  - The user's natural language query
  - The schema context (qualified table names, columns, relationships)
  - Target DB type (to pick the right SQL dialect)
  - The schema name and row limit

It must return a JSON array of query objects:
    [{"query_id": "q1", "description": "...", "sql": "...", "table_refs": [...]}, ...]
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from ..state import DialogState, SQLQuery

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert SQL analyst.  You receive a natural-language question about a
database and a schema context (qualified table names, columns, relationships).
Your job is to decompose the question into one or more SQL SELECT queries that,
when executed and combined, will answer the question completely.

Rules:
1. Return ONLY a JSON array — no prose, no markdown fences.
2. Each element must have exactly these fields:
   - "query_id"   : a short unique identifier (e.g. "q1", "q2")
   - "description": one sentence explaining what this query retrieves
   - "sql"        : a complete, runnable SQL SELECT statement
   - "table_refs" : array of FULLY QUALIFIED table names referenced (e.g. ["public.orders"])
3. Table names MUST be written exactly as shown in the AVAILABLE TABLES list in the
   schema context — including the schema prefix (e.g. `public.orders`, not just `orders`).
   If no schema is listed, use the bare table name.
4. Column names MUST match exactly the column names in the schema context.
5. Apply LIMIT {row_limit} to every query.
6. Prefer simple queries; only join when necessary.
7. If the question cannot be answered from the available schema, return [].
8. Maximum {max_queries} queries total.
9. Do NOT use table aliases that shadow schema-qualified names — always write the
   full qualified reference in FROM/JOIN clauses.
"""

_USER_PROMPT = """\
SCHEMA CONTEXT:
{schema_context}

TARGET DATABASE TYPE: {db_type}
TARGET SCHEMA: {db_schema}

NATURAL LANGUAGE QUESTION:
{natural_query}

Remember: use the exact qualified table names from the AVAILABLE TABLES list above.

Return the JSON array of SQL queries now.
"""


def _call_llm(
    system: str,
    user: str,
    model: str,
    temperature: float,
) -> str:
    """Call Anthropic Claude and return the raw text response."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text if msg.content else ""


def _extract_json(text: str) -> List[Dict[str, Any]]:
    """Extract JSON array from LLM response (handles markdown fences)."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start == -1 or end == -1:
        return []
    return json.loads(cleaned[start:end + 1])


def _qualify_sql(sql: str, db_schema: str, known_tables: List[str]) -> str:
    """
    Safety net: if the LLM wrote `FROM orders` but the schema is `public`,
    rewrite bare table references to schema-qualified form `public.orders`.

    Only touches identifiers that exactly match known table labels and are not
    already preceded by a dot (i.e., not already schema-qualified).
    """
    if not db_schema or not known_tables:
        return sql

    # Sort longest first to avoid partial replacements (e.g. "order" before "orders")
    for table in sorted(known_tables, key=len, reverse=True):
        qualified = f"{db_schema}.{table}"
        # Skip if already present in the SQL
        if re.search(r'(?i)' + re.escape(qualified), sql):
            continue
        # Replace bare table name not preceded by a dot
        # Negative lookbehind for `.` or alphanumeric ensures we don't touch substrings
        pattern = r'(?<![.\w])(?i)\b' + re.escape(table) + r'\b(?!\s*\.)'
        if re.search(pattern, sql):
            sql = re.sub(pattern, qualified, sql)

    return sql


def plan_node(state: DialogState) -> DialogState:
    """Decompose the NQL into SQL queries via LLM."""
    logger.info("=== plan_node ===")

    config         = state["config"]
    natural_query  = state.get("natural_query", "").strip()
    schema_context = state.get("schema_context", "(no schema)")

    if not natural_query:
        state["errors"].append("plan_node: natural_query is empty")
        state["sql_queries"] = []
        state["phase"] = "plan"
        return state

    db_schema = config.db_schema or ""

    # Extract table labels from KG nodes for the SQL post-processor
    kg_nodes     = state.get("kg_nodes") or []
    table_labels = [n.get("label", "") for n in kg_nodes if n.get("label")]

    system = _SYSTEM_PROMPT.format(
        row_limit=config.row_limit,
        max_queries=config.max_sql_queries,
    )
    user = _USER_PROMPT.format(
        schema_context=schema_context,
        db_type=config.db_type,
        db_schema=db_schema or "(default)",
        natural_query=natural_query,
    )

    try:
        raw = _call_llm(system, user, config.llm_model, config.llm_temperature)
        logger.debug("LLM plan response (first 500 chars): %s", raw[:500])
        plan: List[Dict] = _extract_json(raw)
    except Exception as exc:
        logger.exception("plan_node LLM call failed")
        state["errors"].append(f"plan_node: LLM error — {exc}")
        state["sql_queries"] = []
        state["phase"] = "plan"
        return state

    # Validate, qualify, and cap
    sql_queries: List[SQLQuery] = []
    for item in plan[: config.max_sql_queries]:
        if not item.get("sql"):
            continue

        sql = item["sql"].strip()
        # Safety net: ensure table names are schema-qualified even if LLM forgot
        sql = _qualify_sql(sql, db_schema, table_labels)

        sql_queries.append(
            SQLQuery(
                query_id    = item.get("query_id", f"q{len(sql_queries)+1}"),
                description = item.get("description", ""),
                sql         = sql,
                table_refs  = item.get("table_refs", []),
            )
        )

    logger.info("plan_node: %d SQL queries planned", len(sql_queries))
    state["sql_queries"] = sql_queries
    state["phase"] = "plan"
    return state
