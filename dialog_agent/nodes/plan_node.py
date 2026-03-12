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
4. Column names MUST match EXACTLY the column names listed in the schema context.
   NEVER invent or guess a column name that is not explicitly listed.
   If a column you need does not appear in the schema, omit that filter entirely.
5. Apply LIMIT {row_limit} to every query.
6. Prefer simple queries; only join when necessary.
7. If the question cannot be answered from the available schema, return [].
8. Maximum {max_queries} queries total.
9. Do NOT use table aliases that shadow schema-qualified names — always write the
   full qualified reference in FROM/JOIN clauses.
10. String/text filters: ALWAYS use case-insensitive matching.
    - If [sample values] are shown for a column, use the exact spelling from the samples.
    - If sample values are NOT shown, use: LOWER(column_name) LIKE LOWER('%search_term%')
    - Never rely on an exact case-sensitive equality match for text unless you copied
      the value directly from a [sample values] list.
11. Date/period filters: if filtering by year/month, check column names in the schema
    carefully — use the correct column (e.g. Year, Month, Period) and match the sample
    value format (e.g. integer 2026 vs string '2026').
"""

_USER_PROMPT = """\
SCHEMA CONTEXT:
{schema_context}

TARGET DATABASE TYPE: {db_type}
{schema_line}

NATURAL LANGUAGE QUESTION:
{natural_query}

CRITICAL REMINDERS:
- Use ONLY column names that appear in the DETAILED SCHEMA above. Do NOT invent column names.
- Use ONLY table names from the AVAILABLE TABLES list above.
- For any text/string filter, use case-insensitive matching (LOWER() LIKE or exact sample value).
- If [sample values] are shown for a column, pick the matching value verbatim from that list.

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
        # Skip if already present in the SQL (case-insensitive)
        if re.search(re.escape(qualified), sql, re.IGNORECASE):
            continue
        # Replace bare table name not preceded by a dot
        # Negative lookbehind for `.` or alphanumeric ensures we don't touch substrings
        pattern = r'(?<![.\w])\b' + re.escape(table) + r'\b(?!\s*\.)'
        if re.search(pattern, sql, re.IGNORECASE):
            sql = re.sub(pattern, qualified, sql, flags=re.IGNORECASE)

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

    # File-based sources (SQLite / CSV / Excel) load into in-memory SQLite with no schema
    # prefix.  Force empty schema so the LLM and the safety-net qualify step both use
    # bare table names.
    _FILE_BASED_TYPES = {"sqlite", "csv", "excel"}
    db_schema = "" if config.db_type.lower() in _FILE_BASED_TYPES else (config.db_schema or "")

    # Extract table labels from KG nodes for the SQL post-processor.
    # Exclude XSD data types and SQL/SPARQL keywords that must never be
    # treated as table names — these can appear as KG node labels when an
    # ontology generator incorrectly marks xsd:string etc. as owl:Class.
    _EXCLUDED_LABELS = {
        "string", "integer", "int", "float", "double", "decimal", "boolean",
        "date", "datetime", "time", "duration", "anyuri", "literal",
        "long", "short", "byte", "binary", "hexbinary", "base64binary",
        "nonnegativeinteger", "positiveinteger", "negativinteger",
        "unsignedlong", "unsignedint", "unsignedshort", "unsignedbyte",
        # SQL reserved words that must not be table-qualified
        "select", "from", "where", "join", "on", "group", "order", "by",
        "having", "limit", "offset", "as", "and", "or", "not", "null",
        "true", "false", "case", "when", "then", "else", "end", "in",
        "between", "like", "is", "distinct", "all", "any", "exists",
        "union", "intersect", "except", "with", "values", "set",
    }
    kg_nodes     = state.get("kg_nodes") or []
    table_labels = [
        n.get("label", "") for n in kg_nodes
        if n.get("label") and n.get("label", "").lower() not in _EXCLUDED_LABELS
    ]

    system = _SYSTEM_PROMPT.format(
        row_limit=config.row_limit,
        max_queries=config.max_sql_queries,
    )
    schema_line = (
        f"TARGET SCHEMA: {db_schema}"
        if db_schema
        else "TARGET SCHEMA: (none — use bare table names WITHOUT any schema prefix)"
    )
    user = _USER_PROMPT.format(
        schema_context=schema_context,
        db_type=config.db_type,
        schema_line=schema_line,
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
