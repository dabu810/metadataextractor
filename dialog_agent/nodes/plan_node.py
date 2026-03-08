"""
plan_node — Use LLM to decompose a natural language query into SQL statements.

The LLM receives:
  - The user's natural language query
  - The schema context (table names, columns, relationships)
  - Target DB type (to pick the right SQL dialect)
  - Row limit (to avoid pulling millions of rows)

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
database and a schema context (table names, columns, relationships).  Your job is
to decompose the question into one or more SQL SELECT queries that, when executed
and combined, will answer the question completely.

Rules:
1. Return ONLY a JSON array — no prose, no markdown fences.
2. Each element must have exactly these fields:
   - "query_id"   : a short unique identifier (e.g. "q1", "q2")
   - "description": one sentence explaining what this query retrieves
   - "sql"        : a complete, runnable SQL SELECT statement
   - "table_refs" : array of table names referenced in the query
3. Use only tables and columns that exist in the schema context.
4. Apply LIMIT {row_limit} to every query.
5. Prefer simple queries; only join when necessary.
6. If the question cannot be answered from the schema, return an empty array [].
7. Maximum {max_queries} queries total.
"""

_USER_PROMPT = """\
SCHEMA CONTEXT:
{schema_context}

TARGET DATABASE TYPE: {db_type}

NATURAL LANGUAGE QUESTION:
{natural_query}

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
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Find first [...] block
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start == -1 or end == -1:
        return []
    return json.loads(cleaned[start:end + 1])


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

    system = _SYSTEM_PROMPT.format(
        row_limit=config.row_limit,
        max_queries=config.max_sql_queries,
    )
    user = _USER_PROMPT.format(
        schema_context=schema_context,
        db_type=config.db_type,
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

    # Validate and cap
    sql_queries: List[SQLQuery] = []
    for item in plan[: config.max_sql_queries]:
        if not item.get("sql"):
            continue
        sql_queries.append(
            SQLQuery(
                query_id    = item.get("query_id", f"q{len(sql_queries)+1}"),
                description = item.get("description", ""),
                sql         = item["sql"].strip(),
                table_refs  = item.get("table_refs", []),
            )
        )

    logger.info("plan_node: %d SQL queries planned", len(sql_queries))
    state["sql_queries"] = sql_queries
    state["phase"] = "plan"
    return state
