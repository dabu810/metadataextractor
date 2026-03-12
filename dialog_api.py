"""
FastAPI service for the Dialog with Data Agent.

Standalone microservice — completely independent of all other agents.
Accepts a natural language query + KG graph data + DB connection config,
decomposes it into SQL, executes against the target DB, and returns insights.

NLQ caching: completed query results are cached in memory by a fingerprint of
(natural_query, db_type, db_host, db_name, db_schema, kg node labels).  A cache
hit returns instantly without re-running the LLM pipeline.

Endpoints
---------
GET  /health
POST /query                  start async dialog job → {job_id}
GET  /jobs/{job_id}          poll status + summary stats
GET  /jobs/{job_id}/results  get full results (queries + rows + insights)
GET  /list                   list all dialog jobs
GET  /cache                  list all cached NLQ entries
DELETE /cache/{cache_key}    invalidate a specific cache entry
DELETE /cache                clear the entire NLQ cache
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from dialog_agent import DialogAgent, DialogConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Dialog with Data Agent API", version="1.1.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()

# ── NLQ cache ─────────────────────────────────────────────────────────────────
# Maps cache_key → {cache_key, natural_query, db_fingerprint, kg_fingerprint,
#                   cached_at, job_id, insights, sql_queries, query_results, errors}
_nlq_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()

DIALOG_NODES = ["understand", "plan", "execute", "synthesize"]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_key(
    natural_query: str,
    db_type: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_schema: str,
    kg_nodes: List[Dict],
) -> str:
    """Stable fingerprint for a (query, db, schema, kg) combination."""
    kg_labels = ",".join(sorted(n.get("label", "") for n in kg_nodes if n.get("label")))
    raw = "|".join([
        natural_query.strip().lower(),
        db_type.lower(),
        db_host.lower(),
        str(db_port),
        db_name.lower(),
        db_schema.lower(),
        kg_labels,
    ])
    return hashlib.sha256(raw.encode()).hexdigest()


def _db_fingerprint(db_type: str, db_host: str, db_port: int, db_name: str, db_schema: str) -> str:
    return f"{db_type}://{db_host}:{db_port}/{db_name}/{db_schema}"


def _kg_fingerprint(kg_nodes: List[Dict]) -> str:
    labels = sorted(n.get("label", "") for n in kg_nodes if n.get("label"))
    return f"{len(labels)} tables: {', '.join(labels[:10])}" + ("…" if len(labels) > 10 else "")


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    natural_query: str

    # KG graph data (optional — from the KG Agent's /jobs/{id}/graph endpoint)
    kg_nodes: List[Dict[str, Any]] = []
    kg_edges: List[Dict[str, Any]] = []

    # Target database connection
    db_type:              str = "postgres"
    db_host:              str = ""
    db_port:              int = 5432
    db_name:              str = ""
    db_schema:            str = "public"
    db_user:              str = ""
    db_password:          str = ""
    db_connection_string: str = ""
    db_extra:             Dict[str, Any] = {}
    db_file_path:         Optional[str] = None   # for SQLite / CSV / Excel

    # Behaviour
    max_sql_queries: int = 10
    row_limit:       int = 500
    llm_model:       str = "claude-sonnet-4-6"

    # Cache control
    skip_cache: bool = False   # Set True to force a fresh run even if cached


# ── Background runner ──────────────────────────────────────────────────────────

def _run_dialog(
    job_id: str,
    natural_query: str,
    kg_nodes: List[Dict],
    kg_edges: List[Dict],
    cfg: DialogConfig,
    cache_key: str,
    db_fingerprint: str,
    kg_fingerprint: str,
) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"

    try:
        agent  = DialogAgent(cfg)
        result = agent.run(natural_query, kg_nodes, kg_edges)

        with _lock:
            _jobs[job_id].update({
                "status":          "done",
                "completed_nodes": list(DIALOG_NODES),
                "current_node":    "synthesize",
                "query_count":     len(result.get("sql_queries") or []),
                "result_count":    len(result.get("query_results") or []),
                "insights":        result.get("insights", ""),
                "sql_queries":     result.get("sql_queries") or [],
                "query_results":   result.get("query_results") or [],
                "errors":          result.get("errors") or [],
            })

        # Store in NLQ cache
        with _cache_lock:
            _nlq_cache[cache_key] = {
                "cache_key":      cache_key,
                "natural_query":  natural_query,
                "db_fingerprint": db_fingerprint,
                "kg_fingerprint": kg_fingerprint,
                "cached_at":      datetime.now(timezone.utc).isoformat(),
                "job_id":         job_id,
                "insights":       result.get("insights", ""),
                "sql_queries":    result.get("sql_queries") or [],
                "query_results":  result.get("query_results") or [],
                "errors":         result.get("errors") or [],
            }
            logger.info("NLQ cached: key=%s", cache_key[:12])

    except Exception as exc:
        logger.exception("Dialog job %s failed", job_id)
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", status_code=202)
def start_query(req: QueryRequest, background_tasks: BackgroundTasks):
    if not req.natural_query.strip():
        raise HTTPException(status_code=400, detail="natural_query must not be empty")

    _FILE_BASED = {"sqlite", "csv", "excel"}
    cfg = DialogConfig(
        db_type              = req.db_type,
        db_host              = req.db_host,
        db_port              = req.db_port,
        db_name              = req.db_name,
        # File-based sources have no schema; force empty so planner never qualifies names
        db_schema            = "" if req.db_type.lower() in _FILE_BASED else req.db_schema,
        db_user              = req.db_user,
        db_password          = req.db_password,
        db_connection_string = req.db_connection_string,
        db_extra             = req.db_extra,
        db_file_path         = req.db_file_path or "",
        llm_model            = req.llm_model,
        max_sql_queries      = req.max_sql_queries,
        row_limit            = req.row_limit,
    )

    ck = _cache_key(
        req.natural_query, req.db_type, req.db_host,
        req.db_port, req.db_name, req.db_schema, req.kg_nodes,
    )
    dbfp = _db_fingerprint(req.db_type, req.db_host, req.db_port, req.db_name, req.db_schema)
    kgfp = _kg_fingerprint(req.kg_nodes)

    # ── Cache hit: synthesize a completed job from cached data ─────────────────
    if not req.skip_cache:
        with _cache_lock:
            cached = _nlq_cache.get(ck)
        if cached:
            logger.info("NLQ cache HIT: key=%s", ck[:12])
            job_id = str(uuid.uuid4())
            with _lock:
                _jobs[job_id] = {
                    "id":              job_id,
                    "natural_query":   req.natural_query,
                    "db_type":         req.db_type,
                    "status":          "done",
                    "current_node":    "synthesize",
                    "completed_nodes": list(DIALOG_NODES),
                    "query_count":     len(cached.get("sql_queries") or []),
                    "result_count":    len(cached.get("query_results") or []),
                    "insights":        cached.get("insights", ""),
                    "sql_queries":     cached.get("sql_queries") or [],
                    "query_results":   cached.get("query_results") or [],
                    "errors":          cached.get("errors") or [],
                    "error":           None,
                    "cache_hit":       True,
                    "cache_key":       ck,
                }
            return {"job_id": job_id, "cache_hit": True}

    # ── Cache miss: run the full pipeline ─────────────────────────────────────
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "id":              job_id,
            "natural_query":   req.natural_query,
            "db_type":         req.db_type,
            "status":          "queued",
            "current_node":    None,
            "completed_nodes": [],
            "query_count":     0,
            "result_count":    0,
            "insights":        "",
            "sql_queries":     [],
            "query_results":   [],
            "errors":          [],
            "error":           None,
            "cache_hit":       False,
            "cache_key":       ck,
        }

    background_tasks.add_task(
        _run_dialog, job_id,
        req.natural_query, req.kg_nodes, req.kg_edges,
        cfg, ck, dbfp, kgfp,
    )
    return {"job_id": job_id, "cache_hit": False}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items()
            if k not in ("sql_queries", "query_results")}


@app.get("/jobs/{job_id}/results")
def get_results(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Results not yet ready")
    return {
        "natural_query":  job["natural_query"],
        "insights":       job["insights"],
        "sql_queries":    job["sql_queries"],
        "query_results":  job["query_results"],
        "errors":         job["errors"],
        "cache_hit":      job.get("cache_hit", False),
    }


@app.get("/list")
def list_jobs():
    with _lock:
        jobs = list(_jobs.values())
    return [
        {k: v for k, v in j.items() if k not in ("sql_queries", "query_results")}
        for j in jobs
    ]


# ── NLQ Cache endpoints ────────────────────────────────────────────────────────

@app.get("/cache")
def list_cache():
    """List all cached NLQ entries (without the full result blobs)."""
    with _cache_lock:
        entries = list(_nlq_cache.values())
    return [
        {k: v for k, v in e.items() if k not in ("sql_queries", "query_results")}
        for e in entries
    ]


@app.delete("/cache/{cache_key}", status_code=200)
def delete_cache_entry(cache_key: str):
    """Invalidate a single cached NLQ entry by its cache key."""
    with _cache_lock:
        if cache_key not in _nlq_cache:
            raise HTTPException(status_code=404, detail="Cache entry not found")
        del _nlq_cache[cache_key]
    logger.info("NLQ cache entry deleted: key=%s", cache_key[:12])
    return {"deleted": cache_key}


@app.delete("/cache", status_code=200)
def clear_cache():
    """Clear the entire NLQ cache."""
    with _cache_lock:
        count = len(_nlq_cache)
        _nlq_cache.clear()
    logger.info("NLQ cache cleared: %d entries removed", count)
    return {"deleted_count": count}
