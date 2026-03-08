"""
FastAPI service for the Dialog with Data Agent.

Standalone microservice — completely independent of all other agents.
Accepts a natural language query + KG graph data + DB connection config,
decomposes it into SQL, executes against the target DB, and returns insights.

Endpoints
---------
GET  /health
POST /query                  start async dialog job → {job_id}
GET  /jobs/{job_id}          poll status + summary stats
GET  /jobs/{job_id}/results  get full results (queries + rows + insights)
GET  /list                   list all dialog jobs
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
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
app = FastAPI(title="Dialog with Data Agent API", version="1.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()

DIALOG_NODES = ["understand", "plan", "execute", "synthesize"]


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

    # Behaviour
    max_sql_queries: int = 10
    row_limit:       int = 500
    llm_model:       str = "claude-sonnet-4-6"


# ── Background runner ──────────────────────────────────────────────────────────
def _run_dialog(
    job_id: str,
    natural_query: str,
    kg_nodes: List[Dict],
    kg_edges: List[Dict],
    cfg: DialogConfig,
) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"

    completed: List[str] = []
    try:
        agent = DialogAgent(cfg)

        for node_name, state_update in agent.stream_run(natural_query, kg_nodes, kg_edges):
            clean = node_name.strip("_")
            with _lock:
                if clean in DIALOG_NODES and clean not in completed:
                    completed.append(clean)
                _jobs[job_id]["completed_nodes"] = list(completed)
                _jobs[job_id]["current_node"]    = clean

        # Run synchronously to get the full result
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

    cfg = DialogConfig(
        db_type              = req.db_type,
        db_host              = req.db_host,
        db_port              = req.db_port,
        db_name              = req.db_name,
        db_schema            = req.db_schema,
        db_user              = req.db_user,
        db_password          = req.db_password,
        db_connection_string = req.db_connection_string,
        db_extra             = req.db_extra,
        llm_model            = req.llm_model,
        max_sql_queries      = req.max_sql_queries,
        row_limit            = req.row_limit,
    )

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
        }

    background_tasks.add_task(
        _run_dialog, job_id,
        req.natural_query, req.kg_nodes, req.kg_edges, cfg,
    )
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Exclude large blobs from the status poll endpoint
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
    }


@app.get("/list")
def list_jobs():
    with _lock:
        jobs = list(_jobs.values())
    return [
        {k: v for k, v in j.items() if k not in ("sql_queries", "query_results")}
        for j in jobs
    ]
