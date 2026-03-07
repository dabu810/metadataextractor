"""
FastAPI service for the Knowledge Graph Agent.

Standalone microservice — completely independent of metadata_agent and ontology_agent.
Accepts a raw ontology string (Turtle/RDF/N3), converts it to Cypher or Gremlin,
optionally executes on a live Neo4j / Gremlin server, and returns graph data
for visualisation in the UI.

Endpoints
---------
GET  /health
POST /generate                    start async KG creation → {job_id}
GET  /jobs/{job_id}               poll status + stats
GET  /jobs/{job_id}/graph         get {nodes, edges} for UI visualisation
GET  /jobs/{job_id}/queries       get the generated Cypher/Gremlin statements
GET  /list                        list all KG jobs
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

from knowledge_graph_agent import KGAgent, KGConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Knowledge Graph Agent API", version="1.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()

KG_NODES = ["parse", "translate", "execute"]


# ── Request models ─────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    ontology_text:   str
    ontology_format: str = "turtle"     # "turtle" | "xml" | "n3"
    graph_type:      str = "neo4j"      # "neo4j" | "gremlin"

    # Neo4j connection (optional — omit to run in preview/translate-only mode)
    neo4j_uri:      str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # Gremlin connection (optional)
    gremlin_url:              str = ""
    gremlin_traversal_source: str = "g"

    # Behaviour
    clear_existing: bool = False


# ── Background runner ──────────────────────────────────────────────────────────
def _run_kg(job_id: str, ontology_text: str, ontology_format: str, cfg: KGConfig) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"

    completed: List[str] = []
    try:
        agent = KGAgent(cfg)

        for node_name, state_update in agent.stream_run(ontology_text, ontology_format):
            clean = node_name.strip("_").replace("error_end", "error")
            with _lock:
                if "error" in node_name:
                    _jobs[job_id]["status"] = "error"
                    _jobs[job_id]["error"]  = f"Pipeline error at node: {node_name}"
                    return
                real = clean if clean in KG_NODES else None
                if real and real not in completed:
                    completed.append(real)
                _jobs[job_id]["completed_nodes"] = list(completed)
                _jobs[job_id]["current_node"]    = clean

        # Run synchronously to get the full result
        result = agent.run(ontology_text, ontology_format)

        with _lock:
            _jobs[job_id].update({
                "status":           "done",
                "completed_nodes":  list(KG_NODES),
                "current_node":     "execute",
                "node_count":       result["node_count"],
                "edge_count":       result["edge_count"],
                "executed_count":   result["executed_count"],
                "graph_data":       result["graph_data"],
                "queries":          result["queries"],
                "execution_results": result["execution_results"],
                "errors":           result["errors"],
            })

    except Exception as exc:
        logger.exception("KG job %s failed", job_id)
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", status_code=202)
def generate_kg(req: GenerateRequest, background_tasks: BackgroundTasks):
    if not req.ontology_text.strip():
        raise HTTPException(status_code=400, detail="ontology_text must not be empty")
    if req.graph_type not in ("neo4j", "gremlin"):
        raise HTTPException(status_code=400, detail="graph_type must be 'neo4j' or 'gremlin'")

    cfg = KGConfig(
        graph_type               = req.graph_type,
        neo4j_uri                = req.neo4j_uri,
        neo4j_username           = req.neo4j_username,
        neo4j_password           = req.neo4j_password,
        neo4j_database           = req.neo4j_database,
        gremlin_url              = req.gremlin_url,
        gremlin_traversal_source = req.gremlin_traversal_source,
        clear_existing           = req.clear_existing,
    )

    job_id = str(uuid.uuid4())

    with _lock:
        _jobs[job_id] = {
            "id":                job_id,
            "graph_type":        req.graph_type,
            "ontology_format":   req.ontology_format,
            "status":            "queued",
            "current_node":      None,
            "completed_nodes":   [],
            "node_count":        0,
            "edge_count":        0,
            "executed_count":    0,
            "graph_data":        {"nodes": [], "edges": []},
            "queries":           [],
            "execution_results": [],
            "errors":            [],
            "error":             None,
        }

    background_tasks.add_task(
        _run_kg, job_id, req.ontology_text, req.ontology_format, cfg
    )
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Return everything except the large graph_data and queries blobs
    return {k: v for k, v in job.items() if k not in ("graph_data", "queries", "execution_results")}


@app.get("/jobs/{job_id}/graph")
def get_graph(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("done",):
        raise HTTPException(status_code=202, detail="Graph not yet ready")
    return job["graph_data"]


@app.get("/jobs/{job_id}/queries")
def get_queries(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("done",):
        raise HTTPException(status_code=202, detail="Queries not yet ready")
    return {
        "graph_type": job["graph_type"],
        "queries":    job["queries"],
        "count":      len(job["queries"]),
    }


@app.get("/list")
def list_jobs():
    with _lock:
        jobs = list(_jobs.values())
    return [
        {k: v for k, v in j.items() if k not in ("graph_data", "queries", "execution_results")}
        for j in jobs
    ]
