"""
FastAPI service for the Conformity Agent.

Standalone microservice — independent of all other agents.
Accepts KG snapshots, runs analyse + recommend, and supports stitching.

Endpoints
---------
GET  /health
POST /analyse                   start async analyse → recommend job → {job_id}
GET  /jobs/{job_id}             poll status
GET  /jobs/{job_id}/results     get conformities + recommendations
POST /stitch                    run stitch pipeline → {stitch_id}
GET  /stitch/{stitch_id}        poll stitch status
GET  /stitch/{stitch_id}/graph  get super-graph {nodes, edges}
GET  /super-graphs              list saved super-graphs
POST /super-graphs              save a super-graph with a user-supplied name
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

from conformity_agent import ConformityAgent, ConformityConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Conformity Agent API", version="1.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory stores ──────────────────────────────────────────────────────────
_jobs:        Dict[str, Dict] = {}   # analyse jobs
_stitches:    Dict[str, Dict] = {}   # stitch jobs
_super_graphs: Dict[str, Dict] = {}  # saved super-graphs (name → data)
_lock = threading.Lock()


# ── Request models ─────────────────────────────────────────────────────────────

class NodeIn(BaseModel):
    id:    str
    label: str = ""
    title: str = ""
    color: str = "#63b3ed"
    size:  int = 20


class EdgeIn(BaseModel):
    from_: str
    to:    str
    label: str = ""
    title: str = ""

    class Config:
        # allow "from" as field alias since it's a Python keyword
        populate_by_name = True
        fields = {"from_": "from"}


class KGSnapshotIn(BaseModel):
    kg_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class AnalyseRequest(BaseModel):
    kg_snapshots:           List[KGSnapshotIn]
    fuzzy_threshold:        float = 80.0
    jaccard_threshold:      float = 0.30
    max_node_pairs:         int   = 10_000
    llm_model:              str   = "claude-sonnet-4-6"
    max_conformities_in_prompt: int = 60


class StitchRequest(BaseModel):
    job_id:           str          # analyse job to take snapshots + conformities from
    approved_indices: List[int]


class SaveSuperGraphRequest(BaseModel):
    stitch_id: str
    name:      str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_job(extra: Dict | None = None) -> Dict:
    job: Dict = {
        "status":    "running",
        "phase":     "init",
        "progress":  0,
        "errors":    [],
    }
    if extra:
        job.update(extra)
    return job


def _run_analyse(job_id: str, req: AnalyseRequest) -> None:
    cfg = ConformityConfig(
        fuzzy_threshold=req.fuzzy_threshold,
        jaccard_threshold=req.jaccard_threshold,
        max_node_pairs=req.max_node_pairs,
        llm_model=req.llm_model,
        max_conformities_in_prompt=req.max_conformities_in_prompt,
    )
    agent = ConformityAgent(cfg)
    snapshots = [s.model_dump(by_alias=False) for s in req.kg_snapshots]
    # Fix "from_" → "from" in edges for each snapshot
    for snap in snapshots:
        for edge in snap.get("edges", []):
            if "from_" in edge:
                edge["from"] = edge.pop("from_")

    try:
        result = agent.analyse(snapshots)
        with _lock:
            _jobs[job_id].update({
                "status":          "completed",
                "phase":           result.get("phase", "recommended"),
                "progress":        100,
                "errors":          result.get("errors") or [],
                "conformities":    result.get("conformities") or [],
                "exact_count":     result.get("exact_count", 0),
                "fuzzy_count":     result.get("fuzzy_count", 0),
                "jaccard_count":   result.get("jaccard_count", 0),
                "recommendations": result.get("recommendations", ""),
                # Keep snapshots for stitch step
                "kg_snapshots":    snapshots,
            })
    except Exception as exc:
        logger.exception("analyse job %s failed", job_id)
        with _lock:
            _jobs[job_id].update({
                "status": "error",
                "errors": [str(exc)],
            })


def _run_stitch(stitch_id: str, job_id: str, approved_indices: List[int]) -> None:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        with _lock:
            _stitches[stitch_id]["status"] = "error"
            _stitches[stitch_id]["errors"] = [f"Analyse job {job_id} not found"]
        return

    cfg    = ConformityConfig()
    agent  = ConformityAgent(cfg)
    snapshots    = job.get("kg_snapshots", [])
    conformities = job.get("conformities", [])

    try:
        result = agent.stitch(snapshots, conformities, approved_indices)
        with _lock:
            _stitches[stitch_id].update({
                "status":      "completed",
                "phase":       result.get("phase", "stitched"),
                "progress":    100,
                "errors":      result.get("errors") or [],
                "super_graph": result.get("super_graph", {"nodes": [], "edges": []}),
                "stitch_log":  result.get("stitch_log", []),
            })
    except Exception as exc:
        logger.exception("stitch job %s failed", stitch_id)
        with _lock:
            _stitches[stitch_id].update({
                "status": "error",
                "errors": [str(exc)],
            })


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "conformity-agent"}


@app.post("/analyse")
def start_analyse(req: AnalyseRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = _new_job({
            "kg_count": len(req.kg_snapshots),
        })
    bg.add_task(_run_analyse, job_id, req)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return {
        "job_id":       job_id,
        "status":       job["status"],
        "phase":        job.get("phase", ""),
        "progress":     job.get("progress", 0),
        "kg_count":     job.get("kg_count", 0),
        "exact_count":  job.get("exact_count", 0),
        "fuzzy_count":  job.get("fuzzy_count", 0),
        "jaccard_count":job.get("jaccard_count", 0),
        "errors":       job.get("errors", []),
    }


@app.get("/jobs/{job_id}/results")
def get_results(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job {job_id} is not completed (status={job['status']})")
    return {
        "job_id":          job_id,
        "conformities":    job.get("conformities", []),
        "exact_count":     job.get("exact_count", 0),
        "fuzzy_count":     job.get("fuzzy_count", 0),
        "jaccard_count":   job.get("jaccard_count", 0),
        "recommendations": job.get("recommendations", ""),
    }


@app.post("/stitch")
def start_stitch(req: StitchRequest, bg: BackgroundTasks):
    with _lock:
        if req.job_id not in _jobs:
            raise HTTPException(404, f"Analyse job {req.job_id} not found")
        if _jobs[req.job_id]["status"] != "completed":
            raise HTTPException(400, "Analyse job must be completed before stitching")

    stitch_id = str(uuid.uuid4())
    with _lock:
        _stitches[stitch_id] = _new_job({
            "job_id":           req.job_id,
            "approved_count":   len(req.approved_indices),
        })
    bg.add_task(_run_stitch, stitch_id, req.job_id, req.approved_indices)
    return {"stitch_id": stitch_id}


@app.get("/stitch/{stitch_id}")
def get_stitch(stitch_id: str):
    with _lock:
        stitch = _stitches.get(stitch_id)
    if not stitch:
        raise HTTPException(404, f"Stitch job {stitch_id} not found")
    return {
        "stitch_id":     stitch_id,
        "status":        stitch["status"],
        "phase":         stitch.get("phase", ""),
        "progress":      stitch.get("progress", 0),
        "approved_count":stitch.get("approved_count", 0),
        "errors":        stitch.get("errors", []),
        "stitch_log":    stitch.get("stitch_log", []),
    }


@app.get("/stitch/{stitch_id}/graph")
def get_stitch_graph(stitch_id: str):
    with _lock:
        stitch = _stitches.get(stitch_id)
    if not stitch:
        raise HTTPException(404, f"Stitch job {stitch_id} not found")
    if stitch["status"] != "completed":
        raise HTTPException(400, f"Stitch job {stitch_id} is not completed")
    return stitch.get("super_graph", {"nodes": [], "edges": []})


@app.get("/super-graphs")
def list_super_graphs():
    with _lock:
        return [
            {"name": name, "node_count": len(sg["nodes"]), "edge_count": len(sg["edges"])}
            for name, sg in _super_graphs.items()
        ]


@app.post("/super-graphs")
def save_super_graph(req: SaveSuperGraphRequest):
    with _lock:
        stitch = _stitches.get(req.stitch_id)
    if not stitch:
        raise HTTPException(404, f"Stitch job {req.stitch_id} not found")
    if stitch["status"] != "completed":
        raise HTTPException(400, "Stitch job must be completed to save")
    sg = stitch.get("super_graph", {"nodes": [], "edges": []})
    with _lock:
        _super_graphs[req.name] = sg
    return {
        "name":       req.name,
        "node_count": len(sg["nodes"]),
        "edge_count": len(sg["edges"]),
    }


@app.get("/super-graphs/{name}")
def get_super_graph(name: str):
    with _lock:
        sg = _super_graphs.get(name)
    if sg is None:
        raise HTTPException(404, f"Super-graph '{name}' not found")
    return sg


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("conformity_api:app", host="0.0.0.0", port=8004, reload=False)
