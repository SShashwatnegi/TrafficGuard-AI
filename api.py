"""FastAPI REST API for TrafficGuard AI."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.pipeline.processor import TrafficGuardPipeline

app = FastAPI(
    title="TrafficGuard AI",
    description="Multi-Agent traffic enforcement API",
    version="2.0.0",
)

pipeline: TrafficGuardPipeline | None = None


class ProcessResponse(BaseModel):
    image_path: str
    timestamp: str
    detections: list[dict[str, Any]]
    violations: list[dict[str, Any]]
    candidate_violations: list[dict[str, Any]]
    plates: list[dict[str, Any]]
    evidence_path: str | None
    evidence_package: dict[str, Any]
    processing_time_ms: float
    agent_trace: list[dict[str, Any]]
    scene_graph: dict[str, Any]
    insights: dict[str, Any]


class CommandRequest(BaseModel):
    query: str
    days: int = 30


@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = TrafficGuardPipeline()


@app.get("/health")
def health() -> dict[str, Any]:
    assert pipeline is not None
    return {"status": "ok", "service": "TrafficGuard AI", "agents": pipeline.list_agents()}


@app.post("/api/v1/process", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...)) -> dict[str, Any]:
    assert pipeline is not None
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    result = pipeline.process_image(tmp_path)
    return result.to_dict()


@app.post("/api/v1/command")
def command_query(body: CommandRequest) -> dict[str, Any]:
    assert pipeline is not None
    return pipeline.query(body.query, days=body.days)


@app.get("/api/v1/agents")
def list_agents() -> dict[str, Any]:
    assert pipeline is not None
    return {"agents": pipeline.list_agents()}


@app.get("/api/v1/violations")
def list_violations(
    violation_type: str | None = Query(None),
    plate: str | None = Query(None),
    limit: int = Query(50, le=500),
) -> list[dict[str, Any]]:
    assert pipeline is not None
    return pipeline.db.search(violation_type=violation_type, plate_number=plate, limit=limit)


@app.get("/api/v1/analytics")
def analytics(days: int = Query(30, ge=1, le=365)) -> dict[str, Any]:
    assert pipeline is not None
    return pipeline.get_analytics(days=days)


@app.get("/api/v1/evidence/{filename}")
def get_evidence(filename: str) -> FileResponse:
    assert pipeline is not None
    path = Path(pipeline.config["paths"]["evidence_dir"]) / filename
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Evidence not found")
    return FileResponse(path)
