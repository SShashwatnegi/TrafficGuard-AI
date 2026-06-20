from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    category: str = "unknown"

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Violation:
    violation_type: str
    confidence: float
    description: str
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.bbox:
            d["bbox"] = self.bbox.to_dict()
        return d


@dataclass
class PlateResult:
    text: str
    confidence: float
    bbox: BoundingBox | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"text": self.text, "confidence": self.confidence}
        if self.bbox:
            d["bbox"] = self.bbox.to_dict()
        return d


@dataclass
class ProcessingResult:
    image_path: str
    timestamp: str
    detections: list[BoundingBox] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    candidate_violations: list[Violation] = field(default_factory=list)
    plates: list[PlateResult] = field(default_factory=list)
    evidence_path: str | None = None
    evidence_package: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    preprocessing_applied: list[str] = field(default_factory=list)
    scene_graph: dict[str, Any] = field(default_factory=dict)
    spatial_data: dict[str, Any] = field(default_factory=dict)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    insights: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "detections": [d.to_dict() for d in self.detections],
            "violations": [v.to_dict() for v in self.violations],
            "candidate_violations": [v.to_dict() for v in self.candidate_violations],
            "plates": [p.to_dict() for p in self.plates],
            "evidence_path": self.evidence_path,
            "evidence_package": self.evidence_package,
            "processing_time_ms": self.processing_time_ms,
            "preprocessing_applied": self.preprocessing_applied,
            "scene_graph": self.scene_graph,
            "spatial_data": self.spatial_data,
            "agent_trace": self.agent_trace,
            "insights": self.insights,
        }


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
