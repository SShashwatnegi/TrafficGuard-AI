from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.graph.scene_graph import SceneGraph
from src.models.schemas import BoundingBox, PlateResult, Violation


@dataclass
class AgentContext:
    """Shared state passed between agents in the multi-agent pipeline."""

    raw_image: np.ndarray
    image: np.ndarray
    image_path: str
    timestamp: str
    config: dict[str, Any]

    scene_graph: SceneGraph | None = None
    detections: list[BoundingBox] = field(default_factory=list)
    plates: list[PlateResult] = field(default_factory=list)
    spatial_data: dict[str, Any] = field(default_factory=dict)
    candidate_violations: list[Violation] = field(default_factory=list)
    validated_violations: list[Violation] = field(default_factory=list)
    evidence_path: str | None = None
    evidence_package: dict[str, Any] = field(default_factory=dict)
    insights: dict[str, Any] = field(default_factory=dict)

    preprocessing_steps: list[str] = field(default_factory=list)
    frame_history: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)

    def log_agent(self, agent_name: str, status: str, details: dict[str, Any] | None = None, duration_ms: float = 0) -> None:
        self.agent_trace.append({
            "agent": agent_name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "details": details or {},
        })


class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def execute(self, ctx: AgentContext) -> AgentContext:
        ...

    def run(self, ctx: AgentContext) -> AgentContext:
        t0 = time.perf_counter()
        try:
            ctx = self.execute(ctx)
            duration = (time.perf_counter() - t0) * 1000
            ctx.log_agent(self.name, "completed", self._summary(ctx), duration)
        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            ctx.log_agent(self.name, "error", {"error": str(exc)}, duration)
            raise
        return ctx

    def _summary(self, ctx: AgentContext) -> dict[str, Any]:
        return {}
