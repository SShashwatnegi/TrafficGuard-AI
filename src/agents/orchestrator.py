from __future__ import annotations

import time
from typing import Any

from src.agents.base import AgentContext, BaseAgent
from src.agents.command_agent import CommandAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.insight_agent import InsightAgent
from src.agents.plate_agent import PlateAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.spatial_agent import SpatialAgent
from src.agents.tera_agent import TeraAgent
from src.agents.tera_llm_agent import TeraLlmAgent
from src.agents.violation_analyst_agent import ViolationAnalystAgent
from src.agents.vision_agent import VisionAgent
from src.models.schemas import utc_now_iso


class AgentOrchestrator:
    """Coordinates the multi-agent TrafficGuard AI enforcement workflow."""

    def __init__(
        self,
        preprocessing: PreprocessingAgent,
        vision: VisionAgent,
        plate: PlateAgent,
        spatial: SpatialAgent,
        violation_analyst: ViolationAnalystAgent,
        tera: TeraAgent,
        tera_llm: TeraLlmAgent,
        evidence: EvidenceAgent,
        insight: InsightAgent,
        command: CommandAgent,
    ) -> None:
        self.agents: list[BaseAgent] = [
            preprocessing,
            vision,
            plate,
            spatial,
            violation_analyst,
            tera,
            tera_llm,
            evidence,
            insight,
        ]
        self.command = command

    def run(self, raw_image, image_path: str, config: dict[str, Any], frame_history: list | None = None) -> AgentContext:
        ctx = AgentContext(
            raw_image=raw_image,
            image=raw_image,
            image_path=image_path,
            timestamp=utc_now_iso(),
            config=config,
            frame_history=frame_history or [],
        )

        t0 = time.perf_counter()
        for agent in self.agents:
            ctx = agent.run(ctx)
        ctx.insights["total_pipeline_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return ctx

    def query(self, text: str, days: int = 30) -> dict[str, Any]:
        return self.command.execute(text, days=days)

    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents] + [self.command.name]
