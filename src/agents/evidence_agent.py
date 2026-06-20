from __future__ import annotations

import json
from pathlib import Path

from src.agents.base import AgentContext, BaseAgent
from src.evidence.generator import EvidenceGenerator
from src.models.schemas import ProcessingResult


class EvidenceAgent(BaseAgent):
    """Produces explainable evidence packages with legal justifications."""

    name = "EvidenceAgent"

    def __init__(self, generator: EvidenceGenerator) -> None:
        self.generator = generator

    def execute(self, ctx: AgentContext) -> AgentContext:
        result = ProcessingResult(
            image_path=ctx.image_path,
            timestamp=ctx.timestamp,
            detections=ctx.detections,
            violations=ctx.validated_violations,
            plates=ctx.plates,
            preprocessing_applied=ctx.preprocessing_steps,
        )

        source_name = Path(ctx.image_path).name
        ctx.evidence_path = self.generator.generate(ctx.image, result, source_name)

        packages = []
        for v in ctx.validated_violations:
            tera = v.metadata.get("tera", {})
            packages.append({
                "violation_type": v.violation_type,
                "confidence": v.confidence,
                "description": v.description,
                "regulation_code": tera.get("regulation_code", ""),
                "legal_justification": tera.get("legal_justification", ""),
                "reasoning": tera.get("reasoning", ""),
                "bbox": v.bbox.to_dict() if v.bbox else None,
            })

        ctx.evidence_package = {
            "image_path": ctx.image_path,
            "timestamp": ctx.timestamp,
            "evidence_image": ctx.evidence_path,
            "violation_count": len(ctx.validated_violations),
            "plates": [p.to_dict() for p in ctx.plates],
            "violations": packages,
            "agent_trace": ctx.agent_trace,
            "scene_graph_stats": ctx.scene_graph.to_dict().get("stats") if ctx.scene_graph else {},
        }

        # Save JSON evidence package alongside image
        pkg_path = Path(ctx.evidence_path).with_suffix(".json") if ctx.evidence_path else None
        if pkg_path:
            pkg_path.write_text(json.dumps(ctx.evidence_package, indent=2), encoding="utf-8")
            ctx.evidence_package["package_path"] = str(pkg_path)

        return ctx

    def _summary(self, ctx: AgentContext) -> dict:
        return {
            "evidence_path": ctx.evidence_path,
            "packages": len(ctx.validated_violations),
        }
