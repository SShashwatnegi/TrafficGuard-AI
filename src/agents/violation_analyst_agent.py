from __future__ import annotations

from src.agents.base import AgentContext, BaseAgent
from src.violations.analyzers import build_analyzers


class ViolationAnalystAgent(BaseAgent):
    """Identifies candidate violations from scene graph relationships."""

    name = "ViolationAnalystAgent"

    def __init__(self, enabled_types: list[str], min_confidence: float = 0.5) -> None:
        self.analyzers = build_analyzers(enabled_types)
        self.min_confidence = min_confidence

    def execute(self, ctx: AgentContext) -> AgentContext:
        context = dict(ctx.config)
        context["scene_graph"] = ctx.scene_graph
        context["spatial_data"] = ctx.spatial_data

        is_red = context.get("is_red_light", True)

        candidates = []
        for analyzer in self.analyzers:
            found = analyzer.analyze(ctx.image, ctx.detections, context)
            for v in found:
                if v.confidence >= self.min_confidence * 0.8:
                    if not is_red and v.violation_type in ("red_light_violation", "stop_line_violation"):
                        continue
                    candidates.append(v)

        # Enrich candidates with graph context
        for v in candidates:
            if v.bbox and ctx.scene_graph:
                v.metadata["scene_graph_edges"] = _related_edges(ctx, v.bbox)

        ctx.candidate_violations = candidates
        return ctx

    def _summary(self, ctx: AgentContext) -> dict:
        types = [v.violation_type for v in ctx.candidate_violations]
        return {"candidates": len(ctx.candidate_violations), "types": types}


def _related_edges(ctx: AgentContext, bbox) -> list[dict]:
    if not ctx.scene_graph:
        return []
    edges = []
    for node in ctx.scene_graph.nodes.values():
        if node.bbox.label == bbox.label:
            from src.violations.analyzers import iou
            if iou(node.bbox, bbox) > 0.4:
                for edge in ctx.scene_graph.edges:
                    if edge.source == node.id or edge.target == node.id:
                        edges.append(edge.to_dict())
    return edges[:5]
