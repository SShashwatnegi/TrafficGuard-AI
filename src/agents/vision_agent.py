from __future__ import annotations

from src.agents.base import AgentContext, BaseAgent
from src.detection.detector import ObjectDetector
from src.graph.scene_graph import SceneGraph


class VisionAgent(BaseAgent):
    """Detects vehicles, road users, and infrastructure elements."""

    name = "VisionAgent"

    def __init__(self, detector: ObjectDetector) -> None:
        self.detector = detector

    def execute(self, ctx: AgentContext) -> AgentContext:
        detections = self.detector.detect(ctx.image)
        ctx.detections = detections

        graph = SceneGraph()
        for det in detections:
            graph.add_detection(det)

        # Infer infrastructure nodes from scene context
        h, w = ctx.image.shape[:2]
        stop_y = ctx.config.get("stop_line_y")
        if stop_y is not None and stop_y <= 1.0:
            stop_y = stop_y * h
        else:
            stop_y = h * 0.72

        from src.models.schemas import BoundingBox
        graph.add_infrastructure(
            "stop_line",
            BoundingBox(0, stop_y - 2, w, stop_y + 2, "stop_line", 1.0, "infrastructure"),
            inferred=True,
        )

        zones = ctx.config.get("no_parking_zones") or [[0.05, 0.65, 0.45, 0.98], [0.55, 0.65, 0.95, 0.98]]
        for i, zone in enumerate(zones):
            zx1, zy1, zx2, zy2 = zone
            if all(0 <= v <= 1 for v in zone):
                zx1, zy1, zx2, zy2 = zx1 * w, zy1 * h, zx2 * w, zy2 * h
            graph.add_infrastructure(
                f"no_parking_{i}",
                BoundingBox(zx1, zy1, zx2, zy2, "no_parking_zone", 1.0, "infrastructure"),
                zone_type="no_parking",
            )

        ctx.scene_graph = graph
        return ctx

    def _summary(self, ctx: AgentContext) -> dict:
        cats: dict[str, int] = {}
        for d in ctx.detections:
            cats[d.label] = cats.get(d.label, 0) + 1
        return {"detection_count": len(ctx.detections), "by_label": cats}
