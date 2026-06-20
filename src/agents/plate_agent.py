from __future__ import annotations

from src.agents.base import AgentContext, BaseAgent
from src.lpr.plate_recognizer import PlateRecognizer
from src.models.schemas import BoundingBox


class PlateAgent(BaseAgent):
    """Automatic Number Plate Recognition (ANPR)."""

    name = "PlateAgent"

    def __init__(self, recognizer: PlateRecognizer | None) -> None:
        self.recognizer = recognizer

    def execute(self, ctx: AgentContext) -> AgentContext:
        if not self.recognizer:
            return ctx

        vehicles = [
            d for d in ctx.detections
            if d.category in ("four_wheeler", "heavy_vehicle", "two_wheeler")
        ]
        ctx.plates = self.recognizer.recognize(ctx.image, vehicles)

        if ctx.scene_graph:
            vehicle_nodes = (
                ctx.scene_graph.get_nodes_by_category("four_wheeler")
                + ctx.scene_graph.get_nodes_by_category("heavy_vehicle")
                + ctx.scene_graph.get_nodes_by_category("two_wheeler")
            )
            for plate in ctx.plates:
                if not plate.bbox:
                    continue
                best_node = None
                best_iou = 0.0
                for node in vehicle_nodes:
                    score = _iou_boxes(plate.bbox, node.bbox)
                    if score > best_iou:
                        best_iou = score
                        best_node = node
                if best_node and best_iou > 0.05:
                    ctx.scene_graph.add_edge(
                        best_node.id, best_node.id,
                        "plate_of",
                        weight=plate.confidence,
                        plate_text=plate.text,
                    )
                    best_node.attributes["plate"] = plate.text
                    best_node.attributes["plate_confidence"] = plate.confidence

        return ctx

    def _summary(self, ctx: AgentContext) -> dict:
        return {"plates_found": len(ctx.plates), "texts": [p.text for p in ctx.plates]}


def _iou_boxes(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
