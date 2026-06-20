from __future__ import annotations

import cv2
import numpy as np

from src.agents.base import AgentContext, BaseAgent
from src.models.schemas import BoundingBox
from src.violations.analyzers import iou, persons_on_vehicle


class SpatialAgent(BaseAgent):
    """Analyzes movement, direction, tracking, and spatial relationships."""

    name = "SpatialAgent"

    def execute(self, ctx: AgentContext) -> AgentContext:
        h, w = ctx.image.shape[:2]
        flow = ctx.config.get("expected_flow", "right")
        spatial: dict = {
            "image_size": {"width": w, "height": h},
            "expected_flow": flow,
            "tracks": [],
            "relationships": [],
        }

        if not ctx.scene_graph:
            ctx.spatial_data = spatial
            return ctx

        persons = ctx.scene_graph.get_nodes_by_label("person")
        motorcycles = ctx.scene_graph.get_nodes_by_label("motorcycle")
        vehicles = (
            ctx.scene_graph.get_nodes_by_category("four_wheeler")
            + ctx.scene_graph.get_nodes_by_category("heavy_vehicle")
            + ctx.scene_graph.get_nodes_by_category("two_wheeler")
        )

        # Build spatial graph edges
        for bike in motorcycles:
            riders = persons_on_vehicle(bike.bbox, [p.bbox for p in persons])
            for i, rider_bbox in enumerate(riders):
                rider_node = _match_node(persons, rider_bbox)
                if rider_node:
                    ctx.scene_graph.add_edge(rider_node.id, bike.id, "on_vehicle", weight=0.9, position=i + 1)
                    spatial["relationships"].append({
                        "type": "on_vehicle", "rider": rider_node.id, "vehicle": bike.id,
                    })

        for vehicle in vehicles:
            facing = self._estimate_facing(vehicle.bbox, ctx.image)
            vehicle.attributes["facing"] = facing
            vehicle.attributes["lane_zone"] = self._lane_zone(vehicle.bbox, w)
            vehicle.attributes["speed_estimate"] = self._estimate_speed(vehicle.id, vehicle.bbox, ctx.frame_history)

            ctx.scene_graph.add_edge(
                vehicle.id, vehicle.id, "facing", weight=1.0, direction=facing,
            )
            spatial["tracks"].append({
                "node_id": vehicle.id,
                "facing": facing,
                "lane_zone": vehicle.attributes["lane_zone"],
                "center": vehicle.bbox.center,
            })

        # Cross-relationships: vehicle near infrastructure
        infra_nodes = [n for n in ctx.scene_graph.nodes.values() if n.category == "infrastructure"]
        for vehicle in vehicles:
            for infra in infra_nodes:
                overlap = iou(vehicle.bbox, infra.bbox)
                if overlap > 0.01 or _vertical_cross(vehicle.bbox, infra.bbox, infra.label):
                    relation = "in_zone" if infra.label.startswith("no_parking") else "crosses"
                    ctx.scene_graph.add_edge(vehicle.id, infra.id, relation, weight=overlap or 0.5)
                    spatial["relationships"].append({
                        "type": relation, "vehicle": vehicle.id, "infrastructure": infra.id,
                    })

        ctx.spatial_data = spatial
        return ctx

    def _estimate_facing(self, vehicle: BoundingBox, image: np.ndarray) -> str | None:
        x1, y1, x2, y2 = map(int, [vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2])
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        left_energy = abs(sobelx[:, : crop.shape[1] // 2]).mean()
        right_energy = abs(sobelx[:, crop.shape[1] // 2 :]).mean()
        if abs(left_energy - right_energy) < 5:
            return "unknown"
        return "right" if right_energy > left_energy else "left"

    def _lane_zone(self, bbox: BoundingBox, width: int) -> str:
        cx, _ = bbox.center
        if cx < width * 0.33:
            return "left"
        if cx > width * 0.66:
            return "right"
        return "center"

    def _estimate_speed(self, node_id: str, bbox: BoundingBox, history: list[dict]) -> float:
        if not history:
            return 0.0
        prev = history[-1].get("tracks", {}).get(node_id)
        if not prev:
            return 0.0
        px, py = prev["center"]
        cx, cy = bbox.center
        return round(float(np.hypot(cx - px, cy - py)), 2)

    def _summary(self, ctx: AgentContext) -> dict:
        return {
            "tracks": len(ctx.spatial_data.get("tracks", [])),
            "relationships": len(ctx.spatial_data.get("relationships", [])),
        }


def _match_node(persons, bbox: BoundingBox):
    for p in persons:
        if iou(p.bbox, bbox) > 0.3:
            return p
    return None


def _vertical_cross(vehicle: BoundingBox, infra_bbox: BoundingBox, infra_label: str) -> bool:
    return vehicle.y2 > infra_bbox.y1 and infra_label == "stop_line"
