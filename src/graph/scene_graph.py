from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models.schemas import BoundingBox


@dataclass
class SceneNode:
    id: str
    label: str
    category: str
    bbox: BoundingBox
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "bbox": self.bbox.to_dict(),
            "attributes": self.attributes,
        }


@dataclass
class SceneEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class SceneGraph:
    """Graph-based scene representation linking detections and spatial relationships."""

    def __init__(self) -> None:
        self.nodes: dict[str, SceneNode] = {}
        self.edges: list[SceneEdge] = []
        self._counter = 0

    def add_detection(self, detection: BoundingBox) -> str:
        self._counter += 1
        node_id = f"{detection.label}_{self._counter}"
        self.nodes[node_id] = SceneNode(
            id=node_id,
            label=detection.label,
            category=detection.category,
            bbox=detection,
            attributes={"confidence": detection.confidence},
        )
        return node_id

    def add_infrastructure(self, label: str, bbox: BoundingBox, **attrs: Any) -> str:
        self._counter += 1
        node_id = f"{label}_{self._counter}"
        self.nodes[node_id] = SceneNode(
            id=node_id,
            label=label,
            category="infrastructure",
            bbox=bbox,
            attributes=attrs,
        )
        return node_id

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0, **meta: Any) -> None:
        if source in self.nodes and target in self.nodes:
            self.edges.append(SceneEdge(source, target, relation, weight, meta))

    def get_nodes_by_label(self, label: str) -> list[SceneNode]:
        return [n for n in self.nodes.values() if n.label == label]

    def get_nodes_by_category(self, category: str) -> list[SceneNode]:
        return [n for n in self.nodes.values() if n.category == category]

    def neighbors(self, node_id: str, relation: str | None = None) -> list[SceneNode]:
        out = []
        for edge in self.edges:
            if edge.source == node_id and (relation is None or edge.relation == relation):
                out.append(self.nodes[edge.target])
            elif edge.target == node_id and (relation is None or edge.relation == relation):
                out.append(self.nodes[edge.source])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "vehicles": len(self.get_nodes_by_category("four_wheeler"))
                + len(self.get_nodes_by_category("two_wheeler"))
                + len(self.get_nodes_by_category("heavy_vehicle")),
                "pedestrians": len(self.get_nodes_by_label("person")),
            },
        }
