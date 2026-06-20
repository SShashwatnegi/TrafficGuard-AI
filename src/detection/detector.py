from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO

from src.models.schemas import BoundingBox

VEHICLE_CLASSES = {
    "car": "four_wheeler",
    "truck": "heavy_vehicle",
    "bus": "heavy_vehicle",
    "motorcycle": "two_wheeler",
    "bicycle": "two_wheeler",
}

ROAD_USER_CLASSES = {
    "person": "pedestrian",
}

ALL_TARGET_CLASSES = {**VEHICLE_CLASSES, **ROAD_USER_CLASSES}


class ObjectDetector:
    """YOLO-based vehicle and road user detection."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
    ) -> None:
        self.model = YOLO(model_name)
        # Force YOLO to initialize its backend and fuse batch norms synchronously 
        # so it doesn't crash when accessed concurrently by multiple threads later
        self.model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        
        self.confidence = confidence
        self.iou = iou

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )
        detections: list[BoundingBox] = []
        if not results:
            return detections

        result = results[0]
        names: dict[int, str] = result.names
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, str(cls_id))
            if label not in ALL_TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            category = ALL_TARGET_CLASSES[label]
            detections.append(
                BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    label=label,
                    confidence=conf,
                    category=category,
                )
            )
        return detections

    @staticmethod
    def filter_by_label(detections: list[BoundingBox], labels: set[str]) -> list[BoundingBox]:
        return [d for d in detections if d.label in labels]

    @staticmethod
    def filter_by_category(detections: list[BoundingBox], categories: set[str]) -> list[BoundingBox]:
        return [d for d in detections if d.category in categories]
