from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from src.models.schemas import BoundingBox, Violation


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def persons_on_vehicle(
    vehicle: BoundingBox,
    persons: list[BoundingBox],
    iou_threshold: float = 0.05,
) -> list[BoundingBox]:
    expanded = BoundingBox(
        x1=vehicle.x1 - 10,
        y1=vehicle.y1 - 30,
        x2=vehicle.x2 + 10,
        y2=vehicle.y2 + 10,
        label=vehicle.label,
        confidence=vehicle.confidence,
        category=vehicle.category,
    )
    matched = []
    for p in persons:
        if iou(expanded, p) > iou_threshold or _center_inside(p, expanded):
            matched.append(p)
    return matched


def _center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    cx, cy = inner.center
    return outer.x1 <= cx <= outer.x2 and outer.y1 <= cy <= outer.y2


class ViolationAnalyzer:
    violation_type: str = "base"

    def analyze(
        self,
        image: np.ndarray,
        detections: list[BoundingBox],
        context: dict[str, Any],
    ) -> list[Violation]:
        raise NotImplementedError


class HelmetAnalyzer(ViolationAnalyzer):
    violation_type = "helmet_non_compliance"

    def analyze(self, image, detections, context):
        violations = []
        motorcycles = [d for d in detections if d.label == "motorcycle"]
        persons = [d for d in detections if d.label == "person"]
        h, w = image.shape[:2]

        for bike in motorcycles:
            riders = persons_on_vehicle(bike, persons)
            if not riders:
                continue
            for rider in riders:
                head_roi = self._head_region(rider, h, w)
                if head_roi is None:
                    continue
                x1, y1, x2, y2 = head_roi
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                helmet_score = self._helmet_likelihood(crop)
                if helmet_score < 0.45:
                    conf = min(0.95, 0.55 + (0.45 - helmet_score))
                    violations.append(Violation(
                        violation_type=self.violation_type,
                        confidence=round(conf, 3),
                        description="Rider detected without identifiable helmet",
                        bbox=rider,
                        metadata={"helmet_score": round(helmet_score, 3)},
                    ))
        return violations

    def _head_region(self, rider: BoundingBox, h: int, w: int) -> tuple[int, int, int, int] | None:
        rh = rider.y2 - rider.y1
        x1 = int(max(0, rider.x1))
        y1 = int(max(0, rider.y1))
        x2 = int(min(w, rider.x2))
        y2 = int(min(h, rider.y1 + rh * 0.35))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _helmet_likelihood(self, head_crop: np.ndarray) -> float:
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean() / 255.0
        sat = hsv[:, :, 1].mean() / 255.0
        val = hsv[:, :, 2].mean() / 255.0
        # Helmets often have uniform color regions and rounded edge contours
        roundness = min(1.0, edge_density * 3.5)
        uniformity = 1.0 - min(1.0, float(np.std(gray)) / 80.0)
        return min(1.0, 0.35 * roundness + 0.35 * uniformity + 0.15 * sat + 0.15 * val)


class SeatbeltAnalyzer(ViolationAnalyzer):
    violation_type = "seatbelt_non_compliance"

    def analyze(self, image, detections, context):
        violations = []
        cars = [d for d in detections if d.label in ("car", "truck", "bus")]
        persons = [d for d in detections if d.label == "person"]
        h, w = image.shape[:2]

        for vehicle in cars:
            cabin = self._cabin_region(vehicle, w, h)
            if cabin is None:
                continue
            x1, y1, x2, y2 = cabin
            cabin_persons = [
                p for p in persons
                if _center_inside(p, BoundingBox(x1, y1, x2, y2, "roi", 1.0))
            ]
            if not cabin_persons:
                continue
            crop = image[y1:y2, x1:x2]
            belt_score = self._seatbelt_likelihood(crop)
            if belt_score < 0.4:
                conf = min(0.85, 0.5 + (0.4 - belt_score))
                violations.append(Violation(
                    violation_type=self.violation_type,
                    confidence=round(conf, 3),
                    description="Occupant in vehicle cabin without visible seatbelt",
                    bbox=vehicle,
                    metadata={"seatbelt_score": round(belt_score, 3)},
                ))
        return violations

    def _cabin_region(self, vehicle: BoundingBox, w: int, h: int) -> tuple[int, int, int, int] | None:
        vw = vehicle.x2 - vehicle.x1
        vh = vehicle.y2 - vehicle.y1
        x1 = int(max(0, vehicle.x1 + vw * 0.15))
        x2 = int(min(w, vehicle.x2 - vw * 0.15))
        y1 = int(max(0, vehicle.y1 + vh * 0.15))
        y2 = int(min(h, vehicle.y1 + vh * 0.55))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _seatbelt_likelihood(self, cabin_crop: np.ndarray) -> float:
        gray = cv2.cvtColor(cabin_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=8)
        if lines is None:
            return 0.2
        diagonal = 0
        for line in lines[:20]:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if 25 < angle < 75:
                diagonal += 1
        return min(1.0, diagonal / 5.0)


class TripleRidingAnalyzer(ViolationAnalyzer):
    violation_type = "triple_riding"

    def analyze(self, image, detections, context):
        violations = []
        motorcycles = [d for d in detections if d.label == "motorcycle"]
        persons = [d for d in detections if d.label == "person"]

        for bike in motorcycles:
            riders = persons_on_vehicle(bike, persons)
            if len(riders) >= 3:
                conf = min(0.98, 0.7 + 0.1 * (len(riders) - 3))
                violations.append(Violation(
                    violation_type=self.violation_type,
                    confidence=round(conf, 3),
                    description=f"{len(riders)} persons detected on two-wheeler (max 2 allowed)",
                    bbox=bike,
                    metadata={"rider_count": len(riders)},
                ))
        return violations


class WrongSideAnalyzer(ViolationAnalyzer):
    violation_type = "wrong_side_driving"

    def analyze(self, image, detections, context):
        violations = []
        h, w = image.shape[:2]
        flow_direction = context.get("expected_flow", "right")  # right = LHD countries driving right
        vehicles = [d for d in detections if d.category in ("four_wheeler", "two_wheeler", "heavy_vehicle")]

        for v in vehicles:
            cx, _ = v.center
            facing = self._estimate_facing(v, image)
            if facing is None:
                continue
            wrong = (
                (flow_direction == "right" and facing == "left" and cx > w * 0.35)
                or (flow_direction == "left" and facing == "right" and cx < w * 0.65)
            )
            if wrong:
                violations.append(Violation(
                    violation_type=self.violation_type,
                    confidence=0.72,
                    description="Vehicle orientation inconsistent with expected traffic flow",
                    bbox=v,
                    metadata={"facing": facing, "flow": flow_direction},
                ))
        return violations

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
            return None
        return "right" if right_energy > left_energy else "left"


class StopLineAnalyzer(ViolationAnalyzer):
    violation_type = "stop_line_violation"

    def analyze(self, image, detections, context):
        violations = []
        h, w = image.shape[:2]
        stop_y = context.get("stop_line_y")
        if stop_y is None:
            stop_y = self._detect_stop_line(image)
        else:
            stop_y = float(stop_y)
            if stop_y <= 1.0:
                stop_y = stop_y * h

        vehicles = [d for d in detections if d.category in ("four_wheeler", "two_wheeler", "heavy_vehicle")]
        for v in vehicles:
            if v.y2 > stop_y + 5:
                overlap = (v.y2 - stop_y) / max(1.0, v.y2 - v.y1)
                conf = min(0.92, 0.55 + overlap * 0.4)
                violations.append(Violation(
                    violation_type=self.violation_type,
                    confidence=round(conf, 3),
                    description="Vehicle crossed detected stop line",
                    bbox=v,
                    metadata={"stop_line_y": round(stop_y, 1)},
                ))
        return violations

    def _detect_stop_line(self, image: np.ndarray) -> float:
        h, w = image.shape[:2]
        lower = image[int(h * 0.55) :, :]
        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=w // 4, maxLineGap=20)
        candidates = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 8 and abs(x2 - x1) > w // 5:
                    candidates.append(int(h * 0.55) + (y1 + y2) // 2)
        return float(np.median(candidates)) if candidates else h * 0.72


class RedLightAnalyzer(ViolationAnalyzer):
    violation_type = "red_light_violation"

    def analyze(self, image, detections, context):
        violations = []
        h, w = image.shape[:2]
        red_regions = self._find_red_lights(image)
        if not red_regions:
            return violations

        intersection_y = h * context.get("intersection_start", 0.45)
        vehicles = [d for d in detections if d.category in ("four_wheeler", "two_wheeler", "heavy_vehicle")]

        for v in vehicles:
            if v.y2 < intersection_y:
                continue
            cx, _ = v.center
            for rx1, ry1, rx2, ry2 in red_regions:
                if abs(cx - (rx1 + rx2) / 2) < w * 0.35:
                    violations.append(Violation(
                        violation_type=self.violation_type,
                        confidence=0.78,
                        description="Vehicle in intersection while traffic signal is red",
                        bbox=v,
                        metadata={"signal_region": [rx1, ry1, rx2, ry2]},
                    ))
                    break
        return violations

    def _find_red_lights(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 120, 120])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 120, 120])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if 8 < bw < 80 and 8 < bh < 80 and 0.6 < bw / max(bh, 1) < 1.6:
                regions.append((x, y, x + bw, y + bh))
        return regions


class IllegalParkingAnalyzer(ViolationAnalyzer):
    violation_type = "illegal_parking"

    def analyze(self, image, detections, context):
        violations = []
        h, w = image.shape[:2]
        zones = context.get("no_parking_zones") or [[0.05, 0.65, 0.45, 0.98], [0.55, 0.65, 0.95, 0.98]]
        vehicles = [d for d in detections if d.category in ("four_wheeler", "heavy_vehicle")]

        for zone in zones:
            if len(zone) == 4 and all(0 <= v <= 1 for v in zone):
                zx1, zy1, zx2, zy2 = zone[0] * w, zone[1] * h, zone[2] * w, zone[3] * h
            else:
                zx1, zy1, zx2, zy2 = zone
            zone_box = BoundingBox(zx1, zy1, zx2, zy2, "no_parking", 1.0, "zone")

            for v in vehicles:
                overlap = iou(v, zone_box)
                if overlap > 0.25:
                    violations.append(Violation(
                        violation_type=self.violation_type,
                        confidence=round(min(0.9, 0.55 + overlap), 3),
                        description="Vehicle detected in no-parking zone",
                        bbox=v,
                        metadata={"zone": [zx1, zy1, zx2, zy2], "overlap": round(overlap, 3)},
                    ))
        return violations


ANALYZER_REGISTRY: dict[str, type[ViolationAnalyzer]] = {
    "helmet_non_compliance": HelmetAnalyzer,
    "seatbelt_non_compliance": SeatbeltAnalyzer,
    "triple_riding": TripleRidingAnalyzer,
    "wrong_side_driving": WrongSideAnalyzer,
    "stop_line_violation": StopLineAnalyzer,
    "red_light_violation": RedLightAnalyzer,
    "illegal_parking": IllegalParkingAnalyzer,
}


def build_analyzers(enabled: list[str]) -> list[ViolationAnalyzer]:
    analyzers = []
    for name in enabled:
        cls = ANALYZER_REGISTRY.get(name)
        if cls:
            analyzers.append(cls())
    return analyzers
