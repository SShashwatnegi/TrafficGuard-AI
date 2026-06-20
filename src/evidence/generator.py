from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.models.schemas import BoundingBox, ProcessingResult, Violation

VIOLATION_COLORS = {
    "helmet_non_compliance": (0, 0, 255),
    "seatbelt_non_compliance": (0, 128, 255),
    "triple_riding": (0, 165, 255),
    "wrong_side_driving": (255, 0, 255),
    "stop_line_violation": (0, 255, 255),
    "red_light_violation": (0, 0, 200),
    "illegal_parking": (128, 0, 255),
}

DETECTION_COLOR = (0, 255, 0)


class EvidenceGenerator:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        image: np.ndarray,
        result: ProcessingResult,
        source_name: str,
    ) -> str:
        canvas = image.copy()
        h, w = canvas.shape[:2]

        for det in result.detections:
            self._draw_box(canvas, det, DETECTION_COLOR, f"{det.label} {det.confidence:.2f}")

        for v in result.violations:
            color = VIOLATION_COLORS.get(v.violation_type, (0, 0, 255))
            tera = v.metadata.get("tera", {})
            reg = tera.get("regulation_code", "")
            label = f"{v.violation_type} ({v.confidence:.2f})"
            if reg:
                label = f"{label} [{reg}]"
            if v.bbox:
                self._draw_box(canvas, v.bbox, color, label, thickness=3)

        for plate in result.plates:
            if plate.bbox:
                self._draw_box(
                    canvas, plate.bbox, (255, 255, 0),
                    f"PLATE: {plate.text}", thickness=2,
                )

        # Header banner
        header = np.zeros((50, w, 3), dtype=np.uint8)
        header[:] = (27, 58, 107)  # #1B3A6B
        ts = result.timestamp.replace("T", " ").replace("Z", " UTC")
        text = f"TrafficGuard AI | {source_name} | {ts} | Violations: {len(result.violations)}"
        cv2.putText(header, text, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.vstack([header, canvas])

        safe_name = Path(source_name).stem
        out_path = self.output_dir / f"{safe_name}_evidence.jpg"
        cv2.imwrite(str(out_path), canvas)
        return str(out_path)

    def _draw_box(
        self,
        image: np.ndarray,
        box: BoundingBox,
        color: tuple[int, int, int],
        label: str,
        thickness: int = 2,
    ) -> None:
        x1, y1, x2, y2 = map(int, [box.x1, box.y1, box.x2, box.y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
