from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from src.models.schemas import BoundingBox, PlateResult


class PlateRecognizer:
    """Detect and OCR license plates from vehicle regions."""

    PLATE_PATTERN = re.compile(r"^[A-Z0-9]{5,12}$")

    def __init__(self, languages: list[str] | None = None, min_confidence: float = 0.4) -> None:
        self.min_confidence = min_confidence
        self._reader = None
        self.languages = languages or ["en"]

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
        return self._reader

    def recognize(
        self,
        image: np.ndarray,
        vehicle_detections: list[BoundingBox] | None = None,
    ) -> list[PlateResult]:
        plates: list[PlateResult] = []
        candidates = self._find_plate_candidates(image, vehicle_detections or [])

        reader = self._get_reader()
        for bbox, crop in candidates:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            ocr_results = reader.readtext(enhanced, detail=1, paragraph=False)
            for _bbox, text, conf in ocr_results:
                cleaned = re.sub(r"[^A-Za-z0-9]", "", text.upper())
                if len(cleaned) < 5 or conf < self.min_confidence:
                    continue
                if not self.PLATE_PATTERN.match(cleaned):
                    continue
                plates.append(PlateResult(text=cleaned, confidence=round(float(conf), 3), bbox=bbox))
        return self._dedupe(plates)

    def _find_plate_candidates(
        self,
        image: np.ndarray,
        vehicles: list[BoundingBox],
    ) -> list[tuple[BoundingBox, np.ndarray]]:
        h, w = image.shape[:2]
        candidates: list[tuple[BoundingBox, np.ndarray]] = []

        if vehicles:
            for v in vehicles:
                vh = v.y2 - v.y1
                x1 = int(max(0, v.x1))
                x2 = int(min(w, v.x2))
                y1 = int(max(0, v.y2 - vh * 0.25))
                y2 = int(min(h, v.y2 + 5))
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    candidates.append((
                        BoundingBox(x1, y1, x2, y2, "plate_region", 0.5, "plate"),
                        crop,
                    ))
        else:
            candidates.append((
                BoundingBox(0, int(h * 0.6), w, h, "full_lower", 0.3, "plate"),
                image[int(h * 0.6) :, :],
            ))

        # Edge-based plate-like rectangles in lower image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if 2.0 < aspect < 6.5 and bw > 60 and y > h * 0.4:
                crop = image[y : y + bh, x : x + bw]
                if crop.size > 0:
                    candidates.append((
                        BoundingBox(x, y, x + bw, y + bh, "plate", 0.4, "plate"),
                        crop,
                    ))
        return candidates[:12]

    def _dedupe(self, plates: list[PlateResult]) -> list[PlateResult]:
        seen: dict[str, PlateResult] = {}
        for p in plates:
            if p.text not in seen or p.confidence > seen[p.text].confidence:
                seen[p.text] = p
        return list(seen.values())
