from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """Enhance traffic images for robust detection under varying conditions."""

    def __init__(
        self,
        enable_clahe: bool = True,
        enable_denoise: bool = True,
        enable_deblur: bool = True,
        clahe_clip_limit: float = 2.5,
        clahe_tile_size: int = 8,
    ) -> None:
        self.enable_clahe = enable_clahe
        self.enable_denoise = enable_denoise
        self.enable_deblur = enable_deblur
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size),
        )

    def _is_low_light(self, gray: np.ndarray) -> bool:
        return float(np.mean(gray)) < 85.0

    def _estimate_blur(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _gamma_correct(self, image: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / max(gamma, 0.1)
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def process(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        steps: list[str] = ["normalize"]
        out = image.copy()

        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)

        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        if self.enable_clahe and self._is_low_light(gray):
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            out = self._gamma_correct(out, 1.2)
            steps.append("clahe_low_light")

        if self.enable_denoise:
            out = cv2.bilateralFilter(out, d=7, sigmaColor=75, sigmaSpace=75)
            steps.append("denoise")

        blur_score = self._estimate_blur(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
        if self.enable_deblur and blur_score < 120.0:
            out = self._sharpen(out)
            steps.append("deblur")

        # Reduce shadow dominance via mild contrast normalization
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        steps.append("shadow_balance")

        return out, steps
