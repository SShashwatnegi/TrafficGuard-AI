from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def compute_detection_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Evaluate detection with per-image TP/FP/FN at IoU threshold."""
    tp = fp = fn = 0

    for pred_img, gt_img in zip(predictions, ground_truth):
        pred_boxes = pred_img.get("boxes", [])
        gt_boxes = gt_img.get("boxes", [])
        matched_gt = set()

        for pb in pred_boxes:
            found = False
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                if pb.get("label") == gb.get("label") and _box_iou(pb, gb) >= iou_threshold:
                    tp += 1
                    matched_gt.add(i)
                    found = True
                    break
            if not found:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return DetectionMetrics(precision, recall, f1, accuracy, tp, fp, fn)


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def compute_map(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_thresholds: list[float] | None = None,
) -> float:
    """Simplified mAP: mean of per-class AP at fixed IoU thresholds."""
    iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = []
    for thr in iou_thresholds:
        m = compute_detection_metrics(predictions, ground_truth, iou_threshold=thr)
        aps.append(m.precision * m.recall)
    return round(float(np.mean(aps)), 4) if aps else 0.0


def _box_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
