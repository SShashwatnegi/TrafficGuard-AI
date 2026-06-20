#!/usr/bin/env python3
"""Run evaluation metrics on annotated ground-truth JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_detection_metrics,
    compute_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TrafficGuard AI predictions")
    parser.add_argument("ground_truth", type=str, help="JSON file with ground truth annotations")
    parser.add_argument("predictions", type=str, help="JSON file with model predictions")
    args = parser.parse_args()

    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    det_metrics = compute_detection_metrics(pred.get("detections", []), gt.get("detections", []))
    mAP = compute_map(pred.get("detections", []), gt.get("detections", []))

    print("=== Detection Metrics ===")
    print(json.dumps(det_metrics.to_dict(), indent=2))
    print(f"mAP: {mAP}")

    if gt.get("violations") and pred.get("violations"):
        y_true = [v["type"] for v in gt["violations"]]
        y_pred = [v["type"] for v in pred["violations"]]
        cls_metrics = compute_classification_metrics(y_true, y_pred)
        print("\n=== Violation Classification ===")
        print(json.dumps({k: v for k, v in cls_metrics.items() if k != "report"}, indent=2))
        print(cls_metrics["report"])


if __name__ == "__main__":
    main()
