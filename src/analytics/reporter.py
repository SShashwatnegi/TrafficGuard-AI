from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.storage.database import Database


class AnalyticsReporter:
    def __init__(self, db: Database, reports_dir: str | Path) -> None:
        self.db = db
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def summary(self, days: int = 30) -> dict[str, Any]:
        records = self.db.all_records(limit=5000)
        if not records:
            return {"total_records": 0, "violations_by_type": {}, "avg_confidence": 0.0}

        cutoff = datetime.utcnow() - timedelta(days=days)
        filtered = []
        for r in records:
            ts = r.get("timestamp")
            if not ts:
                continue
            dt = datetime.fromisoformat(ts)
            if dt >= cutoff:
                filtered.append(r)

        if not filtered:
            filtered = records

        types = [r["violation_type"] for r in filtered if r["violation_type"] != "none"]
        confidences = [r["confidence"] for r in filtered if r["violation_type"] != "none"]
        plates = [r["plate_number"] for r in filtered if r.get("plate_number")]

        return {
            "total_records": len(filtered),
            "violation_events": len(types),
            "violations_by_type": dict(Counter(types)),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "unique_plates": len(set(plates)),
            "avg_processing_ms": round(
                sum(r.get("processing_time_ms", 0) for r in filtered) / max(len(filtered), 1), 2
            ),
        }

    def generate_charts(self, days: int = 30) -> dict[str, str]:
        summary = self.summary(days)
        paths: dict[str, str] = {}

        by_type = summary.get("violations_by_type", {})
        if by_type:
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = list(by_type.keys())
            values = list(by_type.values())
            ax.barh(labels, values, color="#2E75B6")
            ax.set_xlabel("Count")
            ax.set_title("Traffic Violations by Type")
            fig.tight_layout()
            p = self.reports_dir / "violations_by_type.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths["violations_by_type"] = str(p)

        records = self.db.all_records()
        if records:
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            daily = df[df["violation_type"] != "none"].groupby("date").size()
            if not daily.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                daily.plot(kind="line", ax=ax, color="#1B3A6B", marker="o")
                ax.set_title("Daily Violation Trend")
                ax.set_ylabel("Violations")
                fig.tight_layout()
                p = self.reports_dir / "daily_trend.png"
                fig.savefig(p, dpi=120)
                plt.close(fig)
                paths["daily_trend"] = str(p)

        return paths

    def export_csv(self, path: str | Path | None = None) -> str:
        records = self.db.all_records()
        out = Path(path) if path else self.reports_dir / "violations_export.csv"
        pd.DataFrame(records).to_csv(out, index=False)
        return str(out)
