from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from src.agents.base import AgentContext, BaseAgent
from src.storage.database import Database


class InsightAgent(BaseAgent):
    """Hotspot detection, repeat offender analysis, and enforcement intelligence."""

    name = "InsightAgent"

    def __init__(self, db: Database) -> None:
        self.db = db

    def execute(self, ctx: AgentContext) -> AgentContext:
        days = ctx.config.get("analytics", {}).get("default_days", 30)
        records = self.db.all_records(limit=5000)
        ctx.insights.update(self.analyze(records, days))
        return ctx

    def analyze(self, records: list[dict[str, Any]], days: int = 30) -> dict[str, Any]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        filtered = []
        for r in records:
            ts = r.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if dt >= cutoff:
                filtered.append(r)

        if not filtered:
            filtered = records

        return {
            "hotspots": self._hotspots(filtered),
            "repeat_offenders": self._repeat_offenders(filtered),
            "violation_trends": self._trends(filtered),
            "enforcement_priority": self._priorities(filtered),
        }

    def _hotspots(self, records: list[dict]) -> list[dict]:
        by_type = Counter(
            r["violation_type"] for r in records if r.get("violation_type") not in (None, "none")
        )
        total = sum(by_type.values()) or 1
        return [
            {
                "violation_type": vtype,
                "count": count,
                "share_pct": round(100 * count / total, 1),
                "severity": "high" if count / total > 0.3 else "medium" if count / total > 0.15 else "low",
            }
            for vtype, count in by_type.most_common(10)
        ]

    def _repeat_offenders(self, records: list[dict]) -> list[dict]:
        plate_counts: dict[str, list] = defaultdict(list)
        for r in records:
            plate = r.get("plate_number")
            if plate and r.get("violation_type") != "none":
                plate_counts[plate].append(r)

        offenders = []
        for plate, events in plate_counts.items():
            if len(events) >= 2:
                types = Counter(e["violation_type"] for e in events)
                offenders.append({
                    "plate_number": plate,
                    "violation_count": len(events),
                    "violation_types": dict(types),
                    "risk_level": "high" if len(events) >= 5 else "medium",
                })
        offenders.sort(key=lambda x: x["violation_count"], reverse=True)
        return offenders[:20]

    def _trends(self, records: list[dict]) -> dict:
        daily: dict[str, int] = defaultdict(int)
        for r in records:
            if r.get("violation_type") == "none":
                continue
            ts = r.get("timestamp", "")[:10]
            if ts:
                daily[ts] += 1
        return dict(sorted(daily.items())[-14:])

    def _priorities(self, records: list[dict]) -> list[str]:
        hotspots = self._hotspots(records)
        return [h["violation_type"] for h in hotspots if h["severity"] == "high"][:5]

    def _summary(self, ctx: AgentContext) -> dict:
        return {
            "hotspots": len(ctx.insights.get("hotspots", [])),
            "repeat_offenders": len(ctx.insights.get("repeat_offenders", [])),
        }
