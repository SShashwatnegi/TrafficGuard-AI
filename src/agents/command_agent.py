from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from src.agents.insight_agent import InsightAgent
from src.storage.database import Database


class CommandAgent:
    """Natural-language interface for enforcement officers."""

    name = "CommandAgent"

    def __init__(self, db: Database, insight_agent: InsightAgent) -> None:
        self.db = db
        self.insight_agent = insight_agent

    def execute(self, query: str, days: int = 30) -> dict[str, Any]:
        query_lower = query.lower().strip()
        intent = self._parse_intent(query_lower)

        if intent == "repeat_offenders":
            records = self.db.all_records()
            offenders = self.insight_agent._repeat_offenders(records)
            return {
                "intent": intent,
                "query": query,
                "response": self._format_repeat_offenders(offenders),
                "data": offenders,
            }

        if intent == "hotspots":
            records = self.db.all_records()
            hotspots = self.insight_agent._hotspots(records)
            return {
                "intent": intent,
                "query": query,
                "response": self._format_hotspots(hotspots),
                "data": hotspots,
            }

        if intent == "summary":
            records = self.db.all_records()
            insights = self.insight_agent.analyze(records, days)
            return {
                "intent": intent,
                "query": query,
                "response": self._format_summary(insights),
                "data": insights,
            }

        if intent == "search_violation":
            vtype = self._extract_violation_type(query_lower)
            records = self.db.search(violation_type=vtype, limit=100)
            return {
                "intent": intent,
                "query": query,
                "response": f"Found {len(records)} records for '{vtype}'.",
                "data": records,
            }

        if intent == "search_plate":
            plate = self._extract_plate(query_lower)
            records = self.db.search(plate_number=plate, limit=100)
            return {
                "intent": intent,
                "query": query,
                "response": f"Found {len(records)} records for plate '{plate}'.",
                "data": records,
            }

        if intent == "recent":
            records = self.db.all_records(limit=200)
            cutoff = datetime.utcnow() - timedelta(days=days)
            recent = [
                r for r in records
                if r.get("timestamp") and datetime.fromisoformat(r["timestamp"]) >= cutoff
            ]
            return {
                "intent": intent,
                "query": query,
                "response": f"{len(recent)} violation records in the last {days} days.",
                "data": recent[:50],
            }

        return {
            "intent": "help",
            "query": query,
            "response": (
                "I can help with: 'show repeat offenders', 'violation hotspots', "
                "'summary report', 'helmet violations', 'search plate ABC1234', "
                "'recent violations last 7 days'."
            ),
            "data": [],
        }

    def _parse_intent(self, query: str) -> str:
        if any(k in query for k in ("repeat offender", "repeat offenders", "frequent violator")):
            return "repeat_offenders"
        if any(k in query for k in ("hotspot", "hotspots", "most common violation")):
            return "hotspots"
        if any(k in query for k in ("summary", "overview", "report", "statistics")):
            return "summary"
        if any(k in query for k in ("plate", "registration", "number plate")):
            return "search_plate"
        if any(k in query for k in ("recent", "last week", "last 7 days", "today")):
            return "recent"
        for vtype in (
            "helmet", "seatbelt", "triple riding", "wrong side", "stop line",
            "red light", "illegal parking", "parking",
        ):
            if vtype in query:
                return "search_violation"
        return "help"

    def _extract_violation_type(self, query: str) -> str:
        mapping = {
            "helmet": "helmet_non_compliance",
            "seatbelt": "seatbelt_non_compliance",
            "triple": "triple_riding",
            "wrong side": "wrong_side_driving",
            "stop line": "stop_line_violation",
            "red light": "red_light_violation",
            "parking": "illegal_parking",
        }
        for key, vtype in mapping.items():
            if key in query:
                return vtype
        return "helmet_non_compliance"

    def _extract_plate(self, query: str) -> str:
        match = re.search(r"[A-Z0-9]{5,12}", query.upper())
        if match:
            return match.group()
        words = query.split()
        return words[-1].upper() if words else ""

    def _format_hotspots(self, hotspots: list[dict]) -> str:
        if not hotspots:
            return "No hotspot data available yet."
        lines = ["Top violation hotspots:"]
        for h in hotspots[:5]:
            lines.append(f"  • {h['violation_type']}: {h['count']} cases ({h['share_pct']}%) — {h['severity']} severity")
        return "\n".join(lines)

    def _format_repeat_offenders(self, offenders: list[dict]) -> str:
        if not offenders:
            return "No repeat offenders identified yet."
        lines = ["Repeat offenders:"]
        for o in offenders[:5]:
            lines.append(
                f"  • Plate {o['plate_number']}: {o['violation_count']} violations "
                f"({o['risk_level']} risk)"
            )
        return "\n".join(lines)

    def _format_summary(self, insights: dict) -> str:
        hotspots = insights.get("hotspots", [])
        offenders = insights.get("repeat_offenders", [])
        return (
            f"Enforcement summary: {len(hotspots)} violation categories tracked, "
            f"{len(offenders)} repeat offenders identified. "
            f"Top priority: {', '.join(insights.get('enforcement_priority', [])[:3]) or 'N/A'}."
        )
