from __future__ import annotations

from copy import deepcopy

from src.agents.base import AgentContext, BaseAgent
from src.models.schemas import Violation
from src.regulations.rules import get_rule, is_emergency_entity
from src.violations.analyzers import iou


class TeraAgent(BaseAgent):
    """Traffic Enforcement Reasoning Agent — validates, reasons, and filters violations."""

    name = "TERA"

    def __init__(self, global_min_confidence: float = 0.5) -> None:
        self.global_min = global_min_confidence

    def execute(self, ctx: AgentContext) -> AgentContext:
        validated: list[Violation] = []
        rejected: list[dict] = []

        for candidate in ctx.candidate_violations:
            decision = self._reason(candidate, ctx)
            if decision["approved"]:
                v = deepcopy(candidate)
                v.confidence = decision["adjusted_confidence"]
                v.metadata["tera"] = {
                    "regulation_code": decision["regulation_code"],
                    "legal_justification": decision["justification"],
                    "reasoning": decision["reasoning"],
                    "status": "approved",
                    "reviewer": "TERA-Rules",
                }
                validated.append(v)
            else:
                rejected.append({
                    "violation_type": candidate.violation_type,
                    "confidence": candidate.confidence,
                    "reason": decision["reasoning"],
                })

        ctx.validated_violations = validated
        ctx.insights["tera_rejected"] = rejected
        return ctx

    def _reason(self, violation: Violation, ctx: AgentContext) -> dict:
        rule = get_rule(violation.violation_type)
        if not rule:
            return self._reject(violation, "No regulation rule defined")

        # Emergency vehicle exemption
        if violation.bbox and ctx.scene_graph:
            for node in ctx.scene_graph.nodes.values():
                if iou(node.bbox, violation.bbox) > 0.4:
                    if is_emergency_entity({"label": node.label, **node.attributes}):
                        if "emergency_vehicle" in rule.exemptions:
                            return self._reject(violation, "Emergency vehicle exemption applied")

        min_conf = max(rule.min_confidence, self.global_min)
        if violation.confidence < min_conf:
            return self._reject(
                violation,
                f"Confidence {violation.confidence:.2f} below threshold {min_conf:.2f}",
            )

        # Contextual reasoning adjustments
        adjusted = violation.confidence
        reasoning_steps = [f"Rule {rule.regulation_code} matched"]

        graph_edges = violation.metadata.get("scene_graph_edges", [])
        if graph_edges:
            adjusted = min(0.98, adjusted + 0.03)
            reasoning_steps.append("Scene graph relationships support violation")

        spatial = ctx.spatial_data or {}
        if violation.violation_type == "wrong_side_driving" and spatial.get("expected_flow"):
            reasoning_steps.append(f"Expected flow: {spatial['expected_flow']}")

        if violation.violation_type == "triple_riding":
            count = violation.metadata.get("rider_count", 0)
            if count >= 4:
                adjusted = min(0.98, adjusted + 0.05)
                reasoning_steps.append(f"High rider count ({count}) increases certainty")

        # Ambiguity penalty
        if adjusted < min_conf:
            return self._reject(violation, "Adjusted confidence below threshold after reasoning")

        justification = (
            f"{rule.description}. Detected with {adjusted:.0%} confidence. "
            f"Cited under {rule.regulation_code} ({rule.penalty_category} violation)."
        )

        return {
            "approved": True,
            "adjusted_confidence": round(adjusted, 3),
            "regulation_code": rule.regulation_code,
            "justification": justification,
            "reasoning": "; ".join(reasoning_steps),
        }

    def _reject(self, violation: Violation, reason: str) -> dict:
        return {
            "approved": False,
            "adjusted_confidence": violation.confidence,
            "regulation_code": "",
            "justification": "",
            "reasoning": reason,
        }

    def _summary(self, ctx: AgentContext) -> dict:
        return {
            "approved": len(ctx.validated_violations),
            "rejected": len(ctx.insights.get("tera_rejected", [])),
        }
