from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

import cv2
from langchain_core.tools import tool

from src.agents.base import AgentContext
from src.models.schemas import Violation
from src.regulations.rules import TRAFFIC_RULES, get_rule


@dataclass
class TeraLlmSession:
    """Mutable session state for the LangGraph TERA-LLM agent."""

    ctx: AgentContext
    rule_validated: list[Violation] = field(default_factory=list)
    rule_rejected: list[dict] = field(default_factory=list)
    submitted_decisions: list[dict[str, Any]] = field(default_factory=list)


def build_tera_tools(session: TeraLlmSession) -> list:
    """Build LangChain tools that expose upstream agent outputs to TERA-LLM."""
    ctx = session.ctx

    @tool
    def get_vision_detections() -> str:
        """VisionAgent output: all detected vehicles, persons, and road users with bounding boxes and confidence."""
        data = [
            {
                "label": d.label,
                "category": d.category,
                "confidence": round(d.confidence, 3),
                "bbox": [round(d.x1), round(d.y1), round(d.x2), round(d.y2)],
            }
            for d in ctx.detections
        ]
        return json.dumps({"count": len(data), "detections": data}, indent=2)

    @tool
    def get_scene_graph() -> str:
        """Scene graph from VisionAgent + SpatialAgent: objects and relationships (on_vehicle, crosses, facing, in_zone)."""
        if not ctx.scene_graph:
            return json.dumps({"error": "Scene graph not available"})
        return json.dumps(ctx.scene_graph.to_dict(), indent=2)

    @tool
    def get_spatial_analysis() -> str:
        """SpatialAgent output: vehicle facing direction, lane zones, movement tracks, spatial relationships."""
        return json.dumps(ctx.spatial_data or {"note": "No spatial data"}, indent=2)

    @tool
    def get_license_plates() -> str:
        """PlateAgent output: ANPR results linked to vehicles."""
        plates = [
            {"text": p.text, "confidence": p.confidence, "bbox": p.bbox.to_dict() if p.bbox else None}
            for p in ctx.plates
        ]
        return json.dumps({"count": len(plates), "plates": plates}, indent=2)

    @tool
    def get_preprocessing_info() -> str:
        """PreprocessingAgent output: image enhancement steps applied before detection."""
        return json.dumps({
            "steps": ctx.preprocessing_steps,
            "image_path": ctx.image_path,
            "timestamp": ctx.timestamp,
        }, indent=2)

    @tool
    def get_traffic_regulation(violation_type: str) -> str:
        """Look up MV Act regulation rule for a violation type (used by rule-based TERA)."""
        rule = get_rule(violation_type)
        if not rule:
            return json.dumps({"error": f"No rule for {violation_type}"})
        return json.dumps({
            "violation_type": rule.violation_type,
            "regulation_code": rule.regulation_code,
            "description": rule.description,
            "min_confidence": rule.min_confidence,
            "penalty_category": rule.penalty_category,
            "exemptions": rule.exemptions,
        }, indent=2)

    @tool
    def get_all_traffic_regulations() -> str:
        """All traffic regulations in the rule engine."""
        return json.dumps({
            k: {
                "regulation_code": v.regulation_code,
                "description": v.description,
                "min_confidence": v.min_confidence,
                "exemptions": v.exemptions,
            }
            for k, v in TRAFFIC_RULES.items()
        }, indent=2)

    @tool
    def get_rule_based_tera_assessment() -> str:
        """Rule-based TERA (Phase 1) decisions: which candidates passed/failed regulatory pre-check."""
        return json.dumps({
            "rule_approved": [v.to_dict() for v in session.rule_validated],
            "rule_rejected": session.rule_rejected,
            "note": "Rule TERA is a pre-check. You make the final human-like surveillance decision.",
        }, indent=2)

    @tool
    def get_candidate_violations() -> str:
        """ViolationAnalystAgent output: all AI-flagged candidate violations requiring your review."""
        candidates = [
            {
                "index": i,
                "violation_type": v.violation_type,
                "confidence": v.confidence,
                "description": v.description,
                "metadata": v.metadata,
                "bbox": v.bbox.to_dict() if v.bbox else None,
            }
            for i, v in enumerate(ctx.candidate_violations)
        ]
        return json.dumps({"count": len(candidates), "candidates": candidates}, indent=2)

    @tool
    def get_violation_region_summary(violation_index: int) -> str:
        """Detailed context for one candidate: bbox, nearby detections, graph edges, and scene description."""
        if violation_index < 0 or violation_index >= len(ctx.candidate_violations):
            return json.dumps({"error": f"Invalid index {violation_index}"})

        v = ctx.candidate_violations[violation_index]
        nearby = []
        if v.bbox:
            for d in ctx.detections:
                if _boxes_overlap(v.bbox.to_dict(), d.to_dict()):
                    nearby.append({"label": d.label, "confidence": d.confidence})

        edges = v.metadata.get("scene_graph_edges", [])
        h, w = ctx.image.shape[:2] if ctx.image is not None else (0, 0)

        return json.dumps({
            "violation_index": violation_index,
            "violation_type": v.violation_type,
            "confidence": v.confidence,
            "description": v.description,
            "metadata": v.metadata,
            "bbox": v.bbox.to_dict() if v.bbox else None,
            "image_size": {"width": w, "height": h},
            "nearby_detections": nearby,
            "scene_graph_edges": edges,
        }, indent=2)

    @tool
    def get_violation_crop_base64(violation_index: int, padding: int = 40) -> str:
        """Base64 JPEG of the violation region (for visual inspection). Returns data URI or error."""
        if violation_index < 0 or violation_index >= len(ctx.candidate_violations):
            return json.dumps({"error": f"Invalid index {violation_index}"})
        v = ctx.candidate_violations[violation_index]
        if not v.bbox or ctx.image is None:
            return json.dumps({"error": "No bbox or image"})

        h, w = ctx.image.shape[:2]
        x1 = int(max(0, v.bbox.x1 - padding))
        y1 = int(max(0, v.bbox.y1 - padding))
        x2 = int(min(w, v.bbox.x2 + padding))
        y2 = int(min(h, v.bbox.y2 + padding))
        crop = ctx.image[y1:y2, x1:x2]
        if crop.size == 0:
            return json.dumps({"error": "Empty crop"})

        _, buf = cv2.imencode(".jpg", crop)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return json.dumps({
            "violation_index": violation_index,
            "crop_size": {"width": x2 - x1, "height": y2 - y1},
            "base64_jpeg": b64,
            "note": "Decode base64 JPEG to inspect violation region visually.",
        })

    @tool
    def submit_enforcement_decisions(decisions_json: str) -> str:
        """
        Submit your FINAL enforcement decisions for all reviewed candidates.
        Required JSON array format:
        [{"violation_index": 0, "approved": true, "confidence": 0.85,
          "is_real_violation": true, "reasoning": "...", "legal_justification": "...",
          "regulation_code": "MV Act Sec 129"}]
        Set approved=false for false positives. You MUST call this tool when finished reviewing.
        """
        try:
            decisions = json.loads(decisions_json)
            if not isinstance(decisions, list):
                return "Error: decisions_json must be a JSON array"
            session.submitted_decisions = decisions
            return f"Successfully recorded {len(decisions)} enforcement decision(s)."
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON — {exc}"

    return [
        get_vision_detections,
        get_scene_graph,
        get_spatial_analysis,
        get_license_plates,
        get_preprocessing_info,
        get_traffic_regulation,
        get_all_traffic_regulations,
        get_rule_based_tera_assessment,
        get_candidate_violations,
        get_violation_region_summary,
        get_violation_crop_base64,
        submit_enforcement_decisions,
    ]


def _boxes_overlap(a: dict, b: dict) -> bool:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    return x2 > x1 and y2 > y1
