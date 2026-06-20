from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrafficRule:
    violation_type: str
    regulation_code: str
    description: str
    min_confidence: float
    penalty_category: str
    exemptions: list[str]


TRAFFIC_RULES: dict[str, TrafficRule] = {
    "helmet_non_compliance": TrafficRule(
        violation_type="helmet_non_compliance",
        regulation_code="MV Act Sec 129",
        description="Riding a two-wheeler without protective headgear",
        min_confidence=0.55,
        penalty_category="safety",
        exemptions=["emergency_vehicle"],
    ),
    "seatbelt_non_compliance": TrafficRule(
        violation_type="seatbelt_non_compliance",
        regulation_code="MV Act Sec 194B",
        description="Driving without seatbelt fastened",
        min_confidence=0.50,
        penalty_category="safety",
        exemptions=["emergency_vehicle"],
    ),
    "triple_riding": TrafficRule(
        violation_type="triple_riding",
        regulation_code="MV Act Sec 128",
        description="More than two persons on a two-wheeler",
        min_confidence=0.65,
        penalty_category="safety",
        exemptions=[],
    ),
    "wrong_side_driving": TrafficRule(
        violation_type="wrong_side_driving",
        regulation_code="MV Act Sec 184",
        description="Driving against permitted traffic flow direction",
        min_confidence=0.60,
        penalty_category="dangerous",
        exemptions=["emergency_vehicle", "road_work"],
    ),
    "stop_line_violation": TrafficRule(
        violation_type="stop_line_violation",
        regulation_code="Rule 8(1) CMVR",
        description="Crossing stop line when required to halt",
        min_confidence=0.65,
        penalty_category="signal",
        exemptions=["emergency_vehicle"],
    ),
    "red_light_violation": TrafficRule(
        violation_type="red_light_violation",
        regulation_code="MV Act Sec 119/177",
        description="Entering intersection against red signal",
        min_confidence=0.70,
        penalty_category="signal",
        exemptions=["emergency_vehicle"],
    ),
    "illegal_parking": TrafficRule(
        violation_type="illegal_parking",
        regulation_code="MV Act Sec 122",
        description="Parking in restricted or no-parking zone",
        min_confidence=0.55,
        penalty_category="parking",
        exemptions=["emergency_vehicle", "loading_zone"],
    ),
}

EMERGENCY_LABELS = {"ambulance", "fire_truck", "police", "emergency"}


def get_rule(violation_type: str) -> TrafficRule | None:
    return TRAFFIC_RULES.get(violation_type)


def is_emergency_entity(node_attrs: dict[str, Any]) -> bool:
    label = str(node_attrs.get("label", "")).lower()
    if label in EMERGENCY_LABELS:
        return True
    if node_attrs.get("emergency", False):
        return True
    return False
