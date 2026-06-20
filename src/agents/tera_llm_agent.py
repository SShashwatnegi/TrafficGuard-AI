from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from src.agents.base import AgentContext, BaseAgent
from src.agents.tera_tools import TeraLlmSession, build_tera_tools
from src.models.schemas import Violation
from src.regulations.rules import get_rule


class TeraLlmAgent(BaseAgent):
    """
    LLM-assisted TERA (Phase 2) — LangChain tool-calling agent acting as a human surveillance reviewer.

    Uses upstream agent outputs as tools. Makes the FINAL decision on whether each
    candidate violation is a real violation or a false positive.
    """

    name = "TERA-LLM"

    SYSTEM_PROMPT = """You are TERA (Traffic Enforcement Reasoning Agent), a senior traffic enforcement
officer reviewing AI-flagged violations from CCTV imagery. You replace manual human surveillance review.

YOUR JOB:
1. Review EVERY candidate violation flagged by ViolationAnalystAgent.
2. Use the provided tools and visually inspect the image to gather evidence.
3. Determine if each candidate is a REAL violation or a FALSE POSITIVE.
4. EXTREME CAUTION AGAINST FALSE POSITIVES: The upstream AI frequently hallucinates violations. Reject any candidate if:
   - The image is blurry, objects are too small, or the evidence is ambiguous.
   - It's a static/parked vehicle mistakenly flagged as moving.
   - You cannot clearly and definitively visually confirm the violation yourself.
   - The bounding box coordinates do not clearly align with the suspected violation.
5. Be highly conservative — only approve violations you would confidently cite in legal enforcement.

WORKFLOW:
- Call get_candidate_violations to see all flagged items.
- For each candidate, use get_violation_region_summary and other tools as needed.
- Check get_rule_based_tera_assessment for regulatory pre-check (informational only).
- Call get_traffic_regulation for the applicable law.
- When done, call submit_enforcement_decisions with a JSON array of ALL candidates.

DECISION FORMAT for submit_enforcement_decisions:
[
  {
    "violation_index": 0,
    "approved": true,
    "is_real_violation": true,
    "confidence": 0.82,
    "reasoning": "Why this is or is not a real violation",
    "legal_justification": "Enforcement justification if approved",
    "regulation_code": "MV Act Sec 129"
  }
]

You MUST call submit_enforcement_decisions before finishing."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self.fallback_to_rules = cfg.get("fallback_to_rules_on_error", True)
        self.max_iterations = cfg.get("max_tool_iterations", 12)
        self._llm_config = cfg

    def execute(self, ctx: AgentContext) -> AgentContext:
        rule_validated = list(ctx.validated_violations)
        rule_rejected = list(ctx.insights.get("tera_rejected", []))

        ctx.insights["tera_rules"] = {
            "validated_count": len(rule_validated),
            "rejected_count": len(rule_rejected),
            "validated": [v.to_dict() for v in rule_validated],
            "rejected": rule_rejected,
        }

        if not self.enabled:
            ctx.insights["tera_llm"] = {"status": "disabled", "used_rule_fallback": True}
            return ctx

        if not ctx.candidate_violations:
            ctx.validated_violations = []
            ctx.insights["tera_llm"] = {"status": "no_candidates", "approved": 0}
            return ctx

        llm = self._create_llm()
        if llm is None:
            ctx.insights["tera_llm"] = {
                "status": "skipped",
                "reason": "No LLM API key or provider configured",
                "used_rule_fallback": True,
            }
            return ctx

        session = TeraLlmSession(
            ctx=ctx,
            rule_validated=rule_validated,
            rule_rejected=rule_rejected,
        )

        try:
            validated = self._run_langgraph_review(session, llm)
            ctx.validated_violations = validated
            ctx.insights["tera_llm"] = {
                "status": "completed",
                "approved": len(validated),
                "rejected": len(ctx.candidate_violations) - len(validated),
                "decisions": session.submitted_decisions,
            }
            ctx.insights["tera_rejected"] = [
                *rule_rejected,
                *self._llm_rejected(ctx.candidate_violations, session.submitted_decisions),
            ]
        except Exception as exc:
            ctx.insights["tera_llm"] = {
                "status": "error",
                "error": str(exc),
                "used_rule_fallback": self.fallback_to_rules,
            }
            if not self.fallback_to_rules:
                raise
            ctx.validated_violations = rule_validated

        return ctx

    def _run_langgraph_review(self, session: TeraLlmSession, llm) -> list[Violation]:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        import base64

        tools = build_tera_tools(session)
        tool_map = {t.name: t for t in tools}
        llm_with_tools = llm.bind_tools(tools)

        n = len(session.ctx.candidate_violations)
        
        image_content = []
        if os.path.exists(session.ctx.image_path):
            with open(session.ctx.image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode("utf-8")
                ext = session.ctx.image_path.split('.')[-1].lower()
                mime_type = f"image/{ext}" if ext in ["jpeg", "jpg", "png", "webp"] else "image/jpeg"
                image_content = [{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}]

        text_content = {
            "type": "text",
            "text": (
                f"Review {n} candidate violation(s) from traffic image: {session.ctx.image_path}\n"
                f"Use the provided image and tools to inspect the evidence, then call submit_enforcement_decisions "
                f"with your final verdict for ALL {n} candidate(s) (indices 0 to {n - 1}).\n"
                f"Note: TERA rule-based agent has pre-validated {len(session.rule_validated)} of these candidates."
            )
        }

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=[text_content] + image_content),
        ]

        for _ in range(self.max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                break

            for call in response.tool_calls:
                tool = tool_map.get(call["name"])
                if tool is None:
                    result = f"Unknown tool: {call['name']}"
                else:
                    result = tool.invoke(call.get("args") or {})
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

            if session.submitted_decisions:
                break

        if not session.submitted_decisions:
            return self._fallback_parse_from_rule(session)

        return self._apply_decisions(session)

    def _apply_decisions(self, session: TeraLlmSession) -> list[Violation]:
        validated: list[Violation] = []
        candidates = session.ctx.candidate_violations

        for decision in session.submitted_decisions:
            idx = decision.get("violation_index")
            if idx is None or idx < 0 or idx >= len(candidates):
                continue
            if not decision.get("approved", False):
                continue
            if decision.get("is_real_violation") is False:
                continue

            v = deepcopy(candidates[idx])
            v.confidence = float(decision.get("confidence", v.confidence))
            rule = get_rule(v.violation_type)
            regulation_code = decision.get("regulation_code") or (rule.regulation_code if rule else "")

            v.metadata["tera"] = {
                "regulation_code": regulation_code,
                "legal_justification": decision.get("legal_justification", ""),
                "reasoning": decision.get("reasoning", ""),
                "status": "approved",
                "reviewer": "TERA-LLM",
            }
            v.metadata["tera_rules"] = _rule_metadata_for(session, v)
            validated.append(v)

        return validated

    def _fallback_parse_from_rule(self, session: TeraLlmSession) -> list[Violation]:
        """If LLM did not call submit tool, fall back to rule-based approvals."""
        for v in session.rule_validated:
            v.metadata.setdefault("tera", {})["reviewer"] = "TERA-Rules (LLM did not submit)"
        return session.rule_validated

    def _llm_rejected(self, candidates: list[Violation], decisions: list[dict]) -> list[dict]:
        rejected = []
        decision_map = {d.get("violation_index"): d for d in decisions if "violation_index" in d}
        for i, c in enumerate(candidates):
            d = decision_map.get(i)
            if d and not d.get("approved", False):
                rejected.append({
                    "violation_type": c.violation_type,
                    "confidence": c.confidence,
                    "reason": d.get("reasoning", "Rejected by TERA-LLM"),
                    "reviewer": "TERA-LLM",
                })
            elif d is None:
                rejected.append({
                    "violation_type": c.violation_type,
                    "confidence": c.confidence,
                    "reason": "No LLM decision submitted for this candidate",
                    "reviewer": "TERA-LLM",
                })
        return rejected

    def _create_llm(self):
        provider = self._llm_config.get("provider", "openai").lower()
        model = self._llm_config.get("model", "gpt-4o-mini")
        temperature = self._llm_config.get("temperature", 0.0)

        if provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model,
                base_url=self._llm_config.get("base_url", "http://localhost:11434"),
                temperature=temperature,
            )

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key_env = self._llm_config.get("api_key_env", "GEMINI_API_KEY")
            api_key = os.environ.get(api_key_env) or self._llm_config.get("api_key")
            if not api_key:
                return None
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
            )

        api_key_env = self._llm_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env) or self._llm_config.get("api_key")
        if not api_key:
            return None

        from langchain_openai import ChatOpenAI
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
        }
        if self._llm_config.get("base_url"):
            kwargs["base_url"] = self._llm_config["base_url"]
        return ChatOpenAI(**kwargs)

    def _summary(self, ctx: AgentContext) -> dict:
        tera_llm = ctx.insights.get("tera_llm", {})
        return {
            "status": tera_llm.get("status", "unknown"),
            "approved": len(ctx.validated_violations),
            "llm_decisions": len(tera_llm.get("decisions", [])),
        }


def _rule_metadata_for(session: TeraLlmSession, violation: Violation) -> dict:
    for rv in session.rule_validated:
        if rv.violation_type == violation.violation_type and rv.metadata.get("tera"):
            return rv.metadata["tera"]
    return {}
