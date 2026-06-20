from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from src.agents.command_agent import CommandAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.insight_agent import InsightAgent
from src.agents.orchestrator import AgentOrchestrator
from src.agents.plate_agent import PlateAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.spatial_agent import SpatialAgent
from src.agents.tera_agent import TeraAgent
from src.agents.tera_llm_agent import TeraLlmAgent
from src.agents.violation_analyst_agent import ViolationAnalystAgent
from src.agents.vision_agent import VisionAgent
from src.analytics.reporter import AnalyticsReporter
from src.config import ensure_dirs, load_config
from src.detection.detector import ObjectDetector
from src.evidence.generator import EvidenceGenerator
from src.lpr.plate_recognizer import PlateRecognizer
from src.models.schemas import ProcessingResult
from src.preprocessing.enhancer import ImagePreprocessor
from src.storage.database import Database


class TrafficGuardPipeline:
    """Multi-agent traffic enforcement pipeline orchestrated by AgentOrchestrator."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config = load_config(config_path)
        ensure_dirs(self.config)
        paths = self.config["paths"]
        models = self.config["models"]
        prep = self.config["preprocessing"]
        viol = self.config["violations"]
        lpr_cfg = self.config.get("lpr", {})
        tera_llm_cfg = self.config.get("tera_llm", {})

        preprocessor = ImagePreprocessor(
            enable_clahe=prep.get("enable_clahe", True),
            enable_denoise=prep.get("enable_denoise", True),
            enable_deblur=prep.get("enable_deblur", True),
            clahe_clip_limit=prep.get("clahe_clip_limit", 2.5),
            clahe_tile_size=prep.get("clahe_tile_size", 8),
        )
        detector = ObjectDetector(
            model_name=models.get("detector", "yolov8n.pt"),
            confidence=models.get("confidence_threshold", 0.35),
            iou=models.get("iou_threshold", 0.45),
        )
        lpr_enabled = lpr_cfg.get("enabled", True)
        plate_recognizer = PlateRecognizer(
            languages=lpr_cfg.get("languages", ["en"]),
            min_confidence=lpr_cfg.get("min_plate_confidence", 0.4),
        ) if lpr_enabled else None

        self.db = Database(paths["database"])
        self.reporter = AnalyticsReporter(self.db, paths["reports_dir"])
        insight = InsightAgent(self.db)

        self.orchestrator = AgentOrchestrator(
            preprocessing=PreprocessingAgent(preprocessor),
            vision=VisionAgent(detector),
            plate=PlateAgent(plate_recognizer),
            spatial=SpatialAgent(),
            violation_analyst=ViolationAnalystAgent(
                viol.get("enabled", []),
                min_confidence=viol.get("min_confidence", 0.5),
            ),
            tera=TeraAgent(global_min_confidence=viol.get("min_confidence", 0.5)),
            tera_llm=TeraLlmAgent(config=tera_llm_cfg),
            evidence=EvidenceAgent(EvidenceGenerator(paths["evidence_dir"])),
            insight=insight,
            command=CommandAgent(self.db, insight),
        )
        self.pipeline_config = {
            k: v for k, v in self.config.items()
            if k not in ("paths", "models", "preprocessing", "violations", "lpr", "analytics", "agents")
        }
        self._frame_history: list[dict[str, Any]] = []

    def process_image(self, image_path: str | Path, save_to_db: bool = True, is_red_light: bool = True) -> ProcessingResult:
        image_path = Path(image_path)
        t0 = time.perf_counter()

        raw = cv2.imread(str(image_path))
        if raw is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        run_config = self.pipeline_config.copy()
        run_config["is_red_light"] = is_red_light

        ctx = self.orchestrator.run(
            raw_image=raw,
            image_path=str(image_path),
            config=run_config,
            frame_history=self._frame_history,
        )

        # Update frame history for spatial tracking across sequential frames
        if ctx.spatial_data:
            track_map = {t["node_id"]: t for t in ctx.spatial_data.get("tracks", [])}
            self._frame_history.append({"tracks": track_map})
            if len(self._frame_history) > 30:
                self._frame_history.pop(0)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result = ProcessingResult(
            image_path=str(image_path),
            timestamp=ctx.timestamp,
            detections=ctx.detections,
            violations=ctx.validated_violations,
            candidate_violations=ctx.candidate_violations,
            plates=ctx.plates,
            evidence_path=ctx.evidence_path,
            evidence_package=ctx.evidence_package,
            processing_time_ms=round(elapsed_ms, 2),
            preprocessing_applied=ctx.preprocessing_steps,
            scene_graph=ctx.scene_graph.to_dict() if ctx.scene_graph else {},
            spatial_data=ctx.spatial_data,
            agent_trace=ctx.agent_trace,
            insights=ctx.insights,
        )

        if save_to_db:
            self.db.save_result(result)

        return result

    def process_batch(self, image_dir: str | Path) -> list[ProcessingResult]:
        image_dir = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        results = []
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() in extensions:
                results.append(self.process_image(path))
        return results

    def query(self, text: str, days: int | None = None) -> dict[str, Any]:
        days = days or self.config.get("analytics", {}).get("default_days", 30)
        return self.orchestrator.query(text, days=days)

    def get_analytics(self, days: int | None = None) -> dict[str, Any]:
        days = days or self.config.get("analytics", {}).get("default_days", 30)
        summary = self.reporter.summary(days)
        charts = self.reporter.generate_charts(days)
        insight_agent = InsightAgent(self.db)
        insights = insight_agent.analyze(self.db.all_records(), days)
        return {"summary": summary, "charts": charts, "insights": insights}

    def list_agents(self) -> list[str]:
        return self.orchestrator.agent_names()
