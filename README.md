# TrafficGuard AI

> **Full technical report:** See [PROJECT_REPORT.md](PROJECT_REPORT.md) for complete architecture, techniques, workflow, and evaluation details.

A **Multi-Agent Traffic Enforcement Platform** for explainable and intelligent traffic violation detection — aligned with the TrafficGuard AI Concept Note v2.0.

## Multi-Agent Architecture

```
Image → PreprocessingAgent → VisionAgent → PlateAgent → SpatialAgent
     → ViolationAnalystAgent → TERA (rules) → TERA-LLM (LangChain) → EvidenceAgent → InsightAgent
                                                              ↓
                                                    CommandAgent (on-demand NL queries)
```

| Agent | Role |
|-------|------|
| **PreprocessingAgent** | Image enhancement (CLAHE, denoise, deblur, shadow balance) |
| **VisionAgent** | YOLO detection of vehicles, road users, infrastructure nodes |
| **PlateAgent** | ANPR — license plate detection and OCR |
| **SpatialAgent** | Movement, direction, tracking, scene graph relationships |
| **ViolationAnalystAgent** | Candidate violations from graph-based scene reasoning |
| **TERA** | Rule-based pre-check (MV Act thresholds, exemptions) |
| **TERA-LLM** | LLM tool-calling agent — human-like final review using upstream agents as tools |
| **EvidenceAgent** | Annotated evidence images + JSON evidence packages |
| **InsightAgent** | Hotspots, repeat offenders, enforcement priorities |
| **CommandAgent** | Natural-language queries for enforcement officers |

## Features

- Graph-based scene representation (`SceneGraph` with nodes & edges)
- 7 violation types with confidence scoring
- TERA filters candidates using MV Act regulation rules
- Explainable evidence with legal citations
- SQLite storage, analytics, CSV export
- Performance evaluation (Accuracy, Precision, Recall, F1, mAP)

## Quick Start

```bash
cd trafficguard
pip install -r requirements.txt
python scripts/download_samples.py
python main.py process data/sample/traffic_intersection.jpg
streamlit run app.py
uvicorn api:app --reload --port 8000
```

## CLI

```bash
python main.py process path/to/image.jpg
python main.py batch data/sample
python main.py report --days 30
python main.py search --type helmet_non_compliance
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + active agents |
| GET | `/api/v1/agents` | List all agents |
| POST | `/api/v1/process` | Upload image for multi-agent analysis |
| POST | `/api/v1/command` | Natural-language query (`{"query": "show repeat offenders"}`) |
| GET | `/api/v1/violations` | Search violation records |
| GET | `/api/v1/analytics` | Summary + Insight Agent data |
| GET | `/api/v1/evidence/{filename}` | Download evidence image |

## Project Structure

```
trafficguard/
├── main.py / app.py / api.py
├── config.yaml
├── src/
│   ├── agents/          # All 9 agents + orchestrator
│   ├── graph/           # Scene graph (graph-based reasoning)
│   ├── regulations/     # TERA traffic rules (MV Act)
│   ├── preprocessing/   # Image enhancement
│   ├── detection/       # YOLO backend
│   ├── violations/      # Violation analyzers
│   ├── lpr/             # Plate OCR backend
│   ├── evidence/        # Evidence rendering
│   ├── storage/         # SQLite
│   ├── analytics/       # Reports
│   ├── evaluation/      # Metrics
│   └── pipeline/        # Orchestrator integration
└── output/              # Evidence, reports, database
```

## Command Agent Examples

- "Show violation hotspots"
- "List repeat offenders"
- "Summary report"
- "Show helmet violations"
- "Search plate ABC1234"
- "Recent violations last 7 days"

## TERA-LLM (LLM-Assisted Review)

Phase 1 **TERA** runs rule-based regulatory pre-checks. Phase 2 **TERA-LLM** uses a **LangChain LLM tool-calling loop** (ReAct pattern) to act as a human surveillance reviewer.

The LLM can call tools backed by upstream agents:

| Tool | Source Agent |
|------|--------------|
| `get_vision_detections` | VisionAgent |
| `get_scene_graph` | VisionAgent + SpatialAgent |
| `get_spatial_analysis` | SpatialAgent |
| `get_license_plates` | PlateAgent |
| `get_preprocessing_info` | PreprocessingAgent |
| `get_traffic_regulation` | Rule engine |
| `get_rule_based_tera_assessment` | TERA (Phase 1) |
| `get_candidate_violations` | ViolationAnalystAgent |
| `submit_enforcement_decisions` | Final LLM verdict |

Configure in `config.yaml`:

```yaml
tera_llm:
  enabled: true
  provider: openai    # or ollama
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  fallback_to_rules_on_error: true
```

Set `OPENAI_API_KEY` in your environment (see `.env.example`). If LLM is unavailable, the system falls back to rule-based TERA.

## Configuration

Edit `config.yaml` for detection thresholds, enabled violations, ROI zones, and expected traffic flow direction.

## Evaluation

```bash
python scripts/run_evaluation.py data/evaluation/ground_truth_sample.json data/evaluation/predictions_sample.json
```

## Notes

- First run downloads YOLOv8 weights (~6 MB).
- TERA applies per-violation confidence thresholds and emergency vehicle exemptions.
- Evidence packages saved as `.json` alongside annotated `.jpg` images.
- For production, fine-tune specialized models on local traffic datasets.
