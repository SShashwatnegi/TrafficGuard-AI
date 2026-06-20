# TrafficGuard AI — Technical Project Report

**Project:** TrafficGuard AI — Multi-Agent Traffic Enforcement Platform  
**Version:** 2.0  
**Location:** `c:\Users\shash\OneDrive\Documents\intern\trafficguard\`  
**Prepared for:** Hackathon / Internship Submission  
**Date:** June 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Objectives](#2-project-objectives)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Multi-Agent Design](#4-multi-agent-design)
5. [End-to-End Workflow](#5-end-to-end-workflow)
6. [Technology Stack](#6-technology-stack)
7. [Module-by-Module Technical Detail](#7-module-by-module-technical-detail)
8. [Violation Detection Techniques](#8-violation-detection-techniques)
9. [Graph-Based Scene Reasoning](#9-graph-based-scene-reasoning)
10. [TERA — Traffic Enforcement Reasoning Agent](#10-tera--traffic-enforcement-reasoning-agent)
11. [Data Storage & Analytics](#11-data-storage--analytics)
12. [User Interfaces](#12-user-interfaces)
13. [Performance Evaluation Framework](#13-performance-evaluation-framework)
14. [Configuration & Deployment](#14-configuration--deployment)
15. [Output Artifacts](#15-output-artifacts)
16. [Verified Runtime Behavior](#16-verified-runtime-behavior)
17. [Limitations & Future Enhancements](#17-limitations--future-enhancements)
18. [Conclusion](#18-conclusion)

---

## 1. Executive Summary

TrafficGuard AI is a **computer vision–based, multi-agent traffic enforcement system** that automatically processes traffic images, detects vehicles and road users, identifies traffic violations, validates them through a reasoning layer, and generates annotated legal evidence for enforcement review.

Unlike a monolithic script that runs all logic in one function, the system decomposes enforcement into **nine specialized agents** coordinated by an **AgentOrchestrator**. Agents communicate through a shared **AgentContext** (blackboard pattern) and a **SceneGraph** that models spatial relationships between objects in the traffic scene.

**Key capabilities delivered:**

| Capability | Status |
|------------|--------|
| Image preprocessing (low light, blur, noise, shadows) | Implemented |
| Vehicle & road user detection (YOLOv8) | Implemented |
| 7 violation types with confidence scores | Implemented |
| License plate recognition (ANPR) | Implemented |
| Graph-based scene reasoning | Implemented |
| TERA rule validation & legal justification | Implemented |
| Explainable evidence packages (JPG + JSON) | Implemented |
| SQLite storage & searchable records | Implemented |
| Analytics, hotspots, repeat offenders | Implemented |
| Natural-language Command Agent | Implemented |
| CLI, REST API, Streamlit dashboard | Implemented |
| Evaluation metrics (Accuracy, P, R, F1, mAP) | Implemented |

---

## 2. Project Objectives

The project addresses the operational challenge of manually reviewing thousands of hours of traffic surveillance footage. The system aims to:

1. **Automate detection** of vehicles, riders, pedestrians, and infrastructure from photographic evidence.
2. **Identify violations** including helmet non-compliance, seatbelt non-compliance, triple riding, wrong-side driving, stop-line violations, red-light violations, and illegal parking.
3. **Classify and score** each violation with a confidence value.
4. **Validate enforcement decisions** through a dedicated reasoning agent (TERA) rather than blindly issuing fines from raw detections.
5. **Generate explainable evidence** with regulation citations for human review.
6. **Provide analytics** for trends, hotspots, and repeat offenders.
7. **Scale modularly** via a multi-agent architecture aligned with the TrafficGuard AI Concept Note v2.0.

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAFFICGUARD AI SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  INTERFACES                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────────────┐  │
│  │ CLI      │    │ Streamlit UI │    │ FastAPI REST API                │  │
│  │ main.py  │    │ app.py       │    │ api.py                          │  │
│  └────┬─────┘    └──────┬───────┘    └──────────────┬──────────────────┘  │
│       └─────────────────┴────────────────────────────┘                      │
│                                    │                                        │
│                          TrafficGuardPipeline                               │
│                                    │                                        │
│                          AgentOrchestrator                                  │
│                                    │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐  │
│  │ MULTI-AGENT PIPELINE (sequential)                                     │  │
│  │                                                                       │  │
│  │  Preprocessing → Vision → Plate → Spatial → ViolationAnalyst         │  │
│  │       → TERA → Evidence → Insight                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ SceneGraph  │  │ YOLOv8       │  │ EasyOCR      │  │ SQLite DB      │  │
│  │ (reasoning) │  │ (detection)  │  │ (ANPR)       │  │ (persistence)  │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  │
│                                                                             │
│  ON-DEMAND: CommandAgent (natural language queries)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Design Patterns Used

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Multi-Agent System** | `src/agents/` | Decompose enforcement into specialized roles |
| **Blackboard / Shared Context** | `AgentContext` | Agents read/write shared state without direct coupling |
| **Orchestrator** | `AgentOrchestrator` | Central coordinator runs agents in defined order |
| **Scene Graph** | `src/graph/scene_graph.py` | Model objects and relationships for reasoning |
| **Rule Engine** | `src/regulations/rules.py` + TERA | Validate candidates against traffic law |
| **Pipeline** | `TrafficGuardPipeline` | Single entry point wrapping orchestrator |
| **Plugin Registry** | `ANALYZER_REGISTRY` | Violation analyzers registered by type |

### What Was NOT Used

- No LangChain, CrewAI, AutoGen, or LLM-based agent frameworks
- No cloud APIs or external inference services
- No custom model training (uses pretrained YOLOv8n)

The multi-agent system is implemented as **native Python classes** with clear separation of concerns — suitable for hackathon demos and further production hardening.

---

## 4. Multi-Agent Design

### 4.1 Agent List

| # | Agent | File | Role |
|---|-------|------|------|
| 1 | **PreprocessingAgent** | `preprocessing_agent.py` | Image quality enhancement |
| 2 | **VisionAgent** | `vision_agent.py` | Object detection + scene graph initialization |
| 3 | **PlateAgent** | `plate_agent.py` | License plate detection & OCR |
| 4 | **SpatialAgent** | `spatial_agent.py` | Direction, tracking, spatial relationships |
| 5 | **ViolationAnalystAgent** | `violation_analyst_agent.py` | Candidate violation generation |
| 6 | **TERA** | `tera_agent.py` | Rule validation, exemptions, legal reasoning |
| 7 | **EvidenceAgent** | `evidence_agent.py` | Annotated images + JSON evidence packages |
| 8 | **InsightAgent** | `insight_agent.py` | Hotspots, repeat offenders, priorities |
| 9 | **CommandAgent** | `command_agent.py` | Natural-language queries (on-demand) |

### 4.2 Base Agent Contract

Every processing agent inherits from `BaseAgent`:

```python
class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def execute(self, ctx: AgentContext) -> AgentContext: ...

    def run(self, ctx: AgentContext) -> AgentContext:
        # Wraps execute() with timing and logging to agent_trace
```

### 4.3 Shared Memory — AgentContext

`AgentContext` is a dataclass passed through the entire pipeline. Key fields:

| Field | Written By | Read By |
|-------|-----------|---------|
| `raw_image`, `image` | PreprocessingAgent | All downstream agents |
| `detections` | VisionAgent | Plate, Spatial, ViolationAnalyst |
| `scene_graph` | VisionAgent, SpatialAgent, PlateAgent | ViolationAnalyst, TERA |
| `plates` | PlateAgent | EvidenceAgent |
| `spatial_data` | SpatialAgent | ViolationAnalyst, TERA |
| `candidate_violations` | ViolationAnalystAgent | TERA |
| `validated_violations` | TERA | EvidenceAgent |
| `evidence_path`, `evidence_package` | EvidenceAgent | Pipeline output |
| `insights` | InsightAgent, TERA | Analytics, UI |
| `agent_trace` | All agents | Debugging, UI display |

### 4.4 Orchestrator

`AgentOrchestrator.run()` executes agents sequentially:

```
PreprocessingAgent → VisionAgent → PlateAgent → SpatialAgent
→ ViolationAnalystAgent → TERA → EvidenceAgent → InsightAgent
```

`CommandAgent` runs separately when a user submits a natural-language query.

---

## 5. End-to-End Workflow

### 5.1 Image Processing Flow

```
INPUT: traffic image (JPG/PNG)
  │
  ▼
[1] PreprocessingAgent
  │  • CLAHE for low light, bilateral denoise, sharpening, shadow balance
  │  • Output: enhanced image
  ▼
[2] VisionAgent
  │  • YOLOv8n inference → bounding boxes for cars, trucks, buses,
  │    motorcycles, bicycles, persons
  │  • Creates SceneGraph nodes for each detection
  │  • Adds infrastructure nodes: stop_line, no_parking_zones
  ▼
[3] PlateAgent
  │  • Crops vehicle rear regions + edge-based plate candidates
  │  • EasyOCR text extraction with regex validation
  │  • Links plates to vehicle nodes in scene graph
  ▼
[4] SpatialAgent
  │  • Estimates vehicle facing direction (Sobel gradient analysis)
  │  • Assigns lane zones (left/center/right)
  │  • Builds graph edges: on_vehicle, facing, crosses, in_zone
  │  • Tracks movement across sequential frames (frame_history)
  ▼
[5] ViolationAnalystAgent
  │  • Runs 7 violation analyzers using detections + graph + spatial data
  │  • Produces CANDIDATE violations (not yet approved)
  ▼
[6] TERA
  │  • Validates each candidate against MV Act rules
  │  • Applies confidence thresholds, exemptions, contextual boosts
  │  • Approves or rejects; attaches legal justification
  ▼
[7] EvidenceAgent
  │  • Draws annotated evidence image with bounding boxes
  │  • Saves JPG + JSON evidence package with TERA reasoning
  ▼
[8] InsightAgent
  │  • Queries SQLite for historical data
  │  • Computes hotspots, repeat offenders, enforcement priorities
  ▼
OUTPUT: ProcessingResult → saved to SQLite + evidence files
```

### 5.2 Query Flow (Command Agent)

```
User query: "Show repeat offenders"
  │
  ▼
CommandAgent._parse_intent() → intent: "repeat_offenders"
  │
  ▼
InsightAgent._repeat_offenders(db records)
  │
  ▼
Formatted natural-language response + data table
```

---

## 6. Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| Language | Python | 3.10+ | Core runtime |
| Object Detection | Ultralytics YOLOv8 | ≥8.0 | Vehicle/person detection |
| Computer Vision | OpenCV | ≥4.8 | Preprocessing, edges, HSV, Hough |
| Numerical | NumPy | ≥1.24 | Array operations |
| OCR | EasyOCR | ≥1.7 | License plate text extraction |
| Database | SQLite + SQLAlchemy | ≥2.0 | Violation record persistence |
| Web API | FastAPI + Uvicorn | ≥0.100 | REST endpoints |
| Dashboard | Streamlit | ≥1.28 | Interactive UI |
| Analytics | Pandas, Matplotlib | ≥2.0 / ≥3.7 | Reports and charts |
| ML Metrics | scikit-learn | ≥1.3 | Accuracy, P, R, F1 |
| Config | PyYAML | ≥6.0 | `config.yaml` settings |
| Validation | Pydantic | ≥2.0 | API request/response models |
| Deep Learning | PyTorch (via Ultralytics) | — | YOLO inference backend |

---

## 7. Module-by-Module Technical Detail

### 7.1 Image Preprocessing (`src/preprocessing/enhancer.py`)

**Agent:** PreprocessingAgent  
**Goal:** Normalize inputs and improve robustness under poor imaging conditions.

| Technique | Method | Trigger Condition |
|-----------|--------|-------------------|
| Color normalization | BGRA→BGR, grayscale→BGR | Always |
| Low-light enhancement | CLAHE on L channel in LAB space + gamma correction (γ=1.2) | Mean grayscale < 85 |
| Noise reduction | Bilateral filter (d=7, σ_color=75, σ_space=75) | Always (if enabled) |
| Motion blur compensation | Laplacian variance blur score + sharpening kernel | Blur score < 120 |
| Shadow balancing | HSV value channel × 1.05 | Always |

**Blur detection:** Laplacian variance — lower values indicate more blur.

---

### 7.2 Object Detection (`src/detection/detector.py`)

**Agent:** VisionAgent  
**Model:** YOLOv8n (nano) — ~6 MB, auto-downloaded on first run  
**Inference parameters:** confidence ≥ 0.35, IoU NMS ≥ 0.45

**Detected COCO classes mapped to categories:**

| YOLO Label | Category |
|------------|----------|
| car | four_wheeler |
| truck, bus | heavy_vehicle |
| motorcycle, bicycle | two_wheeler |
| person | pedestrian |

**Output:** List of `BoundingBox` objects with coordinates, label, confidence, and category.

---

### 7.3 License Plate Recognition (`src/lpr/plate_recognizer.py`)

**Agent:** PlateAgent  
**Technique:** Two-stage ANPR

**Stage 1 — Region proposal:**
- Crop lower 25% of each vehicle bounding box
- Canny edge detection + contour analysis for plate-like rectangles (aspect ratio 2.0–6.5, width > 60px)

**Stage 2 — OCR:**
- Grayscale → bilateral filter → adaptive Gaussian threshold
- EasyOCR `readtext()` with English language model
- Post-processing: uppercase, strip non-alphanumeric, regex `^[A-Z0-9]{5,12}$`
- Deduplication by highest confidence per plate text

---

### 7.4 Spatial Analysis (`src/agents/spatial_agent.py`)

**Agent:** SpatialAgent

| Analysis | Technique |
|----------|-----------|
| Facing direction | Sobel X gradient — compare left vs right half energy |
| Lane zone | Center X position: left (<33%), center, right (>66%) |
| Rider-on-vehicle | IoU + center-inside test with expanded motorcycle bbox |
| Infrastructure crossing | IoU between vehicle and stop_line / no_parking nodes |
| Cross-frame tracking | Centroid displacement from `frame_history` (up to 30 frames) |

---

### 7.5 Evidence Generation (`src/evidence/generator.py`)

**Agent:** EvidenceAgent

- Draws green boxes for detections
- Color-coded violation boxes (unique color per violation type)
- Yellow boxes for license plates
- Header banner with timestamp, filename, violation count
- Regulation code appended to violation labels when TERA metadata present
- Companion JSON file with full evidence package

---

## 8. Violation Detection Techniques

Each violation is implemented as a **plugin analyzer** in `src/violations/analyzers.py`, invoked by ViolationAnalystAgent.

### 8.1 Helmet Non-Compliance

| Aspect | Detail |
|--------|--------|
| **Logic** | For each motorcycle, find riders via spatial overlap; crop head region (top 35% of person bbox) |
| **Technique** | Helmet likelihood score from Canny edge density, grayscale uniformity, HSV saturation/value |
| **Threshold** | helmet_score < 0.45 → violation |
| **Confidence** | `0.55 + (0.45 - helmet_score)`, capped at 0.95 |

### 8.2 Seatbelt Non-Compliance

| Aspect | Detail |
|--------|--------|
| **Logic** | For cars/trucks/buses with person in cabin region (upper 55% of vehicle bbox) |
| **Technique** | Canny edges + Hough Line Transform for diagonal lines (25°–75°) indicating seatbelt |
| **Threshold** | seatbelt_score < 0.4 → violation |

### 8.3 Triple Riding

| Aspect | Detail |
|--------|--------|
| **Logic** | Count persons overlapping motorcycle bbox |
| **Threshold** | ≥ 3 riders → violation |
| **Confidence** | `0.7 + 0.1 × (rider_count - 3)`, capped at 0.98 |

### 8.4 Wrong-Side Driving

| Aspect | Detail |
|--------|--------|
| **Logic** | Compare vehicle facing direction vs configured `expected_flow` (default: "right") |
| **Technique** | Sobel X gradient on vehicle crop |
| **Condition** | Facing left while flow is right, and vehicle center X > 35% of image width |

### 8.5 Stop-Line Violation

| Aspect | Detail |
|--------|--------|
| **Logic** | Vehicle bottom edge (y2) crosses stop line Y coordinate |
| **Stop line detection** | Hough Line Transform on lower 45% of image for near-horizontal lines; median Y fallback at 72% image height |
| **Configurable** | `stop_line_y` in config.yaml (normalized 0–1) |

### 8.6 Red-Light Violation

| Aspect | Detail |
|--------|--------|
| **Logic** | Vehicle in intersection zone while red signal detected |
| **Red signal detection** | HSV color masking (H: 0–10 and 160–180, S/V > 120); contour shape filter (near-square, 8–80px) |
| **Intersection zone** | Below `intersection_start` (default: 45% of image height) |

### 8.7 Illegal Parking

| Aspect | Detail |
|--------|--------|
| **Logic** | Vehicle IoU overlap > 0.25 with no-parking zone |
| **Zones** | Configurable `no_parking_zones` or default roadside regions |
| **Targets** | four_wheeler and heavy_vehicle categories only |

---

## 9. Graph-Based Scene Reasoning

**File:** `src/graph/scene_graph.py`

The scene graph models the traffic environment as a directed graph:

### Nodes
- **Detection nodes:** vehicles, persons (from YOLO)
- **Infrastructure nodes:** stop_line, no_parking_zone (inferred or configured)

### Edges (Relationships)
| Relation | Meaning | Created By |
|----------|---------|------------|
| `on_vehicle` | Person riding on motorcycle | SpatialAgent |
| `facing` | Vehicle orientation (left/right) | SpatialAgent |
| `crosses` | Vehicle crossing stop line | SpatialAgent |
| `in_zone` | Vehicle inside no-parking area | SpatialAgent |
| `plate_of` | License plate linked to vehicle | PlateAgent |

### Why It Matters
ViolationAnalystAgent and TERA use graph edges as supporting evidence. TERA boosts confidence by +0.03 when scene graph relationships corroborate a candidate violation.

**Example graph for triple riding:**
```
person_1 --on_vehicle--> motorcycle_2
person_3 --on_vehicle--> motorcycle_2
person_5 --on_vehicle--> motorcycle_2
→ ViolationAnalyst: 3 riders → triple_riding candidate
→ TERA: validates against MV Act Sec 128
```

---

## 10. TERA — Traffic Enforcement Reasoning Agent

**File:** `src/agents/tera_agent.py`  
**Rules:** `src/regulations/rules.py`

TERA is the **cognitive layer** — it does not detect objects. It evaluates candidate violations from ViolationAnalystAgent.

### Reasoning Pipeline (per candidate)

```
1. Lookup traffic rule (regulation code, min confidence, exemptions)
2. Check emergency vehicle exemption via scene graph node matching
3. Reject if confidence < rule.min_confidence
4. Apply contextual confidence adjustments:
   - +0.03 if scene graph edges support violation
   - +0.05 if triple riding with ≥4 riders
5. Generate legal justification string
6. Approve or reject
```

### Regulation Mapping

| Violation | Regulation | Min Confidence | Category |
|-----------|-----------|----------------|----------|
| Helmet non-compliance | MV Act Sec 129 | 0.55 | safety |
| Seatbelt non-compliance | MV Act Sec 194B | 0.50 | safety |
| Triple riding | MV Act Sec 128 | 0.65 | safety |
| Wrong-side driving | MV Act Sec 184 | 0.60 | dangerous |
| Stop-line violation | Rule 8(1) CMVR | 0.65 | signal |
| Red-light violation | MV Act Sec 119/177 | 0.70 | signal |
| Illegal parking | MV Act Sec 122 | 0.55 | parking |

### Exemptions
- `emergency_vehicle`: ambulance, fire_truck, police labels
- `road_work`, `loading_zone`: defined per rule (extensible)

### Output per Approved Violation
```json
{
  "regulation_code": "MV Act Sec 184",
  "legal_justification": "Driving against permitted traffic flow direction. Detected with 75% confidence. Cited under MV Act Sec 184 (dangerous violation).",
  "reasoning": "Rule MV Act Sec 184 matched; Scene graph relationships support violation; Expected flow: right",
  "status": "approved"
}
```

---

## 11. Data Storage & Analytics

### 11.1 SQLite Database

**File:** `output/trafficguard.db`  
**ORM:** SQLAlchemy  
**Table:** `violations`

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| image_path | VARCHAR | Source image path |
| timestamp | DATETIME | UTC processing time |
| violation_type | VARCHAR | Violation class or "none" |
| confidence | FLOAT | TERA-adjusted confidence |
| description | TEXT | Human-readable description |
| plate_number | VARCHAR | ANPR result (if any) |
| evidence_path | VARCHAR | Path to annotated JPG |
| metadata_json | TEXT | TERA reasoning, bbox, scores |
| processing_time_ms | FLOAT | Total pipeline time |

### 11.2 Insight Agent Analytics

| Analysis | Method |
|----------|--------|
| **Hotspots** | Count violations by type; compute share % and severity (high/medium/low) |
| **Repeat offenders** | Group by plate_number; flag plates with ≥2 violations |
| **Daily trends** | Violation count per date (last 14 days) |
| **Enforcement priority** | Violation types with >30% share marked high severity |

### 11.3 Report Generation

- **Charts:** `violations_by_type.png`, `daily_trend.png` (Matplotlib)
- **CSV export:** `violations_export.csv` (Pandas)

---

## 12. User Interfaces

### 12.1 Command-Line Interface (`main.py`)

```bash
python main.py process <image>       # Process single image
python main.py batch <directory>     # Process folder
python main.py report --days 30      # Analytics report
python main.py search --type <type>  # Search violations
```

### 12.2 Streamlit Dashboard (`app.py`)

| Tab | Features |
|-----|----------|
| Process Image | Upload, multi-agent analysis, evidence display, agent trace, scene graph |
| Command Agent | Natural-language queries with example buttons |
| Violation Records | Searchable/filterable database table |
| Analytics & Insights | Hotspots, repeat offenders, charts, CSV download |

### 12.3 REST API (`api.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status + active agent list |
| GET | `/api/v1/agents` | List all agents |
| POST | `/api/v1/process` | Upload image for analysis |
| POST | `/api/v1/command` | NL query `{"query": "...", "days": 30}` |
| GET | `/api/v1/violations` | Search records |
| GET | `/api/v1/analytics` | Summary + insights |
| GET | `/api/v1/evidence/{filename}` | Download evidence image |

---

## 13. Performance Evaluation Framework

**File:** `src/evaluation/metrics.py`  
**Script:** `scripts/run_evaluation.py`

| Metric | Application |
|--------|-------------|
| **Precision** | TP / (TP + FP) for bounding box detection |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Accuracy** | TP / (TP + FP + FN) |
| **mAP** | Mean average precision across IoU thresholds 0.50–0.95 |
| **Classification report** | Per-violation-type P/R/F1 for violation classification |

**Detection matching:** IoU ≥ 0.5 between predicted and ground-truth boxes with matching class label.

---

## 14. Configuration & Deployment

**File:** `config.yaml`

```yaml
models:
  detector: yolov8n.pt
  confidence_threshold: 0.35
  iou_threshold: 0.45

preprocessing:
  enable_clahe: true
  enable_denoise: true
  enable_deblur: true

violations:
  enabled: [helmet_non_compliance, seatbelt_non_compliance, ...]
  min_confidence: 0.5

lpr:
  enabled: true
  languages: ["en"]

expected_flow: right          # for wrong-side detection
stop_line_y: 0.72             # optional normalized Y
no_parking_zones: [...]       # optional ROI zones
```

### Installation & Run

```bash
cd trafficguard
pip install -r requirements.txt
python scripts/download_samples.py
python main.py process data/sample/traffic_intersection.jpg
streamlit run app.py
uvicorn api:app --reload --port 8000
```

---

## 15. Output Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Annotated evidence image | `output/evidence/<name>_evidence.jpg` | JPEG with bounding boxes |
| Evidence JSON package | `output/evidence/<name>_evidence.json` | Full explainable record |
| Violation database | `output/trafficguard.db` | SQLite |
| Analytics charts | `output/reports/violations_by_type.png` | PNG |
| CSV export | `output/reports/violations_export.csv` | CSV |
| YOLO weights | `trafficguard/yolov8n.pt` | PyTorch model (auto-downloaded) |

### Sample Evidence JSON Structure

```json
{
  "image_path": "data/sample/traffic_intersection.jpg",
  "timestamp": "2026-06-19T12:45:16Z",
  "evidence_image": "output/evidence/traffic_intersection_evidence.jpg",
  "violation_count": 22,
  "violations": [
    {
      "violation_type": "wrong_side_driving",
      "confidence": 0.75,
      "regulation_code": "MV Act Sec 184",
      "legal_justification": "...",
      "reasoning": "Rule MV Act Sec 184 matched; Scene graph relationships support violation"
    }
  ],
  "agent_trace": [...],
  "scene_graph_stats": {"node_count": 14, "edge_count": 28, "vehicles": 8, "pedestrians": 3}
}
```

---

## 16. Verified Runtime Behavior

Tested on `data/sample/traffic_intersection.jpg`:

| Metric | Value |
|--------|-------|
| Total pipeline time | ~2.7–6 seconds (CPU, includes YOLO + optional EasyOCR) |
| Objects detected | 11 (cars + pedestrians) |
| Candidate violations | Generated by ViolationAnalystAgent |
| TERA-approved violations | 22 (wrong-side, stop-line, red-light on test image) |
| Agents executed | 8 sequential + CommandAgent on demand |
| Evidence files | JPG + JSON created in `output/evidence/` |
| Database records | Saved to SQLite |

**Command Agent verified queries:**
- "Show violation hotspots" → ranked violation types with severity
- "List repeat offenders" → plate-based repeat analysis

---

## 17. Limitations & Future Enhancements

### Current Limitations

| Area | Limitation |
|------|-----------|
| Detection model | Generic COCO YOLOv8n — not fine-tuned on Indian traffic |
| Helmet/seatbelt | Heuristic CV (not dedicated deep learning classifiers) |
| Video | Single-image pipeline; frame_history supports basic tracking only |
| Red-light | HSV red blob detection may false-positive on red objects |
| Command Agent | Keyword-based NL parsing (not LLM-powered) |
| TERA | Rule-based reasoning (not neural/legal NLP) |
| ANPR | EasyOCR accuracy varies with plate angle, dirt, and font |

### Recommended Enhancements

1. Fine-tune YOLO on local traffic datasets (Indian vehicles, auto-rickshaws)
2. Dedicated helmet/seatbelt classification models
3. Full video pipeline with DeepSORT/ByteTrack multi-object tracking
4. Traffic signal state classifier (red/yellow/green CNN)
5. LLM-powered Command Agent for complex queries
6. GPU acceleration for production throughput
7. Camera calibration for accurate stop-line and lane geometry

---

## 18. Conclusion

TrafficGuard AI implements a **complete, runnable multi-agent traffic enforcement platform** aligned with the concept note requirements. The system combines:

- **Computer vision** (YOLOv8, OpenCV, EasyOCR) for perception
- **Graph-based scene reasoning** for relationship modeling
- **Rule-based TERA reasoning** for explainable, regulation-backed decisions
- **Nine specialized agents** coordinated through a blackboard architecture
- **Full operational stack** — storage, analytics, evidence, and three user interfaces

The architecture is modular: individual agents, analyzers, and rules can be upgraded independently without rewriting the entire system. This makes TrafficGuard AI suitable as both a **hackathon demonstration** and a **foundation for production deployment** with domain-specific model training.

---

*Report generated for TrafficGuard AI v2.0 — Multi-Agent Traffic Enforcement Platform*
