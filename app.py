"""Streamlit dashboard for TrafficGuard AI."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.processor import TrafficGuardPipeline

st.set_page_config(page_title="TrafficGuard AI", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    .main-header { color: #1B3A6B; font-size: 2rem; font-weight: 700; }
    .sub-header { color: #2E75B6; }
    .agent-badge { background:#1B3A6B; color:white; padding:4px 10px; border-radius:12px;
                   font-size:0.8rem; margin:2px; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">TrafficGuard AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-Agent Traffic Enforcement Platform — '
    'Explainable & Intelligent Violation Detection</p>',
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    with st.spinner("Loading multi-agent system (YOLO weights on first run)..."):
        st.session_state.pipeline = TrafficGuardPipeline()

pipeline: TrafficGuardPipeline = st.session_state.pipeline

agents = pipeline.list_agents()
st.markdown("**Active Agents:** " + " ".join(f'<span class="agent-badge">{a}</span>' for a in agents), unsafe_allow_html=True)

tab_process, tab_command, tab_records, tab_analytics = st.tabs(
    ["Process Image", "Command Agent", "Violation Records", "Analytics & Insights"]
)

with tab_process:
    uploaded_files = st.file_uploader("Upload traffic images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)

    if uploaded_files:
        st.write("### Manual Overrides")
        light_statuses = {}
        cols = st.columns(min(len(uploaded_files), 4) if len(uploaded_files) > 0 else 1)
        for i, f in enumerate(uploaded_files):
            with cols[i % 4]:
                st.image(f, width=150)
                light_statuses[f.name] = st.checkbox(f"🔴 Light is RED in {f.name}", value=True)

        if st.button("Analyze Images", type="primary"):
            st.session_state.analysis_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import concurrent.futures

            def process_single_image(idx, uploaded, is_red):
                import tempfile
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                
                result = pipeline.process_image(tmp_path, is_red_light=is_red)
                return idx, uploaded.name, result

            with st.spinner(f"Running multi-agent pipeline concurrently for {len(uploaded_files)} images..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(process_single_image, i, f, light_statuses[f.name]) for i, f in enumerate(uploaded_files)]
                    
                    for count, future in enumerate(concurrent.futures.as_completed(futures)):
                        idx, filename, result = future.result()
                        st.session_state.analysis_results.append({
                            "idx": idx,
                            "filename": filename,
                            "result": result
                        })
                        
                        progress_bar.progress((count + 1) / len(uploaded_files))
                        status_text.text(f"Processed {count+1} of {len(uploaded_files)} images...")
            
            # Sort results to keep the tab order identical to the upload order
            st.session_state.analysis_results.sort(key=lambda x: x["idx"])
            status_text.success("All images processed successfully!")

        if st.session_state.get("analysis_results"):
            results = st.session_state.analysis_results
            
            # Create tabs for each processed image to keep the dashboard clean
            image_tabs = st.tabs([f"Image {i+1} ({r['filename']})" for i, r in enumerate(results)])

            for i, r in enumerate(results):
                result = r["result"]
                filename = r["filename"]

                with image_tabs[i]:
                    st.markdown(f"### Analysis: {filename}")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Annotated Evidence")
                        if result.evidence_path and Path(result.evidence_path).exists():
                            st.image(result.evidence_path, use_container_width=True)
                        else:
                            st.warning("Evidence image not generated.")

                    with col2:
                        st.subheader("Results")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Time", f"{result.processing_time_ms:.0f} ms")
                        m2.metric("Detections", len(result.detections))
                        m3.metric("Candidates", len(result.candidate_violations))
                        m4.metric("Approved", len(result.violations))

                        tera_llm = result.insights.get("tera_llm", {})
                        if tera_llm:
                            st.caption(f"TERA-LLM status: **{tera_llm.get('status', 'n/a')}**")

                        tera_rules = result.insights.get("tera_rules", {})
                        if tera_rules:
                            st.caption(
                                f"Rule TERA pre-check: {tera_rules.get('validated_count', 0)} passed, "
                                f"{tera_rules.get('rejected_count', 0)} rejected"
                            )

                        st.write("**Preprocessing:**", ", ".join(result.preprocessing_applied))

                        if result.violations:
                            for v in result.violations:
                                tera = v.metadata.get("tera", {})
                                reviewer = tera.get("reviewer", "TERA")
                                st.error(
                                    f"**{v.violation_type}** — {v.description}\n\n"
                                    f"Confidence: {v.confidence} | {tera.get('regulation_code', '')} | Reviewer: {reviewer}\n\n"
                                    f"_{tera.get('legal_justification', '')}_\n\n"
                                    f"Reasoning: {tera.get('reasoning', '')}"
                                )
                        else:
                            st.success("No violations approved by TERA.")

                        if result.plates:
                            st.write("**License Plates:**")
                            for p in result.plates:
                                st.info(f"{p.text} (conf: {p.confidence})")

                    with st.expander("Agent Execution Trace"):
                        for step in result.agent_trace:
                            st.write(f"**{step['agent']}** — {step['status']} ({step['duration_ms']:.0f} ms)")
                            if step.get("details"):
                                st.caption(str(step["details"]))

                    with st.expander("Scene Graph"):
                        st.json(result.scene_graph)

                    with st.expander("Full JSON response"):
                        st.json(result.to_dict())

with tab_command:
    st.subheader("Command Agent — Natural Language Queries")
    st.caption("Ask enforcement questions in plain English")

    examples = [
        "Show violation hotspots",
        "List repeat offenders",
        "Summary report",
        "Show helmet violations",
        "Recent violations last 7 days",
    ]
    query = st.text_input("Your query", placeholder="e.g. Show repeat offenders")
    days = st.slider("Lookback period (days)", 7, 90, 30, key="cmd_days")

    ex_cols = st.columns(3)
    for i, ex in enumerate(examples[:3]):
        if ex_cols[i].button(ex, key=f"ex_{i}"):
            st.session_state.cmd_query = ex
    if "cmd_query" in st.session_state:
        query = st.session_state.cmd_query

    if st.button("Run Query", type="primary") or query:
        if query:
            with st.spinner("Command Agent processing..."):
                response = pipeline.query(query, days=days)
            st.markdown("**Response:**")
            st.info(response["response"])
            if response.get("data"):
                st.dataframe(response["data"], use_container_width=True)

with tab_records:
    st.subheader("Searchable Violation Records")
    c1, c2 = st.columns(2)
    vtype = c1.selectbox(
        "Violation type",
        ["All", "helmet_non_compliance", "seatbelt_non_compliance", "triple_riding",
         "wrong_side_driving", "stop_line_violation", "red_light_violation", "illegal_parking", "none"],
    )
    plate = c2.text_input("Plate number (partial match)")

    records = pipeline.db.search(
        violation_type=None if vtype == "All" else vtype,
        plate_number=plate or None,
        limit=200,
    )
    if records:
        st.dataframe(records, use_container_width=True)
    else:
        st.info("No records found. Process images to populate the database.")

with tab_analytics:
    st.subheader("Insight Agent — Statistics & Trends")
    days = st.slider("Report period (days)", 7, 90, 30, key="analytics_days")
    if st.button("Generate Report"):
        analytics = pipeline.get_analytics(days=days)
        summary = analytics["summary"]
        insights = analytics.get("insights", {})
        charts = analytics["charts"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", summary.get("total_records", 0))
        m2.metric("Violation Events", summary.get("violation_events", 0))
        m3.metric("Avg Confidence", summary.get("avg_confidence", 0))
        m4.metric("Repeat Offenders", len(insights.get("repeat_offenders", [])))

        st.write("**Violations by Type:**", summary.get("violations_by_type", {}))

        if insights.get("hotspots"):
            st.subheader("Violation Hotspots")
            st.dataframe(insights["hotspots"], use_container_width=True)

        if insights.get("repeat_offenders"):
            st.subheader("Repeat Offenders")
            st.dataframe(insights["repeat_offenders"], use_container_width=True)

        if insights.get("enforcement_priority"):
            st.write("**Enforcement Priority:**", ", ".join(insights["enforcement_priority"]))

        if charts.get("violations_by_type"):
            st.image(charts["violations_by_type"], caption="Violations by Type")
        if charts.get("daily_trend"):
            st.image(charts["daily_trend"], caption="Daily Trend")

        csv_path = pipeline.reporter.export_csv()
        st.download_button(
            "Download CSV Report",
            data=Path(csv_path).read_bytes(),
            file_name="violations_export.csv",
            mime="text/csv",
        )
