"""Analysis page: explore existing benchmark results and failure breakdowns."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Analysis | Text2SQL LRG", layout="wide")
st.title("Results Analysis")
st.caption("Load saved benchmark reports and explore failure modes and explainability metrics.")

results_dir = st.text_input("Results Directory", value="results")

# ── Load available runs ────────────────────────────────────────────────────────
results_path = Path(results_dir)
if not results_path.exists():
    st.info("No results directory found. Run a benchmark first.")
    st.stop()

run_dirs = sorted([d for d in results_path.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
if not run_dirs:
    st.info("No benchmark runs found yet. Run a benchmark first.")
    st.stop()

run_labels = [d.name for d in run_dirs]
selected_run = st.selectbox("Select Run", run_labels)
run_path = results_path / selected_run

json_file = run_path / "report.json"
if not json_file.exists():
    st.error("Report JSON not found in selected run directory.")
    st.stop()

with open(json_file) as f:
    data = json.load(f)

# ── Overview ──────────────────────────────────────────────────────────────────
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", data.get("model", "—"))
col2.metric("Dataset", data.get("dataset", "—"))
col3.metric("Exact Match", f"{(data.get('exact_match') or 0)*100:.1f}%")
col4.metric("Execution Accuracy", f"{(data.get('execution_accuracy') or 0)*100:.1f}%")

# ── Failure modes chart ───────────────────────────────────────────────────────
fs = data.get("failure_summary", {})
if fs:
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("Failure Mode Distribution")
    cats = ["schema_linking", "join_hallucination", "nested_subquery", "self_join", "context_drift"]
    counts = [fs.get(c, 0) for c in cats]
    total = fs.get("total", 1)
    rates = [fs.get(f"{c}_rate", 0) * 100 for c in cats]

    col_bar, col_pie = st.columns([2, 1])
    with col_bar:
        fig = go.Figure(go.Bar(
            x=cats, y=rates,
            marker_color=["#D96A4A", "#4A90D9", "#D9A84A", "#7B4FD9", "#5BAD6F"],
            text=[f"{r:.1f}%" for r in rates], textposition="auto",
        ))
        fig.update_layout(
            yaxis_title="Error Rate (%)",
            xaxis_title="Failure Category",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_pie:
        fig2 = go.Figure(go.Pie(
            labels=cats, values=counts,
            hole=0.4,
            marker_colors=["#D96A4A", "#4A90D9", "#D9A84A", "#7B4FD9", "#5BAD6F"],
        ))
        fig2.update_layout(height=320, showlegend=True, legend=dict(font_size=9))
        st.plotly_chart(fig2, use_container_width=True)

# ── Explainability ────────────────────────────────────────────────────────────
es = data.get("explainability_summary", {})
if es:
    st.subheader("Explainability Metrics")
    import plotly.graph_objects as go

    metrics = {
        "Faithfulness": es.get("mean_faithfulness", 0),
        "Completeness": es.get("mean_completeness", 0),
        "Error Traceability": es.get("error_traceability_rate", 0),
    }
    fig = go.Figure(go.Bar(
        x=list(metrics.keys()),
        y=[v * 100 for v in metrics.values()],
        marker_color=["#4AD9C9", "#5BAD6F", "#D94A8C"],
        text=[f"{v*100:.1f}%" for v in metrics.values()],
        textposition="auto",
    ))
    fig.update_layout(yaxis_title="Score (%)", height=280, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Component scores ──────────────────────────────────────────────────────────
comp = data.get("component_scores", {})
if comp:
    st.subheader("Clause-Level Accuracy")
    import pandas as pd
    df = pd.DataFrame([{"Clause": k, "Accuracy (%)": round(v * 100, 1)} for k, v in comp.items()])
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── Per-example explorer ───────────────────────────────────────────────────────
st.subheader("Per-Example Explorer")
examples = data.get("per_example", [])
if examples:
    import pandas as pd
    df = pd.DataFrame([
        {
            "ID": e["example_id"],
            "DB": e["db_id"],
            "Question": e["question"][:80],
            "EM": e["exact_match"],
            "EX": e.get("execution_accuracy"),
            "Failures": ", ".join(e.get("failure_modes", [])) or "-",
            "Faithfulness": e.get("faithfulness"),
        }
        for e in examples
    ])

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        only_errors = st.checkbox("Show only incorrect predictions")
    with col_f2:
        failure_filter = st.multiselect(
            "Filter by failure mode",
            ["schema_linking", "join_hallucination", "nested_subquery", "self_join", "context_drift"],
        )

    if only_errors:
        df = df[df["EM"] == False]
    if failure_filter:
        df = df[df["Failures"].apply(lambda f: any(cat in f for cat in failure_filter))]

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detail view
    selected_id = st.selectbox("Select example ID for details", df["ID"].tolist() if not df.empty else [])
    if selected_id:
        ex = next((e for e in examples if e["example_id"] == selected_id), None)
        if ex:
            st.code(f"Q: {ex['question']}\n\nGold SQL:\n{ex['gold_sql']}\n\nPredicted SQL:\n{ex['predicted_sql']}", language="sql")
            if ex.get("lrg_summary"):
                st.text(ex["lrg_summary"])
