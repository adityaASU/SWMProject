"""Benchmark page: run and display evaluation results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Benchmark | Text2SQL LRG", layout="wide")
st.title("Benchmark Runner")
st.caption("Evaluate models on Spider, CoSQL, or a custom JSONL dataset.")

with st.sidebar:
    st.header("Benchmark Settings")
    dataset = st.selectbox("Dataset", ["spider", "cosql", "custom"])
    split = st.selectbox("Split", ["dev", "train"])
    model_choice = st.selectbox("Model", ["lrg", "prompt_few_shot", "prompt_zero_shot"])
    max_examples = st.number_input("Max Examples (0 = all)", min_value=0, value=50)
    data_dir = st.text_input("Data Directory", value=f"data/{dataset}")
    custom_jsonl = ""
    if dataset == "custom":
        custom_jsonl = st.text_input("Custom JSONL Path", value="data/custom/examples.jsonl")

# ── Run benchmark ──────────────────────────────────────────────────────────────
if st.button("Run Benchmark", type="primary"):
    with st.spinner("Running benchmark — this may take a while..."):
        try:
            from src.config import load_config
            from src.llm.factory import create_llm
            from src.baseline.registry import create_baseline
            from src.benchmark.runner import BenchmarkRunner
            from src.benchmark.reporter import save_report

            cfg = load_config()
            llm = create_llm(cfg.llm)
            model = create_baseline(model_choice, llm)

            base = Path(data_dir)
            if dataset == "spider":
                from src.benchmark.datasets.spider import SpiderDataset
                ds = SpiderDataset(base, split=split)
            elif dataset == "cosql":
                from src.benchmark.datasets.cosql import CoSQLDataset
                ds = CoSQLDataset(base, split=split)
            else:
                from src.benchmark.datasets.custom import CustomDataset
                ds = CustomDataset(Path(custom_jsonl), base)

            runner = BenchmarkRunner(
                model=model,
                dataset=ds,
                config=cfg,
                max_examples=int(max_examples) if max_examples > 0 else None,
            )
            report = runner.run()
            json_path, md_path = save_report(report, cfg.paths.results_dir)
            st.session_state["last_report"] = report
            st.session_state["last_json_path"] = str(json_path)
            st.session_state["last_md_path"] = str(md_path)
            st.success(f"Benchmark complete! Results saved to {json_path}")

        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")
            st.stop()

# ── Display report ─────────────────────────────────────────────────────────────
if "last_report" in st.session_state:
    report = st.session_state["last_report"]
    st.divider()
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Exact Match", f"{(report.exact_match or 0)*100:.1f}%")
    col2.metric("Execution Accuracy", f"{(report.execution_accuracy or 0)*100:.1f}%")
    col3.metric("Examples Evaluated", report.n_examples)

    if report.failure_summary:
        st.subheader("Failure Mode Breakdown")
        import plotly.graph_objects as go
        fs = report.failure_summary
        cats = ["schema_linking", "join_hallucination", "nested_subquery", "self_join", "context_drift"]
        counts = [fs.get(c, 0) for c in cats]
        fig = go.Figure(go.Bar(x=cats, y=counts, marker_color="#4A90D9"))
        fig.update_layout(xaxis_title="Failure Category", yaxis_title="Count", height=300)
        st.plotly_chart(fig, use_container_width=True)

    if report.explainability_summary:
        es = report.explainability_summary
        st.subheader("Explainability Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Faithfulness", f"{(es.get('mean_faithfulness', 0))*100:.1f}%")
        c2.metric("Completeness", f"{(es.get('mean_completeness', 0))*100:.1f}%")
        c3.metric("Error Traceability", f"{(es.get('error_traceability_rate', 0))*100:.1f}%")

    # Per-example table
    st.subheader("Per-Example Results")
    import pandas as pd
    rows = []
    for ex in report.per_example[:100]:
        rows.append({
            "ID": ex["example_id"],
            "DB": ex["db_id"],
            "Question": ex["question"][:80],
            "EM": "✓" if ex["exact_match"] else "✗",
            "EX": "✓" if ex.get("execution_accuracy") else ("✗" if ex.get("execution_accuracy") is False else "-"),
            "Failures": ", ".join(ex.get("failure_modes", [])) or "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Download buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.session_state.get("last_json_path"):
            json_data = Path(st.session_state["last_json_path"]).read_text()
            st.download_button("Download JSON Report", json_data, file_name="report.json", mime="application/json")
    with col_b:
        if st.session_state.get("last_md_path"):
            md_data = Path(st.session_state["last_md_path"]).read_text()
            st.download_button("Download Markdown Report", md_data, file_name="report.md", mime="text/markdown")
