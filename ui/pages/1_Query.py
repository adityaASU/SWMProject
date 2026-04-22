"""Query page: interactive NL -> SQL with LRG visualisation."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Query | Text2SQL LRG", layout="wide")
st.title("Interactive Query")
st.caption("Convert a natural language question into SQL and inspect the Logical Reasoning Graph.")

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Query Settings")
    model_choice = st.selectbox(
        "Model",
        ["lrg", "prompt_few_shot", "prompt_zero_shot"],
        help="lrg = LRG-enhanced; prompt = direct LLM baseline",
    )
    data_dir = st.text_input(
        "Data Directory",
        value="data/spider",
        help="Directory containing database folders",
    )
    db_id = st.text_input("Database ID", value="college_2", help="e.g. college_2, flights, yelp")
    show_lrg_graph = st.checkbox("Show LRG graph JSON", value=False)

# ── Main query area ────────────────────────────────────────────────────────────
question = st.text_area(
    "Natural Language Question",
    height=80,
    placeholder="e.g. How many students are enrolled in each department?",
)

if st.button("Generate SQL", type="primary", disabled=not question.strip()):
    with st.spinner("Generating SQL..."):
        try:
            from src.config import load_config
            from src.llm.factory import create_llm
            from src.schema.parser import SchemaParser
            from src.baseline.registry import create_baseline

            cfg = load_config()
            llm = create_llm(cfg.llm)
            parser = SchemaParser()
            schema = parser.auto_parse(Path(data_dir), db_id)

            if model_choice == "lrg":
                from src.lrg.pipeline import LRGText2SQL
                from src.lrg.visualizer import render_lrg

                lrg_model = LRGText2SQL(llm)
                result, lrg, errors = lrg_model.predict_with_lrg(question, schema)
            else:
                model = create_baseline(model_choice, llm)
                result = model.predict(question, schema)
                lrg = None
                errors = []

        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()

    # Results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Generated SQL")
        st.code(result.predicted_sql, language="sql")

        if errors:
            st.warning("LRG Validation Warnings")
            for e in errors:
                st.write(f"- {e}")

        if lrg:
            st.subheader("LRG Summary")
            st.text(lrg.summary())

    with col2:
        if lrg:
            st.subheader("Logical Reasoning Graph")
            try:
                from src.lrg.visualizer import render_lrg
                img_bytes = render_lrg(lrg, title=question[:80])
                st.image(img_bytes, use_container_width=True)
            except Exception as exc:
                st.warning(f"Graph render failed: {exc}")

            if show_lrg_graph:
                with st.expander("LRG JSON"):
                    st.json(lrg.to_dict())
        else:
            st.subheader("Schema")
            st.text(schema.format_for_prompt())

# ── Schema preview ─────────────────────────────────────────────────────────────
with st.expander("Preview Schema"):
    if db_id and data_dir:
        try:
            from src.schema.parser import SchemaParser
            parser = SchemaParser()
            schema = parser.auto_parse(Path(data_dir), db_id)
            st.text(schema.format_for_prompt())
        except Exception as exc:
            st.info(f"Load a valid db_id to preview schema. ({exc})")
