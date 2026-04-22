"""Streamlit UI entry point."""
import streamlit as st

st.set_page_config(
    page_title="Text2SQL LRG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Explainable Text2SQL with Logical Reasoning Graphs")
st.markdown(
    """
    Welcome to the **Text2SQL LRG** research platform.

    Use the sidebar to navigate between:
    - **Query** — convert a natural language question to SQL and inspect the LRG
    - **Benchmark** — run evaluation on Spider / CoSQL / custom datasets
    - **Analysis** — explore failure mode breakdowns and explainability metrics
    """
)

with st.sidebar:
    st.header("Navigation")
    st.markdown("Use the **Pages** menu above to switch between views.")
    st.divider()
    st.header("Quick Config")
    backend = st.selectbox("LLM Backend", ["gemini", "ollama"])
    if backend == "gemini":
        api_key = st.text_input("Gemini API Key", type="password", key="gemini_key")
        if api_key:
            import os
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["LLM_BACKEND"] = "gemini"
    else:
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        if ollama_url:
            import os
            os.environ["OLLAMA_BASE_URL"] = ollama_url
            os.environ["LLM_BACKEND"] = "ollama"

    st.divider()
    st.caption("v0.1.0 | ASU SWM Project")
