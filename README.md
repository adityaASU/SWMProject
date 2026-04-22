# Text2SQL with Logical Reasoning Graphs

Explainable Text-to-SQL research platform with a **Logical Reasoning Graph (LRG)** as an interpretable intermediate representation. Supports Gemini (cloud) and Ollama (local GPU) as LLM backends, plug-and-play benchmarking on Spider / CoSQL / custom datasets, a FastAPI REST backend, and a Streamlit UI.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Download Data](#download-data)
6. [Run a Single Query](#run-a-single-query)
7. [Run a Benchmark](#run-a-benchmark)
8. [Start the API](#start-the-api)
9. [Start the UI](#start-the-ui)
10. [Adding a New Baseline](#adding-a-new-baseline)
11. [Adding a Custom Dataset](#adding-a-custom-dataset)
12. [Running Tests](#running-tests)
13. [Documentation Index](#documentation-index)
14. [Project Structure](#project-structure)

---

## Architecture

```
Natural Language Question
        │
        ▼
  Schema Parser ──► Schema Graph (NetworkX, FK edges)
        │                    │
        │         ┌──────────┘
        ▼         ▼
  ┌─────────────────────┐      ┌───────────────────────┐
  │  Baseline (direct   │      │  LRG Builder           │
  │  LLM prompting)     │      │  (structured LLM +     │
  └──────────┬──────────┘      │   schema validation)   │
             │                 └──────────┬─────────────┘
             │                            │
             │                 ┌──────────▼─────────────┐
             │                 │  Logical Reasoning Graph │
             │                 │  (Entity, Filter, Agg,  │
             │                 │   Join, Subgraph, Alias) │
             │                 └──────────┬─────────────┘
             │                            │
             │                 ┌──────────▼─────────────┐
             │                 │  SQL Synthesizer         │
             │                 │  (deterministic, no LLM) │
             │                 └──────────┬─────────────┘
             │                            │
             └────────────────┬───────────┘
                              ▼
                    ┌─────────────────┐
                    │   Evaluator      │
                    │  EM / EX / FM /  │
                    │  Explainability  │
                    └─────────────────┘
```

---

## Prerequisites

- Python 3.11+
- (Optional) [Graphviz](https://graphviz.org/download/) installed for LRG DOT rendering
- For Gemini: a Google AI Studio API key
- For Ollama: [Ollama](https://ollama.com/) running locally with a model pulled (e.g. `ollama pull llama3`)

---

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/text2sql-lrg.git
cd text2sql-lrg

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Choose "gemini" or "ollama"
LLM_BACKEND=gemini

# Gemini (required if LLM_BACKEND=gemini)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash

# Ollama (required if LLM_BACKEND=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

Advanced settings (model temperature, benchmark limits, etc.) live in `configs/default.yaml`.

---

## Download Data

```bash
# Download both Spider and CoSQL
python scripts/download_data.py

# Download only Spider
python scripts/download_data.py --dataset spider

# Download only CoSQL
python scripts/download_data.py --dataset cosql
```

> **Note:** If automatic download fails (Google Drive rate limits), see the manual download links printed by the script and place the folders at `data/spider/` and `data/cosql/`.

---

## Run a Single Query

```bash
# LRG model (default)
python scripts/run_query.py --db college_2 --question "How many students are enrolled in each department?"

# Save the LRG visualisation to a PNG
python scripts/run_query.py --db college_2 --question "..." --save-lrg lrg.png

# Use the direct LLM baseline instead
python scripts/run_query.py --db college_2 --question "..." --model prompt_few_shot

# Print the LRG JSON
python scripts/run_query.py --db college_2 --question "..." --show-lrg-json
```

---

## Run a Benchmark

```bash
# Evaluate the LRG model on Spider dev set (first 50 examples)
python scripts/run_benchmark.py --dataset spider --model lrg --max-examples 50

# Evaluate a baseline on CoSQL
python scripts/run_benchmark.py --dataset cosql --model prompt_few_shot --max-examples 100

# Evaluate on a custom JSONL dataset
python scripts/run_benchmark.py --dataset custom \
    --custom-jsonl data/custom/my_examples.jsonl \
    --model lrg

# Full Spider dev run (no limit)
python scripts/run_benchmark.py --dataset spider --model lrg
```

Results are saved to `results/<run_id>/` as `report.json` and `report.md`.

---

## Start the API

```bash
uvicorn src.api.app:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | NL → SQL (with optional LRG image) |
| `GET` | `/schema/{db_id}` | Inspect a database schema |
| `POST` | `/benchmark/run` | Start a background benchmark run |
| `GET` | `/benchmark/status/{run_id}` | Poll benchmark status |

Example `POST /query` body:
```json
{
  "question": "How many students are in each department?",
  "db_id": "college_2",
  "model": "lrg",
  "return_lrg": true,
  "return_lrg_image": true
}
```

---

## Start the UI

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`. Three pages:

- **Query** — interactive NL → SQL with LRG graph visualisation
- **Benchmark** — run evaluations and view results inline
- **Analysis** — load saved reports and explore failure mode charts

---

## Adding a New Baseline

1. Create `src/baseline/my_model.py` with a class extending `BaseText2SQL`:

```python
from src.baseline.base import BaseText2SQL, PredictionResult
from src.schema.parser import SchemaInfo

class MyModel(BaseText2SQL):
    @property
    def model_name(self) -> str:
        return "my_model"

    def predict(self, question, schema, conversation_history=None):
        sql = "SELECT ..."  # your logic here
        return PredictionResult(question=question, db_id=schema.db_id,
                                predicted_sql=sql, model_name=self.model_name)
```

2. Register it in `src/baseline/registry.py`:

```python
from src.baseline.my_model import MyModel

_REGISTRY["my_model"] = lambda llm, **kw: MyModel(llm, **kw)
```

3. Use it:

```bash
python scripts/run_benchmark.py --model my_model --dataset spider
```

---

## Adding a Custom Dataset

Create a JSONL file where each line is:

```json
{"question": "How many students?", "db_id": "my_db", "gold_sql": "SELECT COUNT(*) FROM students"}
```

Optional fields: `"example_id"`, `"conversation_history"`, `"metadata"`.

Place your SQLite databases in `data/custom/<db_id>/<db_id>.sqlite`, then run:

```bash
python scripts/run_benchmark.py \
    --dataset custom \
    --custom-jsonl data/custom/my_examples.jsonl \
    --data-dir data/custom \
    --model lrg
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover schema parsing, LRG graph operations, SQL synthesis, and all evaluation metrics. Tests are fully offline (no LLM calls).

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | Full system architecture and data flow |
| [docs/lrg_design.md](docs/lrg_design.md) | LRG node/edge specification and design rationale |
| [docs/evaluation.md](docs/evaluation.md) | Metric definitions and evaluation methodology |
| [docs/contributing.md](docs/contributing.md) | Contribution guidelines for teammates |

---

## Project Structure

```
text2sql-lrg/
├── .env.example          # Template for credentials
├── configs/              # YAML config files
├── data/                 # Downloaded datasets (gitignored)
├── docs/                 # Project documentation
├── results/              # Benchmark outputs (gitignored)
├── scripts/              # CLI entry points
│   ├── download_data.py
│   ├── run_query.py
│   └── run_benchmark.py
├── src/
│   ├── config.py         # Central config loader
│   ├── llm/              # LLM backends (Gemini, Ollama)
│   ├── schema/           # Schema parser + NetworkX graph
│   ├── baseline/         # Plug-and-play baseline models
│   ├── lrg/              # Core LRG framework
│   │   ├── nodes.py      # Node/edge Pydantic models
│   │   ├── graph.py      # LRGGraph DiGraph wrapper
│   │   ├── builder.py    # NL → LRG via structured LLM
│   │   ├── synthesizer.py# LRG → SQL (deterministic)
│   │   ├── visualizer.py # PNG rendering
│   │   └── pipeline.py   # LRGText2SQL (implements BaseText2SQL)
│   ├── evaluation/       # Metrics, failure modes, explainability
│   ├── benchmark/        # Runner, reporter, dataset loaders
│   └── api/              # FastAPI app
├── tests/                # Pytest test suite
└── ui/                   # Streamlit UI
    └── pages/
        ├── 1_Query.py
        ├── 2_Benchmark.py
        └── 3_Analysis.py
```
