# Text2SQL with Logical Reasoning Graphs

Explainable Text-to-SQL research platform with a **Logical Reasoning Graph (LRG)** as an interpretable intermediate representation. Supports Gemini (cloud) and Ollama (local GPU) as LLM backends, plug-and-play benchmarking on Spider / CoSQL / custom datasets, a FastAPI REST backend, and a Streamlit UI.

**Repo:** [github.com/adityaASU/SWMProject](https://github.com/adityaASU/SWMProject)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Clone & Install](#step-1--clone--install)
4. [Step 2 — Configure API Key](#step-2--configure-api-key)
5. [Step 3 — Download the Spider Dataset](#step-3--download-the-spider-dataset)
6. [Step 4 — Run a Single Query](#step-4--run-a-single-query)
7. [Step 5 — Run a Benchmark](#step-5--run-a-benchmark)
8. [Step 6 — Start the API](#step-6--start-the-api)
9. [Step 7 — Start the UI](#step-7--start-the-ui)
10. [Troubleshooting](#troubleshooting)
11. [Using Ollama Instead of Gemini](#using-ollama-instead-of-gemini)
12. [Adding a New Baseline](#adding-a-new-baseline)
13. [Adding a Custom Dataset](#adding-a-custom-dataset)
14. [Running Tests](#running-tests)
15. [Documentation Index](#documentation-index)
16. [Project Structure](#project-structure)

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
  ┌──────────────────┐      ┌────────────────────────┐
  │  Baseline         │      │  LRG Builder            │
  │  (direct LLM      │      │  (structured LLM call + │
  │   prompting)      │      │   schema validation)    │
  └────────┬──────────┘      └────────────┬───────────┘
           │                              │
           │                 ┌────────────▼───────────┐
           │                 │  Logical Reasoning Graph │
           │                 │  Entity · Filter · Agg  │
           │                 │  Join · Subgraph · Alias │
           │                 └────────────┬───────────┘
           │                              │
           │                 ┌────────────▼───────────┐
           │                 │  SQL Synthesizer         │
           │                 │  (deterministic, no LLM) │
           │                 └────────────┬───────────┘
           └──────────────────────┬───────┘
                                  ▼
                        ┌─────────────────┐
                        │   Evaluator      │
                        │  EM · EX · FM   │
                        │  Explainability  │
                        └─────────────────┘
```

---

## Prerequisites

- **Python 3.11+** — check with `python --version`
- **Git**
- **A Gemini API key** — free at [aistudio.google.com](https://aistudio.google.com/apikey) (takes 30 seconds)
- *(Optional)* [Graphviz](https://graphviz.org/download/) for richer LRG graph rendering
- *(Optional)* [Ollama](https://ollama.com/) if you want to run locally on GPU instead of Gemini

---

## Step 1 — Clone & Install

```bash
git clone https://github.com/adityaASU/SWMProject.git
cd SWMProject
```

Create a virtual environment (keeps your system Python clean):

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

> This installs FastAPI, Streamlit, NetworkX, google-genai, sqlglot, and everything else automatically.

---

## Step 2 — Configure API Key

Copy the template and fill in your key:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in any text editor and set:

```env
LLM_BACKEND=gemini
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.0-flash-lite
```

**Where to get a free Gemini API key:**
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click **Create API key**
3. Copy the key into `.env`

> `gemini-2.0-flash-lite` is recommended for the free tier — it has a higher rate limit than `gemini-2.0-flash`.

---

## Step 3 — Download the Spider Dataset

The Spider dataset has two parts:
- **Questions + SQL** (downloaded automatically via HuggingFace)
- **SQLite databases** (needed for execution accuracy — manual download from Kaggle)

### Part A — Auto-download questions and SQL

```bash
python scripts/download_data.py --dataset spider
```

This converts the HuggingFace parquet files into `data/spider/dev.json` (1,034 examples) and `data/spider/train_spider.json` (7,000 examples). Exact-match benchmarking works after this step.

### Part B — SQLite databases (for execution accuracy)

1. Go to [kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset)
2. Click **Download** (free Kaggle account required)
3. Extract the zip — you'll get a folder containing `database/`, `dev.json`, `tables.json`, etc.
4. Copy **everything** from inside that folder into `data/spider/`:

```
data/spider/
├── database/          ← 166 SQLite files (from Kaggle)
│   ├── college_2/
│   │   └── college_2.sqlite
│   ├── flights/
│   └── ...
├── dev.json           ← already there from Step A
├── train_spider.json  ← already there from Step A
└── tables.json        ← copy from Kaggle zip
```

> **Without the databases:** exact-match accuracy still works fully. Execution accuracy will show `N/A`.

---

## Step 4 — Run a Single Query

```bash
# Basic query with the LRG model
python scripts/run_query.py --db college_2 --question "How many students are enrolled in each department?"

# Save the LRG graph as a PNG image
python scripts/run_query.py --db college_2 --question "How many students are enrolled in each department?" --save-lrg lrg.png

# Use the direct-prompt baseline (no LRG)
python scripts/run_query.py --db college_2 --question "List all students" --model prompt_few_shot

# Print the full LRG JSON structure
python scripts/run_query.py --db college_2 --question "..." --show-lrg-json
```

**Expected output:**
```
Model  : lrg
DB     : college_2
Question: How many students are enrolled in each department?

=== Generated SQL ===
SELECT departments.dept_name, COUNT(student.id)
FROM student
JOIN departments ON student.dept_id = departments.id
GROUP BY departments.dept_name

=== LRG Summary ===
LRGGraph: 5 nodes, 4 edges
  entity: 2
  aggregation: 1
  filter: 0
  grouping: 1
```

> **Rate limit note:** The Gemini free tier allows ~30 requests/minute. If you see a 429 error, the script automatically retries after 62 seconds.

---

## Step 5 — Run a Benchmark

Evaluate on the Spider dev set:

```bash
# Quick run — first 20 examples
python scripts/run_benchmark.py --dataset spider --model lrg --max-examples 20

# Compare the baseline
python scripts/run_benchmark.py --dataset spider --model prompt_few_shot --max-examples 20

# Full dev set (1034 examples — takes a while on free tier)
python scripts/run_benchmark.py --dataset spider --model lrg
```

Results are saved to `results/<run_id>/report.json` and `results/<run_id>/report.md`.

**Sample output:**
```
RESULTS — lrg/gemini/gemini-2.0-flash-lite on spider
==================================================
  Examples evaluated : 20
  Exact Match        : 45.0%
  Execution Accuracy : 50.0%

  Failure Modes:
    schema_linking           :  3 (15.0%)
    join_hallucination        :  2 (10.0%)
    nested_subquery           :  1 (5.0%)
    self_join                 :  0 (0.0%)
    context_drift             :  0 (0.0%)
```

---

## Step 6 — Start the API

```bash
uvicorn src.api.app:app --reload --port 8000
```

Open **[localhost:8000/docs](http://localhost:8000/docs)** for interactive Swagger UI.

| Method | Path | What it does |
|--------|------|-------------|
| `POST` | `/query` | Convert NL question to SQL |
| `GET` | `/schema/{db_id}` | View a database schema |
| `POST` | `/benchmark/run` | Start a background benchmark run |
| `GET` | `/benchmark/status/{run_id}` | Check benchmark progress |

**Quick API test:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many students?", "db_id": "college_2", "model": "lrg"}'
```

---

## Step 7 — Start the UI

```bash
streamlit run ui/app.py
```

Opens at **[localhost:8501](http://localhost:8501)**. Three pages:

| Page | What you can do |
|------|----------------|
| **Query** | Type a question, pick a database, see the SQL + LRG graph side-by-side |
| **Benchmark** | Run evaluations in-browser and view charts |
| **Analysis** | Load saved reports, filter failure modes, explore per-example results |

---

## Troubleshooting

### `google.generativeai` deprecation warning
The old SDK is no longer used. Make sure you installed the latest requirements:
```bash
pip install -r requirements.txt
```

### `models/gemini-X is not found` (404 error)
Your model name is wrong. Run this to see what's available for your API key:
```python
from dotenv import load_dotenv; load_dotenv('.env')
import os
from google import genai
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
for m in client.models.list():
    if 'generateContent' in (m.supported_actions or []):
        print(m.name)
```
Then update `GEMINI_MODEL=` in your `.env` to one of the listed names.

### `429 RESOURCE_EXHAUSTED` (rate limit)
You've hit the free tier per-minute limit. The script retries automatically after 62 seconds. For benchmarks, use `--max-examples 20` to stay within limits, or add a delay between requests by setting `parallel_workers: 1` (already the default) in `configs/default.yaml`.

### `Cannot find schema for db_id='...'`
The SQLite databases are not in `data/spider/database/`. Follow **Step 3 Part B** to download them from Kaggle. Alternatively use `--data-dir` to point to where you placed the files:
```bash
python scripts/run_query.py --db college_2 --question "..." --data-dir /path/to/spider
```

### Schema found but no tables (empty LRG)
The LLM extraction failed silently. Check your API key is set correctly in `.env` and that the model name is valid.

---

## Using Ollama Instead of Gemini

If you have an NVIDIA GPU (e.g. RTX 3070) and want to run fully locally:

```bash
# Install Ollama from https://ollama.com/
ollama pull llama3      # or sqlcoder, mistral, etc.
```

Update `.env`:
```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

Everything else works exactly the same.

---

## Adding a New Baseline

1. Create `src/baseline/my_model.py`:

```python
from src.baseline.base import BaseText2SQL, PredictionResult
from src.schema.parser import SchemaInfo

class MyModel(BaseText2SQL):
    @property
    def model_name(self) -> str:
        return "my_model"

    def predict(self, question, schema, conversation_history=None):
        sql = "SELECT ..."  # your generation logic
        return PredictionResult(
            question=question, db_id=schema.db_id,
            predicted_sql=sql, model_name=self.model_name
        )
```

2. Register it in `src/baseline/registry.py`:

```python
_REGISTRY["my_model"] = lambda llm, **kw: MyModel(llm, **kw)
```

3. Run it:

```bash
python scripts/run_benchmark.py --model my_model --dataset spider --max-examples 20
```

---

## Adding a Custom Dataset

Create a JSONL file — one JSON object per line:

```jsonl
{"question": "How many students?", "db_id": "my_db", "gold_sql": "SELECT COUNT(*) FROM students"}
{"question": "List all courses", "db_id": "my_db", "gold_sql": "SELECT * FROM courses"}
```

Place your SQLite DB at `data/custom/my_db/my_db.sqlite`, then run:

```bash
python scripts/run_benchmark.py \
    --dataset custom \
    --custom-jsonl data/custom/my_examples.jsonl \
    --data-dir data/custom \
    --model lrg
```

---

## Running Tests

All tests are **fully offline** — no API key or database files needed:

```bash
pytest tests/ -v
```

Covers schema parsing, LRG graph construction, SQL synthesis, and all evaluation metrics.

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | Full system architecture and module dependencies |
| [docs/lrg_design.md](docs/lrg_design.md) | LRG node/edge specification and design rationale |
| [docs/evaluation.md](docs/evaluation.md) | Metric definitions (EM, EX, failure modes, explainability) |
| [docs/contributing.md](docs/contributing.md) | How to add baselines, datasets, node types |

---

## Project Structure

```
SWMProject/
├── .env.example          ← copy to .env and add your API key
├── .env                  ← your credentials (gitignored)
├── configs/
│   ├── default.yaml      ← model, paths, benchmark settings
│   ├── gemini.yaml
│   └── ollama.yaml
├── data/                 ← datasets live here (gitignored)
│   ├── spider/
│   │   ├── dev.json               (auto-downloaded)
│   │   ├── train_spider.json      (auto-downloaded)
│   │   ├── tables.json            (from Kaggle zip)
│   │   └── database/              (from Kaggle zip)
│   │       ├── college_2/
│   │       └── ...
│   └── custom/           ← drop your own JSONL + SQLite here
├── docs/                 ← extended documentation
├── results/              ← benchmark outputs (gitignored)
├── scripts/
│   ├── download_data.py  ← run this first
│   ├── run_query.py      ← test a single question
│   └── run_benchmark.py  ← run full evaluation
├── src/
│   ├── config.py         ← loads .env + YAML
│   ├── llm/              ← Gemini + Ollama backends
│   ├── schema/           ← DB schema parser + NetworkX graph
│   ├── baseline/         ← plug-and-play model registry
│   ├── lrg/              ← core LRG framework
│   │   ├── nodes.py      ← all node/edge types (Pydantic)
│   │   ├── graph.py      ← LRGGraph DiGraph + validation
│   │   ├── builder.py    ← NL → LRG via structured LLM
│   │   ├── synthesizer.py← LRG → SQL (no LLM, deterministic)
│   │   ├── visualizer.py ← renders LRG as PNG
│   │   └── pipeline.py   ← LRGText2SQL end-to-end model
│   ├── evaluation/       ← EM, EX, failure modes, explainability
│   ├── benchmark/        ← runner, reporter, dataset loaders
│   └── api/              ← FastAPI app + routers
├── tests/                ← offline pytest suite
└── ui/                   ← Streamlit UI
    └── pages/
        ├── 1_Query.py    ← interactive NL→SQL + LRG viz
        ├── 2_Benchmark.py← run + view benchmarks
        └── 3_Analysis.py ← failure mode charts
```
