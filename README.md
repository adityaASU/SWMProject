# Text2SQL with Logical Reasoning Graphs

Explainable Text-to-SQL research platform with a **Logical Reasoning Graph (LRG)** as an interpretable intermediate representation between natural language and SQL.

Supports **Gemini** (cloud) and **Ollama** (local GPU) as LLM backends, plug-and-play benchmarking on Spider / CoSQL / custom datasets, a FastAPI REST backend, and a Streamlit UI.

**Repo:** [github.com/adityaASU/SWMProject](https://github.com/adityaASU/SWMProject)

---

## Table of Contents

1. [How it Works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Clone & Install](#step-1--clone--install)
4. [Step 2 — Choose an LLM Backend](#step-2--choose-an-llm-backend)
   - [Option A: Ollama (local, recommended)](#option-a-ollama-local-recommended)
   - [Option B: Gemini (cloud)](#option-b-gemini-cloud)
5. [Step 3 — Download the Spider Dataset](#step-3--download-the-spider-dataset)
6. [Step 4 — Run a Single Query](#step-4--run-a-single-query)
7. [Step 5 — Run a Benchmark](#step-5--run-a-benchmark)
8. [Step 6 — Start the API](#step-6--start-the-api)
9. [Step 7 — Start the UI](#step-7--start-the-ui)
10. [Troubleshooting](#troubleshooting)
11. [Adding a New Baseline](#adding-a-new-baseline)
12. [Adding a Custom Dataset](#adding-a-custom-dataset)
13. [Running Tests](#running-tests)
14. [Documentation Index](#documentation-index)
15. [Project Structure](#project-structure)

---

## How it Works

```
Natural Language Question
        │
        ▼
  Schema Parser ──► Schema Graph (NetworkX, FK edges)
        │                    │
        │         ┌──────────┘
        ▼         ▼
  ┌──────────────────┐      ┌────────────────────────────┐
  │  Baseline         │      │  LRG Builder                │
  │  (direct LLM      │      │  LLM extracts entities,     │
  │   prompting)      │      │  filters, aggs → validates  │
  └────────┬──────────┘      │  against schema graph       │
           │                 └────────────┬───────────────┘
           │                              │
           │                 ┌────────────▼───────────┐
           │                 │  Logical Reasoning Graph │
           │                 │  Entity · Filter · Agg   │
           │                 │  Join · Subgraph · Alias  │
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
                        │  EM · EX · FM    │
                        │  Explainability  │
                        └─────────────────┘
```

The key idea: instead of asking the LLM to write SQL directly, we ask it to identify *logical components* (tables, columns, filters, aggregations) as structured JSON. Python then assembles a **Logical Reasoning Graph** and deterministically synthesizes SQL from it — no LLM hallucination in the final SQL generation step.

---

## Prerequisites

- **Python 3.11+** — check with `python --version`
- **Git**
- One of:
  - **Ollama** (local, free, no API key) — [ollama.com](https://ollama.com/) — recommended if you have a GPU
  - **Gemini API key** (cloud, free tier) — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- *(Optional)* [Graphviz](https://graphviz.org/download/) for richer LRG graph rendering

---

## Step 1 — Clone & Install

```bash
git clone https://github.com/adityaASU/SWMProject.git
cd SWMProject/text2sql-lrg
```

Create a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

> **Windows note:** Use `.venv\Scripts\Activate.ps1` in PowerShell, **not** `source` (that's a bash command). If you get an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once and try again.

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 2 — Choose an LLM Backend

### Option A: Ollama (local, recommended)

Ollama runs models on your own GPU — no API key, no rate limits, fully private.

**1. Install Ollama**

Download and install from [ollama.com](https://ollama.com/). After installation, Ollama starts automatically in the background on Windows (check your system tray).

**2. Fix PATH in PowerShell (Windows)**

Ollama is installed but may not be in your current shell's PATH. Run this once per terminal session:

```powershell
$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"
```

Verify it works:
```powershell
ollama list
```

**3. Check if Ollama server is running**

Ollama on Windows auto-starts in the background. Verify:
```powershell
ollama list   # should show installed models without error
```

If you see `bind: Only one usage of each socket address` when running `ollama serve`, that means **Ollama is already running** — this is fine, ignore it.

If you need to start it manually:
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```
Leave this window open.

**4. Pull the model**

We use `llama3.2:3b` — a small (~2 GB), fast general-purpose model that correctly follows JSON instructions:

```powershell
ollama pull llama3.2:3b
```

> **Why not sqlcoder?** `sqlcoder:7b` is trained to output SQL directly and ignores JSON format instructions. Our LRG builder needs the model to output structured JSON to build the reasoning graph. Use `llama3.2:3b`, `mistral:7b`, or `qwen2.5:7b` for the LRG pipeline.

**5. Configure `.env`**

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and set:
```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

---

### Option B: Gemini (cloud)

**1. Get a free API key**

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click **Create API key**
3. Copy the key

**2. Configure `.env`**

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

```env
LLM_BACKEND=gemini
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.0-flash-lite
```

> `gemini-2.0-flash-lite` is recommended for the free tier — higher rate limit than `gemini-2.0-flash`. The script auto-retries on rate limit errors (waits 62 seconds between retries).

---

## Step 3 — Download the Spider Dataset

The Spider dataset has two parts:
- **Questions + SQL** — auto-downloaded via HuggingFace
- **SQLite databases** — manual download from Kaggle (required for execution accuracy)

### Part A — Auto-download questions and SQL

```bash
python scripts/download_data.py --dataset spider
```

This creates `data/spider/dev.json` (1,034 examples) and `data/spider/train_spider.json`.

### Part B — SQLite databases (needed for execution accuracy)

1. Go to [kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset)
2. Click **Download** (free Kaggle account required)
3. Extract the zip
4. Copy **everything** from inside that extracted folder into `data/spider/`:

```
data/spider/
├── database/             ← 166 SQLite files (from Kaggle zip)
│   ├── college_2/
│   │   └── college_2.sqlite
│   ├── flights/
│   └── ...
├── dev.json              ← from Step A
├── train_spider.json     ← from Step A
└── tables.json           ← from Kaggle zip
```

> **Without the databases:** exact-match accuracy still works. Execution accuracy will be skipped.

---

## Step 4 — Run a Single Query

Make sure your venv is activated and (if using Ollama) Ollama is running.

```bash
# Basic LRG query
python scripts/run_query.py --db college_2 --question "How many students are enrolled in each department?"

# Save the LRG graph as a PNG image
python scripts/run_query.py --db college_2 --question "How many students are enrolled in each department?" --save-lrg lrg.png

# Use the direct-prompt baseline (no LRG)
python scripts/run_query.py --db college_2 --question "List all students" --model prompt_zero_shot

# Print the full LRG JSON
python scripts/run_query.py --db college_2 --question "..." --show-lrg-json
```

**Expected output:**
```
Model  : lrg
DB     : college_2
Question: How many students are enrolled in each department?

=== Generated SQL ===
SELECT COUNT(student.*) AS cnt, student.dept_name
FROM student
GROUP BY student.dept_name

=== LRG Summary ===
LRGGraph: 4 nodes, 3 edges
  entity: 1
  attribute: 1
  aggregation: 1
  grouping: 1

LRG image saved to lrg.png
```

---

## Step 5 — Run a Benchmark

```bash
# Quick run — first 20 examples, LRG model
python scripts/run_benchmark.py --dataset spider --model lrg --max-examples 20

# Compare the direct-prompt baseline
python scripts/run_benchmark.py --dataset spider --model prompt_zero_shot --max-examples 20

# Full dev set (1034 examples)
python scripts/run_benchmark.py --dataset spider --model lrg
```

Results are saved to `results/<run_id>/report.json` and `results/<run_id>/report.md`.

---

## Step 6 — Start the API

```bash
uvicorn src.api.app:app --reload --port 8000
```

Open **[localhost:8000/docs](http://localhost:8000/docs)** for the interactive Swagger UI.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Convert NL question → SQL |
| `GET` | `/schema/{db_id}` | View a database schema |
| `POST` | `/benchmark/run` | Start a background benchmark |
| `GET` | `/benchmark/status/{run_id}` | Check benchmark progress |

Quick test:
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
| **Query** | Type a question, pick a database, see SQL + LRG graph side-by-side |
| **Benchmark** | Run evaluations in-browser, view charts |
| **Analysis** | Load saved reports, filter failure modes, explore per-example results |

---

## Troubleshooting

### `ollama` not recognized in PowerShell
Ollama is installed but not in this session's PATH. Run:
```powershell
$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama list
```
To make this permanent, add Ollama's path to your system PATH in Windows Settings → System → Advanced system settings → Environment Variables.

### Ollama — `bind: Only one usage of each socket address`
This means Ollama is **already running** in the background (it auto-starts on Windows). This is normal — just proceed to use it.

### Ollama — LRG produces empty output / no entities extracted
This almost always means the model doesn't follow JSON instructions. Make sure you're using `llama3.2:3b` (or `mistral:7b` / `qwen2.5:7b`), **not** `sqlcoder:7b`. Verify your `.env` has `OLLAMA_MODEL=llama3.2:3b`.

### `Cannot find schema for db_id='...'`
The SQLite databases are not in `data/spider/database/`. Follow **Step 3 Part B** to download from Kaggle.

### `429 RESOURCE_EXHAUSTED` (Gemini rate limit)
You've hit the free tier per-minute limit. The script auto-retries after 62 seconds. For benchmarks, use `--max-examples 20` to stay within limits.

### `models/gemini-X not found` (404 error)
Your model name is wrong. Run this to see available models:
```python
from dotenv import load_dotenv; load_dotenv('.env')
import os
from google import genai
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
for m in client.models.list():
    print(m.name)
```
Then update `GEMINI_MODEL=` in `.env`.

### `source` not recognized on Windows
Use `.venv\Scripts\Activate.ps1` in PowerShell, not `source .venv/Scripts/activate` (that's bash syntax).

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
text2sql-lrg/
├── .env.example          ← copy to .env and configure your backend
├── .env                  ← your credentials (gitignored)
├── configs/
│   ├── default.yaml      ← model, paths, benchmark settings
│   ├── gemini.yaml       ← Gemini-specific overrides
│   └── ollama.yaml       ← Ollama-specific overrides
├── data/                 ← datasets live here (gitignored)
│   ├── spider/
│   │   ├── dev.json               (auto-downloaded)
│   │   ├── train_spider.json      (auto-downloaded)
│   │   ├── tables.json            (from Kaggle zip)
│   │   └── database/              (from Kaggle zip — SQLite files)
│   └── custom/           ← drop your own JSONL + SQLite here
├── docs/                 ← extended documentation
├── results/              ← benchmark outputs (gitignored)
├── scripts/
│   ├── download_data.py  ← run once to fetch Spider/CoSQL
│   ├── run_query.py      ← test a single NL question
│   ├── run_benchmark.py  ← full evaluation run
│   └── debug_llm.py      ← debug LLM connection + response format
├── src/
│   ├── config.py         ← loads .env + YAML into typed config
│   ├── llm/
│   │   ├── base.py       ← BaseLLM abstract interface
│   │   ├── gemini.py     ← Gemini backend (google-genai SDK)
│   │   ├── ollama.py     ← Ollama backend (local inference)
│   │   └── factory.py    ← creates LLM from config
│   ├── schema/
│   │   ├── parser.py     ← SQLite + Spider JSON → SchemaInfo
│   │   └── graph.py      ← SchemaInfo → NetworkX FK graph
│   ├── baseline/
│   │   ├── base.py       ← BaseText2SQL ABC
│   │   ├── prompt.py     ← zero-shot / few-shot baselines
│   │   └── registry.py   ← model name → class mapping
│   ├── lrg/
│   │   ├── nodes.py      ← all LRG node/edge types (Pydantic)
│   │   ├── graph.py      ← LRGGraph DiGraph wrapper + validation
│   │   ├── builder.py    ← NL → LRG via structured LLM call
│   │   ├── synthesizer.py← LRG → SQL (deterministic, no LLM)
│   │   ├── visualizer.py ← renders LRG as PNG
│   │   └── pipeline.py   ← LRGText2SQL end-to-end wrapper
│   ├── evaluation/
│   │   ├── metrics.py    ← EM, EX, component-level scores
│   │   ├── failure_modes.py ← 5 failure category classifiers
│   │   └── explainability.py← faithfulness, traceability
│   ├── benchmark/
│   │   ├── datasets/     ← Spider, CoSQL, Custom dataset loaders
│   │   ├── runner.py     ← orchestrates benchmark evaluation
│   │   └── reporter.py   ← saves JSON + Markdown reports
│   └── api/              ← FastAPI app + routers
├── tests/                ← offline pytest suite
└── ui/                   ← Streamlit web interface
    └── pages/
        ├── 1_Query.py    ← interactive NL→SQL + LRG viz
        ├── 2_Benchmark.py← run + view benchmarks
        └── 3_Analysis.py ← failure mode charts
```
