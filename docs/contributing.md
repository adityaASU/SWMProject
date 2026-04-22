# Contributing Guide

Welcome to the Text2SQL LRG project. This guide explains how to set up your development environment and contribute new features.

## Development Setup

```bash
git clone https://github.com/adityaASU/SWMProject.git
cd SWMProject/text2sql-lrg

python -m venv .venv

# Windows (PowerShell) — use Activate.ps1, NOT source
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then edit .env to set your backend
```

See the main [README](../README.md) for full backend setup instructions (Ollama or Gemini).

### LLM Backend Notes for Contributors

**Ollama (recommended for local dev):**
- Install from [ollama.com](https://ollama.com/) — it auto-starts on Windows
- Use `llama3.2:3b` (or `mistral:7b`, `qwen2.5:7b`) — these models follow JSON instructions correctly
- **Do not use `sqlcoder:7b`** for the LRG pipeline — it outputs raw SQL and ignores JSON format requests
- On Windows, Ollama may not be in PATH after a fresh install — add it: `$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"`
- If you see `bind: Only one usage of each socket address`, Ollama is already running — that's fine

**Gemini:**
- Free tier key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Use `gemini-2.0-flash-lite` for highest free-tier rate limits
- The LLM client has automatic retry with backoff for 429 rate limit errors

### Debugging LLM Issues

If your queries produce an empty LRG (`0 nodes, 0 edges`), run the debug script:
```bash
python scripts/debug_llm.py
```
This shows the exact schema loaded, the prompt sent, and the raw LLM response. The most common cause is the model returning a different JSON structure — fix it by updating the prompt in `src/lrg/builder.py`.

## Branch Strategy

- `main` — stable, always passing tests
- `feature/<name>` — new features
- `fix/<name>` — bug fixes
- `experiment/<name>` — research experiments (may not always pass tests)

Always branch from `main` and open a pull request when done.

## Running Tests

```bash
pytest tests/ -v
```

All tests run offline (no LLM calls required). Add tests for any new code you write.

## Code Style

- Python 3.11+, type hints everywhere
- Line length: 100 characters (configured in `pyproject.toml`)
- Use `ruff` for linting: `ruff check src/ tests/`
- No comments that just restate the code — only explain non-obvious intent

## How to Add a New Baseline Model

1. Create `src/baseline/<model_name>.py` with a class extending `BaseText2SQL`
2. Implement `model_name` property and `predict()` method
3. Register in `src/baseline/registry.py`
4. Add tests in `tests/test_baselines.py` (mock the LLM)
5. Document in `docs/contributing.md`

## How to Add a New LRG Node Type

1. Add a new Pydantic class to `src/lrg/nodes.py` extending `BaseNode`
2. Add it to the `LRGNode` union type and `node_from_dict()` dispatch dict
3. Add handling in `src/lrg/synthesizer.py` for the SQL translation
4. Add a colour in `src/lrg/visualizer.py` (`_NODE_COLORS` dict)
5. Add tests in `tests/test_lrg.py`

## How to Add a New Dataset

1. Create `src/benchmark/datasets/<dataset>.py` extending `BaseDataset`
2. Implement `load()`, `__iter__()`, `__len__()`, and `db_dir` property
3. Register in `src/benchmark/datasets/__init__.py`
4. Add a case in `scripts/run_benchmark.py` and `src/api/routers/benchmark.py`
5. Document the expected directory layout in the class docstring

## Experiment Tracking

After a benchmark run, results are saved to `results/<run_id>/report.json`. Use the Analysis page in the UI to compare runs side-by-side. For larger experiments, consider exporting results and using the JSON files directly with pandas or a notebook.

## Directory Conventions

| Directory | Purpose |
|-----------|---------|
| `src/` | All production source code |
| `tests/` | Pytest tests (mirror `src/` structure) |
| `scripts/` | CLI entry points (thin wrappers over `src/`) |
| `ui/` | Streamlit UI pages |
| `docs/` | Project documentation |
| `configs/` | YAML config files |
| `data/` | Downloaded datasets (gitignored) |
| `results/` | Benchmark outputs (gitignored) |

## Contact

Open a GitHub issue or discussion for questions. Tag teammates directly in PR reviews.
