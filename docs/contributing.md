# Contributing Guide

Welcome to the Text2SQL LRG project. This guide explains how to set up your development environment and contribute new features.

## Development Setup

```bash
git clone <repo-url>
cd text2sql-lrg
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env
# Fill in GEMINI_API_KEY or set LLM_BACKEND=ollama
```

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
