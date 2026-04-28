"""Saves benchmark reports to JSON and Markdown."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.benchmark.runner import BenchmarkReport


def save_report(report: BenchmarkReport, results_dir: Path) -> tuple[Path, Path]:
    """Save *report* as both JSON and Markdown. Returns (json_path, md_path)."""
    results_dir = Path(results_dir)
    safe_name = report.model_name.replace("/", "_").replace(":", "-")
    run_dir = results_dir / f"{report.run_id}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"

    # JSON
    data = {
        "run_id": report.run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": report.model_name,
        "dataset": report.dataset_name,
        "n_examples": report.n_examples,
        "exact_match": report.exact_match,
        "execution_accuracy": report.execution_accuracy,
        "component_scores": report.component_scores,
        "failure_summary": report.failure_summary,
        "explainability_summary": report.explainability_summary,
        "elapsed_seconds": round(report.elapsed_seconds, 2),
        "per_example": report.per_example,
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Markdown
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    return json_path, md_path


def _to_markdown(r: BenchmarkReport) -> str:
    lines = [
        f"# Benchmark Report",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| Run ID | `{r.run_id}` |",
        f"| Model | `{r.model_name}` |",
        f"| Dataset | `{r.dataset_name}` |",
        f"| Examples | {r.n_examples} |",
        f"| Elapsed | {r.elapsed_seconds:.1f}s |",
        f"",
        f"## Accuracy Metrics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Exact Match | {_pct(r.exact_match)} |",
        f"| Execution Accuracy | {_pct(r.execution_accuracy)} |",
    ]

    if r.component_scores:
        lines += ["", "### Clause-Level Scores", "", "| Clause | Accuracy |", "|---|---|"]
        for clause, score in sorted(r.component_scores.items()):
            lines.append(f"| {clause} | {_pct(score)} |")

    if r.failure_summary:
        lines += ["", "## Failure Mode Analysis", "", "| Category | Count | Rate |", "|---|---|---|"]
        fs = r.failure_summary
        for cat in ["schema_linking", "join_hallucination", "nested_subquery", "self_join", "context_drift"]:
            if cat in fs:
                lines.append(f"| {cat} | {fs[cat]} | {_pct(fs.get(f'{cat}_rate'))} |")

    if r.explainability_summary:
        es = r.explainability_summary
        lines += [
            "",
            "## Explainability Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Mean Faithfulness | {_pct(es.get('mean_faithfulness'))} |",
            f"| Mean Completeness | {_pct(es.get('mean_completeness'))} |",
            f"| Error Traceability Rate | {_pct(es.get('error_traceability_rate'))} |",
        ]

    lines += ["", "## Per-Example Results (first 20)", "", "| ID | Question | EM | EX | Failures |", "|---|---|---|---|---|"]
    for ex in r.per_example[:20]:
        em = "✓" if ex["exact_match"] else "✗"
        ex_acc = "✓" if ex.get("execution_accuracy") else ("✗" if ex.get("execution_accuracy") is False else "-")
        failures = ", ".join(ex.get("failure_modes", [])) or "-"
        q = ex["question"][:60].replace("|", "\\|")
        lines.append(f"| {ex['example_id']} | {q} | {em} | {ex_acc} | {failures} |")

    if len(r.per_example) > 20:
        lines.append(f"| ... | _{len(r.per_example) - 20} more in JSON_ | | | |")

    return "\n".join(lines) + "\n"


def _pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%" if isinstance(v, float) and v <= 1.0 else str(v)
