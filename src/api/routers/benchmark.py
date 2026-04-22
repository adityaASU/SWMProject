"""POST /benchmark/run — run a benchmark asynchronously."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# Track running / completed runs in memory (production should use a DB)
_runs: dict[str, dict] = {}


class BenchmarkRequest(BaseModel):
    dataset: str = "spider"       # "spider" | "cosql" | "custom"
    model: str = "lrg"
    split: str = "dev"
    max_examples: Optional[int] = 50
    custom_jsonl: Optional[str] = None
    data_dir: Optional[str] = None


class BenchmarkStatusResponse(BaseModel):
    run_id: str
    status: str
    report: Optional[dict] = None
    json_path: Optional[str] = None
    md_path: Optional[str] = None


@router.post("/run", response_model=BenchmarkStatusResponse)
def run_benchmark(req: BenchmarkRequest, background: BackgroundTasks):
    import uuid
    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {"status": "pending"}
    background.add_task(_execute_benchmark, run_id, req)
    return BenchmarkStatusResponse(run_id=run_id, status="pending")


@router.get("/status/{run_id}", response_model=BenchmarkStatusResponse)
def benchmark_status(run_id: str):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    entry = _runs[run_id]
    return BenchmarkStatusResponse(
        run_id=run_id,
        status=entry["status"],
        report=entry.get("report"),
        json_path=entry.get("json_path"),
        md_path=entry.get("md_path"),
    )


@router.get("/results", summary="List all completed runs")
def list_results():
    return [
        {"run_id": rid, "status": info["status"]}
        for rid, info in _runs.items()
    ]


def _execute_benchmark(run_id: str, req: BenchmarkRequest) -> None:
    """Background task that runs the benchmark."""
    try:
        _runs[run_id]["status"] = "running"

        from src.config import load_config
        from src.llm.factory import create_llm
        from src.baseline.registry import create_baseline
        from src.benchmark.runner import BenchmarkRunner
        from src.benchmark.reporter import save_report

        cfg = load_config()
        llm = create_llm(cfg.llm)
        model = create_baseline(req.model, llm)

        data_dir = Path(req.data_dir) if req.data_dir else None

        if req.dataset == "spider":
            from src.benchmark.datasets.spider import SpiderDataset
            base = data_dir or cfg.paths.spider_data
            dataset = SpiderDataset(base, split=req.split)
        elif req.dataset == "cosql":
            from src.benchmark.datasets.cosql import CoSQLDataset
            base = data_dir or cfg.paths.cosql_data
            dataset = CoSQLDataset(base, split=req.split)
        elif req.dataset == "custom":
            from src.benchmark.datasets.custom import CustomDataset
            if not req.custom_jsonl:
                raise ValueError("custom_jsonl path required for custom dataset")
            db_dir = data_dir or cfg.paths.custom_data
            dataset = CustomDataset(Path(req.custom_jsonl), db_dir)
        else:
            raise ValueError(f"Unknown dataset '{req.dataset}'")

        runner = BenchmarkRunner(
            model=model,
            dataset=dataset,
            config=cfg,
            max_examples=req.max_examples,
        )
        report = runner.run()
        json_path, md_path = save_report(report, cfg.paths.results_dir)

        import dataclasses
        report_dict = {
            "run_id": report.run_id,
            "model": report.model_name,
            "dataset": report.dataset_name,
            "n_examples": report.n_examples,
            "exact_match": report.exact_match,
            "execution_accuracy": report.execution_accuracy,
            "failure_summary": report.failure_summary,
            "explainability_summary": report.explainability_summary,
        }
        _runs[run_id].update({
            "status": "completed",
            "report": report_dict,
            "json_path": str(json_path),
            "md_path": str(md_path),
        })

    except Exception as exc:
        _runs[run_id].update({"status": "failed", "error": str(exc)})
