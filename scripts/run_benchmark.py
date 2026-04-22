#!/usr/bin/env python
"""CLI tool: run a full benchmark evaluation.

Usage:
    python scripts/run_benchmark.py --dataset spider --model lrg
    python scripts/run_benchmark.py --dataset cosql --model prompt_few_shot --max-examples 100
    python scripts/run_benchmark.py --dataset custom --custom-jsonl data/custom/my_data.jsonl --model lrg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Text2SQL benchmark evaluation")
    parser.add_argument(
        "--dataset",
        choices=["spider", "cosql", "custom"],
        default="spider",
        help="Dataset to evaluate on (default: spider)",
    )
    parser.add_argument(
        "--model",
        default="lrg",
        choices=["lrg", "prompt_zero_shot", "prompt_few_shot"],
        help="Model to evaluate (default: lrg)",
    )
    parser.add_argument("--split", default="dev", choices=["dev", "train"])
    parser.add_argument("--max-examples", type=int, default=None, help="Limit evaluation to N examples")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--custom-jsonl", default=None, help="Path to custom JSONL file")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--results-dir", default=None, help="Override results output directory")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")

    from src.config import load_config
    from src.llm.factory import create_llm
    from src.baseline.registry import create_baseline
    from src.benchmark.runner import BenchmarkRunner
    from src.benchmark.reporter import save_report

    cfg = load_config(Path(args.config) if args.config else None)
    results_dir = Path(args.results_dir) if args.results_dir else cfg.paths.results_dir

    print(f"Model   : {args.model}")
    print(f"Dataset : {args.dataset} ({args.split})")
    print(f"LLM     : {cfg.llm.backend} / {cfg.llm.gemini_model if cfg.llm.backend == 'gemini' else cfg.llm.ollama_model}")
    if args.max_examples:
        print(f"Limit   : {args.max_examples} examples")
    print()

    llm = create_llm(cfg.llm)
    model = create_baseline(args.model, llm)

    if args.dataset == "spider":
        from src.benchmark.datasets.spider import SpiderDataset
        base = Path(args.data_dir) if args.data_dir else cfg.paths.spider_data
        dataset = SpiderDataset(base, split=args.split)
    elif args.dataset == "cosql":
        from src.benchmark.datasets.cosql import CoSQLDataset
        base = Path(args.data_dir) if args.data_dir else cfg.paths.cosql_data
        dataset = CoSQLDataset(base, split=args.split)
    else:
        from src.benchmark.datasets.custom import CustomDataset
        if not args.custom_jsonl:
            print("Error: --custom-jsonl is required for the custom dataset.", file=sys.stderr)
            sys.exit(1)
        db_dir = Path(args.data_dir) if args.data_dir else cfg.paths.custom_data
        dataset = CustomDataset(Path(args.custom_jsonl), db_dir)

    runner = BenchmarkRunner(
        model=model,
        dataset=dataset,
        config=cfg,
        max_examples=args.max_examples,
    )

    report = runner.run()
    json_path, md_path = save_report(report, results_dir)

    print("\n" + "=" * 50)
    print(f"RESULTS — {report.model_name} on {report.dataset_name}")
    print("=" * 50)
    print(f"  Examples evaluated : {report.n_examples}")
    print(f"  Exact Match        : {(report.exact_match or 0)*100:.1f}%")
    print(f"  Execution Accuracy : {(report.execution_accuracy or 0)*100:.1f}%")

    if report.failure_summary:
        print("\n  Failure Modes:")
        for cat in ["schema_linking", "join_hallucination", "nested_subquery", "self_join", "context_drift"]:
            rate = report.failure_summary.get(f"{cat}_rate", 0) * 100
            count = report.failure_summary.get(cat, 0)
            print(f"    {cat:25s}: {count:3d} ({rate:.1f}%)")

    if report.explainability_summary:
        es = report.explainability_summary
        print("\n  Explainability:")
        print(f"    Faithfulness       : {es.get('mean_faithfulness', 0)*100:.1f}%")
        print(f"    Completeness       : {es.get('mean_completeness', 0)*100:.1f}%")
        print(f"    Error Traceability : {es.get('error_traceability_rate', 0)*100:.1f}%")

    print(f"\n  JSON report : {json_path}")
    print(f"  MD report   : {md_path}")
    print()


if __name__ == "__main__":
    main()
