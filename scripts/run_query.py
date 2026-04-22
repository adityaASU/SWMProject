#!/usr/bin/env python
"""CLI tool: generate SQL for a single natural language question.

Usage:
    python scripts/run_query.py --db college_2 --question "How many students are enrolled?"
    python scripts/run_query.py --db flights --question "List all airports" --model prompt_few_shot
    python scripts/run_query.py --db college_2 --question "..." --save-lrg lrg.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Text2SQL prediction")
    parser.add_argument("--db", required=True, help="Database ID (e.g. college_2)")
    parser.add_argument("--question", "-q", required=True, help="Natural language question")
    parser.add_argument(
        "--model",
        default="lrg",
        choices=["lrg", "prompt_zero_shot", "prompt_few_shot"],
        help="Model to use (default: lrg)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(_ROOT / "data" / "spider"),
        help="Directory containing database folders",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--save-lrg", default=None, help="Save LRG visualisation to this PNG path")
    parser.add_argument("--show-lrg-json", action="store_true", help="Print LRG JSON to stdout")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")

    from src.config import load_config
    from src.llm.factory import create_llm
    from src.schema.parser import SchemaParser
    from src.baseline.registry import create_baseline

    cfg = load_config(Path(args.config) if args.config else None)
    llm = create_llm(cfg.llm)

    parser_obj = SchemaParser()
    data_dir = Path(args.data_dir)
    try:
        schema = parser_obj.auto_parse(data_dir, args.db)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nModel  : {args.model}")
    print(f"DB     : {args.db}")
    print(f"Question: {args.question}\n")

    if args.model == "lrg":
        from src.lrg.pipeline import LRGText2SQL
        model_obj = LRGText2SQL(llm)
        result, lrg, errors = model_obj.predict_with_lrg(args.question, schema)

        print("=== Generated SQL ===")
        print(result.predicted_sql)

        print("\n=== LRG Summary ===")
        print(lrg.summary())

        if errors:
            print("\n=== Validation Warnings ===")
            for e in errors:
                print(f"  - {e}")

        if args.save_lrg:
            from src.lrg.visualizer import render_lrg
            img_bytes = render_lrg(lrg, title=args.question[:80], output_path=Path(args.save_lrg))
            print(f"\nLRG image saved to {args.save_lrg}")

        if args.show_lrg_json:
            import json
            print("\n=== LRG JSON ===")
            print(json.dumps(lrg.to_dict(), indent=2))

    else:
        model_obj = create_baseline(args.model, llm)
        result = model_obj.predict(args.question, schema)
        print("=== Generated SQL ===")
        print(result.predicted_sql)

    print()


if __name__ == "__main__":
    main()
