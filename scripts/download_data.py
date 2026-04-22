#!/usr/bin/env python
"""Download Spider and CoSQL datasets into the data/ directory.

Spider (questions + SQL):
  - Downloaded automatically via HuggingFace Hub (parquet -> JSON conversion)
  - SQLite databases must be added manually for execution accuracy (see below)

CoSQL: same approach via HuggingFace Hub.

Usage:
    python scripts/download_data.py                  # downloads both
    python scripts/download_data.py --dataset spider
    python scripts/download_data.py --dataset cosql
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pip_install(*pkgs: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure(*pkgs: str) -> None:
    import importlib
    missing = []
    for pkg in pkgs:
        try:
            importlib.import_module(pkg.replace("-", "_").split(">=")[0])
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  Installing: {', '.join(missing)} ...")
        _pip_install(*missing)


# ── Spider ─────────────────────────────────────────────────────────────────────

def download_spider(data_dir: Path) -> None:
    print("\n=== Spider Dataset ===")
    spider_dir = data_dir / "spider"

    # Already done?
    if (spider_dir / "dev.json").exists() and (spider_dir / "train_spider.json").exists():
        print("Spider JSON splits already present.")
        _print_spider_db_status(spider_dir)
        return

    # Convert from parquet files if they exist (HF already ran)
    parquet_val = spider_dir / "spider" / "validation-00000-of-00001.parquet"
    parquet_train = spider_dir / "spider" / "train-00000-of-00001.parquet"

    if parquet_val.exists() and parquet_train.exists():
        print("Found HuggingFace parquet files — converting to JSON...")
        _parquet_to_json(parquet_val, spider_dir / "dev.json", split="dev")
        _parquet_to_json(parquet_train, spider_dir / "train_spider.json", split="train")
        print("Conversion complete.")
        _print_spider_db_status(spider_dir)
        return

    # Fresh download via datasets library
    print("Downloading Spider via HuggingFace datasets library...")
    _ensure("datasets", "pyarrow")
    try:
        from datasets import load_dataset
        ds = load_dataset("spider", trust_remote_code=True)
        spider_dir.mkdir(parents=True, exist_ok=True)

        dev_records = _hf_split_to_records(ds["validation"])
        train_records = _hf_split_to_records(ds["train"])

        (spider_dir / "dev.json").write_text(json.dumps(dev_records, indent=2), encoding="utf-8")
        (spider_dir / "train_spider.json").write_text(json.dumps(train_records, indent=2), encoding="utf-8")

        print(f"  dev split   : {len(dev_records)} examples -> data/spider/dev.json")
        print(f"  train split : {len(train_records)} examples -> data/spider/train_spider.json")
        _print_spider_db_status(spider_dir)
    except Exception as exc:
        print(f"  Download failed: {exc}")
        _print_manual_spider_instructions()


def _hf_split_to_records(split) -> list[dict]:
    return [
        {
            "question": ex["question"],
            "db_id": ex["db_id"],
            "query": ex["query"],
            "query_toks_no_value": ex.get("query_toks_no_value", []),
        }
        for ex in split
    ]


def _parquet_to_json(parquet_path: Path, out_path: Path, split: str) -> None:
    _ensure("pyarrow", "pandas")
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "question": row.get("question", ""),
            "db_id": row.get("db_id", ""),
            "query": row.get("query", ""),
            "query_toks_no_value": list(row.get("query_toks_no_value", [])),
        })
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"  {split}: {len(records)} examples -> {out_path.name}")


def _download_tables_json(spider_dir: Path) -> None:
    """Download Spider tables.json (schema for all 166 databases) from GitHub."""
    out = spider_dir / "tables.json"
    if out.exists():
        return
    url = "https://raw.githubusercontent.com/taoyds/spider/master/tables.json"
    print(f"  Downloading tables.json (schema definitions) from GitHub...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, out)
        with open(out) as f:
            n = len(json.load(f))
        print(f"  tables.json: {n} database schemas saved.")
    except Exception as exc:
        print(f"  tables.json download failed: {exc}")
        print("  Schema-based queries will not work without tables.json.")


def _print_spider_db_status(spider_dir: Path) -> None:
    db_dir = spider_dir / "database"
    if db_dir.exists() and any(db_dir.iterdir()):
        n = sum(1 for _ in db_dir.iterdir() if _.is_dir())
        print(f"  SQLite databases: {n} found in data/spider/database/ — execution accuracy ENABLED")
    else:
        print("""
  SQLite databases: NOT FOUND — execution accuracy will be DISABLED.
  To enable full evaluation, add the database files:

  Option A — Kaggle (free account required):
    https://www.kaggle.com/datasets/jeromeblanchet/yale-spider-10-text2sql-benchmark
    Download, extract, copy the 'database/' folder to: data/spider/database/

  Option B — Yale form (official):
    https://yale-lily.github.io/spider
    Fill the form, download the zip, extract 'database/' to: data/spider/database/

  Exact-match benchmarking works without the databases.
""")


def _print_manual_spider_instructions() -> None:
    print("""
  MANUAL STEPS:
    1. Kaggle (easiest): https://www.kaggle.com/datasets/jeromeblanchet/yale-spider-10-text2sql-benchmark
       Download -> extract -> place 'dev.json', 'train_spider.json', 'database/' in data/spider/
    2. Yale official: https://yale-lily.github.io/spider
""")


# ── CoSQL ──────────────────────────────────────────────────────────────────────

def download_cosql(data_dir: Path) -> None:
    print("\n=== CoSQL Dataset ===")
    cosql_dir = data_dir / "cosql"

    candidates = [
        cosql_dir / "cosql_dev.json",
        cosql_dir / "cosql_all_info" / "cosql_dev.json",
        cosql_dir / "data" / "cosql_dev.json",
    ]
    if any(p.exists() for p in candidates):
        print("CoSQL already present.")
        return

    print("Downloading CoSQL via HuggingFace datasets library...")
    _ensure("datasets")
    try:
        from datasets import load_dataset
        ds = load_dataset("cosql_dataset", trust_remote_code=True)
        cosql_dir.mkdir(parents=True, exist_ok=True)

        for split_name, hf_key in [("dev", "validation"), ("train", "train")]:
            if hf_key not in ds:
                continue
            records = [dict(ex) for ex in ds[hf_key]]
            out = cosql_dir / f"cosql_{split_name}.json"
            out.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"  {split_name}: {len(records)} examples -> {out.name}")

        print("""
  CoSQL SQLite databases: same databases as Spider.
  If you already placed them in data/spider/database/, point CoSQL at that folder
  or symlink: data/cosql/database -> data/spider/database
""")
    except Exception as exc:
        print(f"  Download failed: {exc}")
        print("  Manual: https://yale-lily.github.io/cosql")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Download Text2SQL benchmark datasets")
    parser.add_argument(
        "--dataset", choices=["spider", "cosql", "both"], default="both",
        help="Which dataset to download (default: both)",
    )
    parser.add_argument(
        "--data-dir", default=str(_ROOT / "data"),
        help="Target directory (default: data/)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")

    if args.dataset in ("spider", "both"):
        download_spider(data_dir)
    if args.dataset in ("cosql", "both"):
        download_cosql(data_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
