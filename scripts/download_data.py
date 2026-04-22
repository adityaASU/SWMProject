#!/usr/bin/env python
"""Download Spider and CoSQL datasets into the data/ directory.

Usage:
    python scripts/download_data.py                  # downloads both
    python scripts/download_data.py --dataset spider
    python scripts/download_data.py --dataset cosql
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

_ROOT = Path(__file__).parent.parent

SPIDER_URL = "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6C&export=download&confirm=t"
COSQL_URL = "https://drive.usercontent.google.com/download?id=1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF&export=download&confirm=t"


def download(url: str, dest: Path, name: str) -> Path:
    print(f"Downloading {name}...")
    zip_path = dest / f"{name}.zip"
    urlretrieve(url, zip_path, reporthook=_progress)
    print()
    return zip_path


def extract(zip_path: Path, dest: Path) -> None:
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()
    print(f"Extracted to {dest}")


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        pct = block_num * block_size * 100 / total_size
        sys.stdout.write(f"\r  {min(pct, 100):.1f}%")
        sys.stdout.flush()


def download_spider(data_dir: Path) -> None:
    spider_dir = data_dir / "spider"
    if (spider_dir / "dev.json").exists():
        print("Spider already downloaded.")
        return
    spider_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download(SPIDER_URL, data_dir, "spider")
    extract(zip_path, data_dir)
    print("Spider download complete.")


def download_cosql(data_dir: Path) -> None:
    cosql_dir = data_dir / "cosql"
    if (cosql_dir / "cosql_all_info").exists():
        print("CoSQL already downloaded.")
        return
    cosql_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download(COSQL_URL, data_dir, "cosql")
    extract(zip_path, data_dir)
    print("CoSQL download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Text2SQL benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["spider", "cosql", "both"],
        default="both",
        help="Which dataset to download (default: both)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(_ROOT / "data"),
        help="Target directory (default: data/)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print(f"Data directory: {data_dir}")

    if args.dataset in ("spider", "both"):
        try:
            download_spider(data_dir)
        except Exception as exc:
            print(f"\nSpider download failed: {exc}")
            print("Manual download: https://yale-lily.github.io/spider")

    if args.dataset in ("cosql", "both"):
        try:
            download_cosql(data_dir)
        except Exception as exc:
            print(f"\nCoSQL download failed: {exc}")
            print("Manual download: https://yale-lily.github.io/cosql")


if __name__ == "__main__":
    main()
