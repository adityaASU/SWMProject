"""Spider dataset loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.benchmark.datasets.base import BaseDataset, Example


class SpiderDataset(BaseDataset):
    """Loads the Spider Text2SQL benchmark.

    Expected directory layout (standard Spider release):
        <root>/
            train_spider.json
            dev.json
            tables.json
            database/
                <db_id>/
                    <db_id>.sqlite
    """

    name = "spider"

    def __init__(self, root: Path, split: str = "dev") -> None:
        """
        Args:
            root: Path to the Spider dataset root directory.
            split: "train" | "dev"
        """
        self._root = Path(root)
        self._split = split
        self._examples: list[Example] = []

    @property
    def db_dir(self) -> Path:
        return self._root / "database"

    def load(self) -> None:
        split_file = "dev.json" if self._split == "dev" else "train_spider.json"
        data_path = self._root / split_file
        if not data_path.exists():
            raise FileNotFoundError(
                f"Spider {self._split} file not found at {data_path}. "
                "Run: python scripts/download_data.py --dataset spider"
            )

        with open(data_path) as f:
            raw = json.load(f)

        self._examples = []
        for idx, item in enumerate(raw):
            ex = Example(
                question=item["question"],
                db_id=item["db_id"],
                gold_sql=item.get("query", item.get("sql", "")),
                example_id=str(idx),
                metadata={"difficulty": item.get("query_toks_no_value", [])},
            )
            self._examples.append(ex)

    def __iter__(self) -> Iterator[Example]:
        if not self._examples:
            self.load()
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)
