"""Custom dataset loader for user-provided JSONL files.

JSONL format — each line must be a JSON object with:
  Required: "question", "db_id", "gold_sql"
  Optional: "example_id", "conversation_history", "metadata"

Example line:
  {"question": "How many students are enrolled?", "db_id": "university", "gold_sql": "SELECT COUNT(*) FROM students"}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.benchmark.datasets.base import BaseDataset, Example


class CustomDataset(BaseDataset):
    """Load a user-provided JSONL file as a benchmark dataset."""

    name = "custom"

    def __init__(self, jsonl_path: Path, db_dir: Path) -> None:
        """
        Args:
            jsonl_path: Path to the JSONL file containing examples.
            db_dir: Directory containing per-db SQLite files.
        """
        self._jsonl_path = Path(jsonl_path)
        self._db_dir = Path(db_dir)
        self._examples: list[Example] = []

    @property
    def db_dir(self) -> Path:
        return self._db_dir

    def load(self) -> None:
        if not self._jsonl_path.exists():
            raise FileNotFoundError(f"Custom dataset file not found: {self._jsonl_path}")

        self._examples = []
        with open(self._jsonl_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {idx + 1}: {exc}") from exc

                required = ("question", "db_id", "gold_sql")
                missing = [k for k in required if k not in item]
                if missing:
                    raise ValueError(f"Line {idx + 1} missing required keys: {missing}")

                ex = Example(
                    question=item["question"],
                    db_id=item["db_id"],
                    gold_sql=item["gold_sql"],
                    example_id=item.get("example_id", str(idx)),
                    conversation_history=item.get("conversation_history"),
                    metadata=item.get("metadata", {}),
                )
                self._examples.append(ex)

    def __iter__(self) -> Iterator[Example]:
        if not self._examples:
            self.load()
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)
