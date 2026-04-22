"""CoSQL dataset loader (multi-turn conversational Text2SQL)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.benchmark.datasets.base import BaseDataset, Example


class CoSQLDataset(BaseDataset):
    """Loads the CoSQL conversational Text2SQL benchmark.

    Expected directory layout (standard CoSQL release):
        <root>/
            cosql_all_info/
                cosql_dev.json   (or cosql_train.json)
            database/
                <db_id>/
                    <db_id>.sqlite
    """

    name = "cosql"

    def __init__(self, root: Path, split: str = "dev") -> None:
        self._root = Path(root)
        self._split = split
        self._examples: list[Example] = []

    @property
    def db_dir(self) -> Path:
        return self._root / "database"

    def load(self) -> None:
        fname = f"cosql_{self._split}.json"
        candidates = [
            self._root / fname,
            self._root / "cosql_all_info" / fname,
            self._root / "data" / fname,
        ]
        data_path = next((p for p in candidates if p.exists()), None)
        if data_path is None:
            raise FileNotFoundError(
                f"CoSQL {self._split} file not found. "
                "Run: python scripts/download_data.py --dataset cosql"
            )

        with open(data_path) as f:
            raw = json.load(f)

        self._examples = []
        for dialogue_idx, dialogue in enumerate(raw):
            db_id = dialogue["database_id"]
            history: list[dict] = []

            for turn_idx, turn in enumerate(dialogue["interaction"]):
                question = turn["utterance"]
                gold_sql = turn.get("query", "")

                ex = Example(
                    question=question,
                    db_id=db_id,
                    gold_sql=gold_sql,
                    example_id=f"{dialogue_idx}_{turn_idx}",
                    conversation_history=list(history),
                    metadata={"dialogue_id": dialogue_idx, "turn": turn_idx},
                )
                self._examples.append(ex)
                history.append({"question": question, "sql": gold_sql})

    def __iter__(self) -> Iterator[Example]:
        if not self._examples:
            self.load()
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)
