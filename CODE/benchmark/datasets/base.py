"""Abstract base class for all benchmark datasets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class Example:
    """A single benchmark example."""

    question: str
    db_id: str
    gold_sql: str
    example_id: str = ""
    conversation_history: Optional[list[dict]] = None   # For CoSQL
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseDataset(ABC):
    """Every dataset must implement load() and __iter__."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load data from disk into memory."""

    @abstractmethod
    def __iter__(self) -> Iterator[Example]:
        """Yield Examples one at a time."""

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def db_dir(self) -> Path:
        """Root directory containing per-db SQLite files."""
