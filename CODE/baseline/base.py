"""Abstract base class for all Text2SQL baseline implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.schema.parser import SchemaInfo


@dataclass
class PredictionResult:
    """Output of a Text2SQL prediction."""

    question: str
    db_id: str
    predicted_sql: str
    model_name: str
    raw_response: str = ""
    conversation_history: Optional[list[dict]] = None
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class BaseText2SQL(ABC):
    """Every Text2SQL model (baseline or LRG-enhanced) must implement this interface."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier for this model."""

    @abstractmethod
    def predict(
        self,
        question: str,
        schema: SchemaInfo,
        conversation_history: Optional[list[dict]] = None,
    ) -> PredictionResult:
        """Generate a SQL query for *question* given *schema*.

        Args:
            question: The natural language question.
            schema: Parsed database schema.
            conversation_history: Optional list of previous turns for CoSQL-style
                multi-turn dialogue. Each dict has keys 'question' and 'sql'.

        Returns:
            A PredictionResult with the predicted SQL.
        """
