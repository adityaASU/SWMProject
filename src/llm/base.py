"""Abstract base class for all LLM backends."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Common interface every LLM backend must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send *prompt* and return the raw text response."""

    def generate_structured(self, prompt: str, response_schema: dict[str, Any]) -> dict:
        """Send *prompt* and parse the response as JSON matching *response_schema*.

        The default implementation wraps *generate* and handles JSON extraction.
        Backends can override this for native structured-output support.
        """
        instruction = (
            "\n\nRespond ONLY with a valid JSON object that matches this schema:\n"
            + json.dumps(response_schema, indent=2)
            + "\nDo NOT include markdown fences or extra text."
        )
        raw = self.generate(prompt + instruction)
        return _extract_json(raw)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the backend + model."""


def _extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from a model response."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")
