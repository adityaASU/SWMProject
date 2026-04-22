"""Ollama local LLM backend."""
from __future__ import annotations

import json
from typing import Any

import ollama as _ollama

from src.llm.base import BaseLLM, _extract_json


class OllamaLLM(BaseLLM):
    """Wrapper around the Ollama local inference server."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._client = _ollama.Client(host=base_url)
        self._options = {"temperature": temperature, "num_predict": max_tokens}

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, prompt: str) -> str:
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
            options=self._options,
        )
        return response["response"].strip()

    def generate_structured(self, prompt: str, response_schema: dict[str, Any]) -> dict:
        """Use Ollama's JSON format option for structured output."""
        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                format="json",
                options=self._options,
            )
            return json.loads(response["response"])
        except Exception:
            return super().generate_structured(prompt, response_schema)
