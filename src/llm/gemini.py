"""Gemini LLM backend using google-generativeai."""
from __future__ import annotations

from typing import Any

import google.generativeai as genai

from src.llm.base import BaseLLM, _extract_json


class GeminiLLM(BaseLLM):
    """Wrapper around the Gemini generative AI API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        genai.configure(api_key=api_key)
        self._model_name = model_name
        self._generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self._generation_config,
        )

    @property
    def name(self) -> str:
        return f"gemini/{self._model_name}"

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text.strip()

    def generate_structured(self, prompt: str, response_schema: dict[str, Any]) -> dict:
        """Use Gemini's JSON response mode when available, fall back to text parsing."""
        import json

        try:
            json_model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self._generation_config.temperature,
                    max_output_tokens=self._generation_config.max_output_tokens,
                    response_mime_type="application/json",
                ),
            )
            response = json_model.generate_content(prompt)
            return json.loads(response.text)
        except Exception:
            return super().generate_structured(prompt, response_schema)
