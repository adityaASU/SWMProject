"""Gemini LLM backend using the google-genai SDK."""
from __future__ import annotations

import json
import time
import logging
from typing import Any

from google import genai
from google.genai import types

from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_RETRY_CODES = {429, 503}   # rate-limit / overloaded
_MAX_RETRIES = 3
_RETRY_DELAY = 62            # seconds (free tier resets per-minute)


class GeminiLLM(BaseLLM):
    """Wrapper around the Gemini API (google-genai SDK)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-lite",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        # Strip the "models/" prefix if present
        self._model_name = model_name.removeprefix("models/")
        self._base_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    @property
    def name(self) -> str:
        return f"gemini/{self._model_name}"

    def generate(self, prompt: str) -> str:
        return self._call_with_retry(prompt, self._base_config)

    def generate_structured(self, prompt: str, response_schema: dict[str, Any]) -> dict:
        """Use Gemini's native JSON response mode."""
        json_config = types.GenerateContentConfig(
            temperature=self._base_config.temperature,
            max_output_tokens=self._base_config.max_output_tokens,
            response_mime_type="application/json",
        )
        try:
            raw = self._call_with_retry(prompt, json_config)
            return json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return super().generate_structured(prompt, response_schema)

    def _call_with_retry(self, prompt: str, config: types.GenerateContentConfig) -> str:
        last_exc = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text.strip()
            except Exception as exc:
                last_exc = exc
                code = _http_code(exc)
                if code in _RETRY_CODES and attempt < _MAX_RETRIES:
                    wait = _RETRY_DELAY * attempt
                    logger.warning(
                        "Gemini rate limit (attempt %d/%d) — waiting %ds...",
                        attempt, _MAX_RETRIES, wait,
                    )
                    print(f"  [Gemini] Rate limited — waiting {wait}s before retry {attempt}/{_MAX_RETRIES - 1}...")
                    time.sleep(wait)
                else:
                    raise
        raise last_exc


def _http_code(exc: Exception) -> int:
    """Extract HTTP status code from a google-genai exception."""
    msg = str(exc)
    for code in _RETRY_CODES:
        if str(code) in msg:
            return code
    return 0
