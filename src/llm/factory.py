"""Factory that creates the correct LLM backend from config."""
from __future__ import annotations

from src.llm.base import BaseLLM


def create_llm(config=None) -> BaseLLM:
    """Instantiate and return an LLM backend based on *config*.

    If *config* is None, the global AppConfig is loaded automatically.
    """
    if config is None:
        from src.config import load_config

        config = load_config().llm

    backend = config.backend.lower()

    if backend == "gemini":
        from src.llm.gemini import GeminiLLM

        if not config.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file or set "
                "the environment variable before running."
            )
        return GeminiLLM(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if backend == "ollama":
        from src.llm.ollama import OllamaLLM

        return OllamaLLM(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ValueError(
        f"Unknown LLM backend '{backend}'. Choose 'gemini' or 'ollama' "
        "in your .env (LLM_BACKEND=...) or configs/default.yaml."
    )
