"""Baseline registry — maps string names to factory functions.

To add a new baseline:
1. Create a class in src/baseline/ that extends BaseText2SQL.
2. Register it here with a unique key.
"""
from __future__ import annotations

from typing import Callable

from src.baseline.base import BaseText2SQL
from src.llm.base import BaseLLM


def _make_prompt_baseline(llm: BaseLLM, **kwargs) -> BaseText2SQL:
    from src.baseline.prompt_baseline import PromptBaseline

    return PromptBaseline(llm=llm, **kwargs)


def _make_lrg_baseline(llm: BaseLLM, **kwargs) -> BaseText2SQL:
    from src.lrg.pipeline import LRGText2SQL

    return LRGText2SQL(llm=llm, **kwargs)


# Registry maps name -> factory(llm, **kwargs) -> BaseText2SQL
_REGISTRY: dict[str, Callable[..., BaseText2SQL]] = {
    "prompt_zero_shot": lambda llm, **kw: _make_prompt_baseline(llm, use_few_shot=False, **kw),
    "prompt_few_shot": lambda llm, **kw: _make_prompt_baseline(llm, use_few_shot=True, **kw),
    "lrg": _make_lrg_baseline,
}


def list_baselines() -> list[str]:
    """Return all registered baseline names."""
    return list(_REGISTRY.keys())


def create_baseline(name: str, llm: BaseLLM, **kwargs) -> BaseText2SQL:
    """Instantiate a baseline by *name* using the given *llm*.

    Args:
        name: One of the keys returned by :func:`list_baselines`.
        llm: LLM backend instance.
        **kwargs: Extra keyword arguments forwarded to the factory.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(f"Unknown baseline '{name}'. Available: {available}")
    return _REGISTRY[name](llm, **kwargs)
