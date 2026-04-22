"""Central configuration loaded from .env and configs/default.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_ROOT = Path(__file__).parent.parent


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    backend: Literal["gemini", "ollama"] = Field("gemini", alias="LLM_BACKEND")
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", alias="GEMINI_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3", alias="OLLAMA_MODEL")
    temperature: float = 0.0
    max_tokens: int = 2048


class PathConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    spider_data: Path = Field(_ROOT / "data" / "spider", alias="SPIDER_DATA_PATH")
    cosql_data: Path = Field(_ROOT / "data" / "cosql", alias="COSQL_DATA_PATH")
    custom_data: Path = Field(_ROOT / "data" / "custom")
    results_dir: Path = Field(_ROOT / "results", alias="RESULTS_DIR")


class BenchmarkConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    max_examples: Optional[int] = None
    timeout_seconds: int = 30
    parallel_workers: int = 1


class EvaluationConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    compute_exact_match: bool = True
    compute_execution_accuracy: bool = True
    compute_failure_modes: bool = True
    compute_explainability: bool = True


class AppConfig:
    """Merged configuration from YAML file + environment variables."""

    def __init__(self, yaml_path: Optional[Path] = None) -> None:
        yaml_path = yaml_path or (_ROOT / "configs" / "default.yaml")
        raw = _load_yaml(yaml_path)

        llm_raw = raw.get("llm", {})
        self.llm = LLMConfig(
            LLM_BACKEND=os.getenv("LLM_BACKEND", llm_raw.get("backend", "gemini")),
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
            GEMINI_MODEL=os.getenv("GEMINI_MODEL", llm_raw.get("gemini_model", "gemini-1.5-flash")),
            OLLAMA_BASE_URL=os.getenv(
                "OLLAMA_BASE_URL", llm_raw.get("ollama_base_url", "http://localhost:11434")
            ),
            OLLAMA_MODEL=os.getenv("OLLAMA_MODEL", llm_raw.get("ollama_model", "llama3")),
            temperature=llm_raw.get("temperature", 0.0),
            max_tokens=llm_raw.get("max_tokens", 2048),
        )

        path_raw = raw.get("paths", {})
        self.paths = PathConfig(
            SPIDER_DATA_PATH=os.getenv(
                "SPIDER_DATA_PATH", path_raw.get("spider_data", str(_ROOT / "data" / "spider"))
            ),
            COSQL_DATA_PATH=os.getenv(
                "COSQL_DATA_PATH", path_raw.get("cosql_data", str(_ROOT / "data" / "cosql"))
            ),
            RESULTS_DIR=os.getenv(
                "RESULTS_DIR", path_raw.get("results_dir", str(_ROOT / "results"))
            ),
        )

        bench_raw = raw.get("benchmark", {})
        self.benchmark = BenchmarkConfig(**bench_raw)

        eval_raw = raw.get("evaluation", {})
        self.evaluation = EvaluationConfig(**eval_raw)


def load_config(yaml_path: Optional[Path] = None) -> AppConfig:
    """Load and return application configuration."""
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=False)
    return AppConfig(yaml_path)
