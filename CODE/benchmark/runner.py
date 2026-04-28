"""BenchmarkRunner: orchestrates prediction + evaluation over a dataset."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.baseline.base import BaseText2SQL, PredictionResult
from src.benchmark.datasets.base import BaseDataset, Example
from src.evaluation.metrics import (
    aggregate_metrics,
    component_match,
    exact_match,
    execution_accuracy,
)
from src.evaluation.failure_modes import analyse_failure_modes, summarise_failures
from src.evaluation.explainability import aggregate_explainability, evaluate_explainability
from src.schema.graph import SchemaGraph
from src.schema.parser import SchemaParser

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
    example: Example
    prediction: PredictionResult
    exact_match: bool
    execution_accuracy: Optional[bool]
    component_scores: dict
    failure_mode: object = None  # FailureModeResult
    explainability: object = None  # ExplainabilityResult
    elapsed_ms: float = 0.0


@dataclass
class BenchmarkReport:
    run_id: str
    model_name: str
    dataset_name: str
    n_examples: int
    exact_match: Optional[float]
    execution_accuracy: Optional[float]
    component_scores: dict
    failure_summary: dict
    explainability_summary: dict
    per_example: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class BenchmarkRunner:
    """Runs any BaseText2SQL model on any BaseDataset and computes all metrics."""

    def __init__(
        self,
        model: BaseText2SQL,
        dataset: BaseDataset,
        config=None,
        max_examples: Optional[int] = None,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._config = config
        self._max_examples = max_examples
        self._parser = SchemaParser()

    def run(self) -> BenchmarkReport:
        """Execute the full benchmark and return a report."""
        run_id = str(uuid.uuid4())[:8]
        logger.info(
            "Starting benchmark run=%s model=%s dataset=%s",
            run_id, self._model.model_name, self._dataset.name,
        )

        start_time = time.time()
        example_results: list[ExampleResult] = []
        schema_cache: dict[str, object] = {}

        for idx, example in enumerate(self._dataset):
            if self._max_examples and idx >= self._max_examples:
                break

            logger.info("[%d] db=%s q=%s...", idx + 1, example.db_id, example.question[:60])

            # Load schema (cached per db_id)
            if example.db_id not in schema_cache:
                try:
                    schema = self._parser.auto_parse(self._dataset.db_dir, example.db_id)
                    schema_cache[example.db_id] = schema
                except FileNotFoundError as exc:
                    logger.warning("Schema not found for %s: %s", example.db_id, exc)
                    continue
            schema = schema_cache[example.db_id]
            schema_graph = SchemaGraph(schema)

            # Predict
            t0 = time.time()
            try:
                prediction = self._model.predict(
                    question=example.question,
                    schema=schema,
                    conversation_history=example.conversation_history,
                )
            except Exception as exc:
                logger.error("Prediction failed for example %s: %s", example.example_id, exc)
                prediction = PredictionResult(
                    question=example.question,
                    db_id=example.db_id,
                    predicted_sql=f"-- ERROR: {exc}",
                    model_name=self._model.model_name,
                )
            elapsed_ms = (time.time() - t0) * 1000

            # Exact match
            em = exact_match(prediction.predicted_sql, example.gold_sql)

            # Execution accuracy
            db_path = self._dataset.db_dir / example.db_id / f"{example.db_id}.sqlite"
            ex_acc: Optional[bool] = None
            if db_path.exists():
                timeout = 30
                if self._config and hasattr(self._config, "benchmark"):
                    timeout = self._config.benchmark.timeout_seconds
                ex_acc = execution_accuracy(
                    prediction.predicted_sql, example.gold_sql, db_path, timeout
                )

            # Component scores
            comp = component_match(prediction.predicted_sql, example.gold_sql)

            # Failure mode analysis
            fm = analyse_failure_modes(
                prediction.predicted_sql,
                example.gold_sql,
                schema,
                schema_graph,
                is_correct=em,
                conversation_history=example.conversation_history,
            )

            # Explainability (only for LRG models that expose lrg in metadata)
            expl = None
            lrg = None
            if prediction.metadata and "lrg" in prediction.metadata:
                from src.lrg.graph import LRGGraph
                lrg = LRGGraph.from_dict(prediction.metadata["lrg"])
                expl = evaluate_explainability(
                    lrg=lrg,
                    sql=prediction.predicted_sql,
                    question=example.question,
                    is_correct=em,
                    failure_categories=fm.categories(),
                )

            example_results.append(ExampleResult(
                example=example,
                prediction=prediction,
                exact_match=em,
                execution_accuracy=ex_acc,
                component_scores=comp,
                failure_mode=fm,
                explainability=expl,
                elapsed_ms=elapsed_ms,
            ))

        elapsed_total = time.time() - start_time

        # Aggregate
        metric_dicts = [
            {
                "exact_match": r.exact_match,
                "execution_accuracy": r.execution_accuracy,
                "component_scores": r.component_scores,
            }
            for r in example_results
        ]
        agg = aggregate_metrics(metric_dicts)
        failure_summary = summarise_failures([r.failure_mode for r in example_results if r.failure_mode])
        expl_results = [r.explainability for r in example_results if r.explainability]
        expl_summary = aggregate_explainability(expl_results)

        per_example = [_serialise_result(r) for r in example_results]

        report = BenchmarkReport(
            run_id=run_id,
            model_name=self._model.model_name,
            dataset_name=self._dataset.name,
            n_examples=len(example_results),
            exact_match=agg.exact_match,
            execution_accuracy=agg.execution_accuracy,
            component_scores=agg.component_scores,
            failure_summary=failure_summary,
            explainability_summary=expl_summary,
            per_example=per_example,
            elapsed_seconds=elapsed_total,
        )
        logger.info(
            "Benchmark complete. EM=%.3f EX=%.3f elapsed=%.1fs",
            agg.exact_match or 0,
            agg.execution_accuracy or 0,
            elapsed_total,
        )
        return report


def _serialise_result(r: ExampleResult) -> dict:
    fm = r.failure_mode
    expl = r.explainability
    return {
        "example_id": r.example.example_id,
        "db_id": r.example.db_id,
        "question": r.example.question,
        "gold_sql": r.example.gold_sql,
        "predicted_sql": r.prediction.predicted_sql,
        "exact_match": r.exact_match,
        "execution_accuracy": r.execution_accuracy,
        "component_scores": r.component_scores,
        "elapsed_ms": round(r.elapsed_ms, 1),
        "failure_modes": fm.categories() if fm else [],
        "failure_details": fm.details if fm else {},
        "faithfulness": expl.faithfulness if expl else None,
        "completeness": expl.completeness if expl else None,
        "error_traceability": expl.error_traceability if expl else None,
        "lrg_summary": r.prediction.metadata.get("lrg_summary") if r.prediction.metadata else None,
    }
