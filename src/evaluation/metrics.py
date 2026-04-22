"""Standard Text2SQL evaluation metrics: Exact Match and Execution Accuracy."""
from __future__ import annotations

import re
import sqlite3
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    exact_match: Optional[float] = None          # 0.0 or 1.0 per example; mean over dataset
    execution_accuracy: Optional[float] = None
    component_scores: dict[str, float] = field(default_factory=dict)
    n_examples: int = 0


def normalise_sql(sql: str) -> str:
    """Normalise SQL for exact-match comparison (case, whitespace, aliases)."""
    sql = sql.strip().lower()
    # Collapse whitespace
    sql = re.sub(r"\s+", " ", sql)
    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()
    return sql


def exact_match(predicted: str, gold: str) -> bool:
    """Return True if *predicted* matches *gold* after normalisation."""
    return normalise_sql(predicted) == normalise_sql(gold)


def execution_accuracy(
    predicted_sql: str,
    gold_sql: str,
    db_path: Path,
    timeout: int = 30,
) -> bool:
    """Return True if *predicted_sql* and *gold_sql* produce the same result set.

    Connects to *db_path* (SQLite file). Returns False on any execution error.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            pred_rows = _execute(conn, predicted_sql, timeout)
            gold_rows = _execute(conn, gold_sql, timeout)
            return _results_equal(pred_rows, gold_rows)
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("Execution accuracy check failed: %s", exc)
        return False


def _execute(conn: sqlite3.Connection, sql: str, timeout: int) -> list[tuple]:
    conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    return [tuple(row) for row in rows]


def _results_equal(a: list[tuple], b: list[tuple]) -> bool:
    """Compare two result sets ignoring row order."""
    try:
        return sorted(str(r) for r in a) == sorted(str(r) for r in b)
    except Exception:
        return False


def component_match(predicted: str, gold: str) -> dict[str, bool]:
    """Clause-level comparison for diagnostic purposes."""
    clauses = ["select", "from", "join", "where", "group by", "having", "order by", "limit"]
    pred_norm = normalise_sql(predicted)
    gold_norm = normalise_sql(gold)

    results = {}
    for clause in clauses:
        results[clause] = _extract_clause(pred_norm, clause) == _extract_clause(gold_norm, clause)
    return results


def _extract_clause(sql: str, clause: str) -> str:
    """Heuristically extract a clause from normalised SQL."""
    all_keywords = [
        "select", "from", "join", "where", "group by", "having", "order by", "limit"
    ]
    try:
        idx = sql.find(clause)
        if idx == -1:
            return ""
        end = len(sql)
        for kw in all_keywords:
            if kw == clause:
                continue
            pos = sql.find(kw, idx + len(clause))
            if pos != -1 and pos < end:
                end = pos
        return sql[idx:end].strip()
    except Exception:
        return ""


def aggregate_metrics(results: list[dict]) -> MetricResult:
    """Aggregate per-example metric dicts into a MetricResult."""
    if not results:
        return MetricResult()

    em_scores = [r["exact_match"] for r in results if r.get("exact_match") is not None]
    ex_scores = [r["execution_accuracy"] for r in results if r.get("execution_accuracy") is not None]

    comp_keys = set()
    for r in results:
        comp_keys.update(r.get("component_scores", {}).keys())
    comp_scores: dict[str, float] = {}
    for key in comp_keys:
        vals = [r["component_scores"][key] for r in results if key in r.get("component_scores", {})]
        comp_scores[key] = sum(vals) / len(vals) if vals else 0.0

    return MetricResult(
        exact_match=sum(em_scores) / len(em_scores) if em_scores else None,
        execution_accuracy=sum(ex_scores) / len(ex_scores) if ex_scores else None,
        component_scores=comp_scores,
        n_examples=len(results),
    )
