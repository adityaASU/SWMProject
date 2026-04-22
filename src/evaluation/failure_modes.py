"""Failure mode analysis for Text2SQL predictions.

Five categories matching the project proposal:
  1. schema_linking  — wrong table/column referenced
  2. join_hallucination — join path not valid in schema FK graph
  3. nested_subquery  — incorrect nesting or aggregation scope
  4. self_join        — missing or wrong alias for repeated table
  5. context_drift    — wrong constraint carried from conversation history
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

import sqlglot
import sqlglot.expressions as exp

from src.schema.graph import SchemaGraph
from src.schema.parser import SchemaInfo

logger = logging.getLogger(__name__)


@dataclass
class FailureModeResult:
    is_correct: bool
    schema_linking: bool = False
    join_hallucination: bool = False
    nested_subquery: bool = False
    self_join: bool = False
    context_drift: bool = False
    details: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def any_failure(self) -> bool:
        return any([
            self.schema_linking,
            self.join_hallucination,
            self.nested_subquery,
            self.self_join,
            self.context_drift,
        ])

    def categories(self) -> list[str]:
        cats = []
        if self.schema_linking:
            cats.append("schema_linking")
        if self.join_hallucination:
            cats.append("join_hallucination")
        if self.nested_subquery:
            cats.append("nested_subquery")
        if self.self_join:
            cats.append("self_join")
        if self.context_drift:
            cats.append("context_drift")
        return cats


def analyse_failure_modes(
    predicted_sql: str,
    gold_sql: str,
    schema: SchemaInfo,
    schema_graph: SchemaGraph,
    is_correct: bool,
    conversation_history: Optional[list[dict]] = None,
) -> FailureModeResult:
    """Analyse *predicted_sql* for structural failure modes.

    Even when *is_correct* is True we run the analysis (for faithfulness measurement).
    """
    result = FailureModeResult(is_correct=is_correct)

    try:
        pred_parsed = _safe_parse(predicted_sql)
        gold_parsed = _safe_parse(gold_sql)
    except Exception as exc:
        logger.debug("SQL parse error in failure analysis: %s", exc)
        result.details["parse_error"] = str(exc)
        return result

    if pred_parsed is None:
        return result

    # 1. Schema linking
    result.schema_linking, result.details["schema_linking"] = _check_schema_linking(
        pred_parsed, schema
    )

    # 2. Join hallucination
    result.join_hallucination, result.details["join_hallucination"] = _check_join_hallucination(
        pred_parsed, schema_graph
    )

    # 3. Nested subquery
    result.nested_subquery, result.details["nested_subquery"] = _check_nested_subquery(
        pred_parsed, gold_parsed
    )

    # 4. Self-join
    result.self_join, result.details["self_join"] = _check_self_join(pred_parsed, gold_parsed)

    # 5. Context drift
    if conversation_history:
        result.context_drift, result.details["context_drift"] = _check_context_drift(
            pred_parsed, gold_parsed, conversation_history
        )

    return result


# ── Individual checkers ───────────────────────────────────────────────────────

def _safe_parse(sql: str):
    try:
        return sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None


def _check_schema_linking(pred, schema: SchemaInfo) -> tuple[bool, dict]:
    """Detect references to tables/columns that don't exist in the schema."""
    known_tables = {t.lower() for t in schema.table_names()}
    known_cols: set[str] = set()
    for tinfo in schema.tables.values():
        for col in tinfo.columns:
            known_cols.add(col.name.lower())

    bad_tables = []
    bad_cols = []

    for table in pred.find_all(exp.Table):
        tname = table.name.lower() if table.name else ""
        if tname and tname not in known_tables:
            bad_tables.append(tname)

    for col in pred.find_all(exp.Column):
        cname = col.name.lower() if col.name else ""
        if cname and cname not in known_cols and cname != "*":
            bad_cols.append(cname)

    has_error = bool(bad_tables or bad_cols)
    return has_error, {"bad_tables": bad_tables, "bad_cols": bad_cols}


def _check_join_hallucination(pred, schema_graph: SchemaGraph) -> tuple[bool, dict]:
    """Detect JOIN clauses between tables with no FK path in the schema graph."""
    bad_joins = []

    for join in pred.find_all(exp.Join):
        right_table = join.find(exp.Table)
        if right_table is None:
            continue
        right_name = right_table.name if right_table.name else ""

        # Look for the ON condition to find left table
        on_cond = join.args.get("on")
        if on_cond is None:
            continue

        cols = list(on_cond.find_all(exp.Column))
        tables_in_on = {col.table.lower() for col in cols if col.table}

        for left_name in tables_in_on:
            if left_name == right_name.lower():
                continue
            path = schema_graph.join_path(left_name, right_name)
            if path is None:
                bad_joins.append(f"{left_name} -> {right_name}")

    has_error = bool(bad_joins)
    return has_error, {"hallucinated_joins": bad_joins}


def _check_nested_subquery(pred, gold) -> tuple[bool, dict]:
    """Detect nesting-related errors: present in gold but absent/incorrect in pred."""
    pred_subqueries = len(list(pred.find_all(exp.Subquery))) if pred else 0
    gold_subqueries = len(list(gold.find_all(exp.Subquery))) if gold else 0

    mismatch = pred_subqueries != gold_subqueries
    return mismatch, {"pred_subquery_count": pred_subqueries, "gold_subquery_count": gold_subqueries}


def _check_self_join(pred, gold) -> tuple[bool, dict]:
    """Detect missing or wrong aliasing in self-joins."""
    pred_tables = _collect_table_refs(pred)
    gold_tables = _collect_table_refs(gold)

    # Self-join needed if any table appears multiple times in gold
    gold_self_join_tables = {t for t, cnt in gold_tables.items() if cnt > 1}
    pred_self_join_tables = {t for t, cnt in pred_tables.items() if cnt > 1}

    missing = gold_self_join_tables - pred_self_join_tables
    return bool(missing), {"missing_self_join_tables": list(missing)}


def _collect_table_refs(parsed) -> dict[str, int]:
    if parsed is None:
        return {}
    counts: dict[str, int] = {}
    for table in parsed.find_all(exp.Table):
        name = (table.name or "").lower()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _check_context_drift(pred, gold, history: list[dict]) -> tuple[bool, dict]:
    """Detect when predicted SQL ignores constraints from conversation history."""
    if not history:
        return False, {}

    # Heuristic: check if gold references tables from previous turns
    # that pred omits entirely
    prev_tables: set[str] = set()
    for turn in history[:-1]:
        sql = turn.get("sql", "")
        if sql:
            try:
                parsed = sqlglot.parse_one(sql, dialect="sqlite")
                for t in parsed.find_all(exp.Table):
                    if t.name:
                        prev_tables.add(t.name.lower())
            except Exception:
                pass

    gold_tables = set(_collect_table_refs(gold).keys())
    pred_tables = set(_collect_table_refs(pred).keys())

    # Tables in gold that came from history but are missing from pred
    drifted = (gold_tables & prev_tables) - pred_tables
    return bool(drifted), {"drifted_tables": list(drifted)}


def summarise_failures(results: list[FailureModeResult]) -> dict:
    """Aggregate failure mode counts across a list of results."""
    total = len(results)
    if total == 0:
        return {}
    categories = ["schema_linking", "join_hallucination", "nested_subquery", "self_join", "context_drift"]
    summary = {"total": total, "correct": sum(1 for r in results if r.is_correct)}
    for cat in categories:
        summary[cat] = sum(1 for r in results if getattr(r, cat))
        summary[f"{cat}_rate"] = round(summary[cat] / total, 4)
    return summary
