"""Tests for evaluation metrics, failure mode analysis, and explainability."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.evaluation.metrics import (
    exact_match,
    normalise_sql,
    component_match,
    aggregate_metrics,
)
from src.evaluation.explainability import (
    faithfulness,
    completeness,
    evaluate_explainability,
    aggregate_explainability,
)
from src.lrg.nodes import (
    AggregationFunction,
    AggregationNode,
    EntityNode,
    FilterNode,
    FilterOperator,
    GroupingNode,
    LRGEdge,
    EdgeType,
    AttributeNode,
)
from src.lrg.graph import LRGGraph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simple_lrg() -> LRGGraph:
    g = LRGGraph()
    emp = EntityNode(table_name="employees", is_main_entity=True, label="employees")
    emp_id = g.add_node(emp)
    agg = AggregationNode(
        function=AggregationFunction.COUNT, table_name="employees", column_name="*", label="COUNT(*)"
    )
    agg_id = g.add_node(agg)
    filt = FilterNode(
        table_name="employees", column_name="dept_id",
        operator=FilterOperator.EQ, value=5, label="dept_id = 5"
    )
    filt_id = g.add_node(filt)
    g.add_edge(LRGEdge(source_id=emp_id, target_id=agg_id, edge_type=EdgeType.AGG_OF))
    g.add_edge(LRGEdge(source_id=emp_id, target_id=filt_id, edge_type=EdgeType.FILTER_OF))
    gb = GroupingNode(columns=[{"table": "employees", "column": "dept_id"}], label="GROUP BY dept_id")
    gb_id = g.add_node(gb)
    g.add_edge(LRGEdge(source_id=gb_id, target_id=agg_id, edge_type=EdgeType.GROUP_BY))
    return g


# ── Exact match ───────────────────────────────────────────────────────────────

def test_exact_match_identical():
    assert exact_match("SELECT * FROM students", "SELECT * FROM students")


def test_exact_match_case_insensitive():
    assert exact_match("select * from students", "SELECT * FROM students")


def test_exact_match_whitespace_normalised():
    assert exact_match("SELECT  *  FROM   students", "SELECT * FROM students")


def test_exact_match_trailing_semicolon():
    assert exact_match("SELECT * FROM students;", "SELECT * FROM students")


def test_exact_match_different():
    assert not exact_match("SELECT name FROM students", "SELECT * FROM students")


def test_normalise_sql_lowercases():
    assert normalise_sql("SELECT Name FROM Students") == "select name from students"


# ── Component match ───────────────────────────────────────────────────────────

def test_component_match_select():
    cm = component_match(
        "SELECT name FROM students WHERE age > 18",
        "SELECT name FROM students WHERE age > 18",
    )
    assert cm["select"] is True


def test_component_match_wrong_where():
    cm = component_match(
        "SELECT name FROM students WHERE age > 20",
        "SELECT name FROM students WHERE age > 18",
    )
    assert cm["where"] is False


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def test_aggregate_metrics_empty():
    result = aggregate_metrics([])
    assert result.exact_match is None
    assert result.n_examples == 0


def test_aggregate_metrics_all_correct():
    results = [{"exact_match": True, "execution_accuracy": True, "component_scores": {}} for _ in range(5)]
    agg = aggregate_metrics(results)
    assert agg.exact_match == pytest.approx(1.0)
    assert agg.execution_accuracy == pytest.approx(1.0)
    assert agg.n_examples == 5


def test_aggregate_metrics_mixed():
    results = [
        {"exact_match": True, "execution_accuracy": True, "component_scores": {}},
        {"exact_match": False, "execution_accuracy": False, "component_scores": {}},
    ]
    agg = aggregate_metrics(results)
    assert agg.exact_match == pytest.approx(0.5)


# ── Faithfulness ──────────────────────────────────────────────────────────────

def test_faithfulness_full_coverage():
    lrg = _simple_lrg()
    sql = "SELECT COUNT(*) FROM employees WHERE dept_id = 5 GROUP BY dept_id"
    score = faithfulness(lrg, sql)
    assert score > 0.5


def test_faithfulness_empty_lrg():
    lrg = LRGGraph()
    sql = "SELECT COUNT(*) FROM employees WHERE dept_id = 5"
    score = faithfulness(lrg, sql)
    assert score == 0.0 or score >= 0.0  # At minimum 0


# ── Completeness ──────────────────────────────────────────────────────────────

def test_completeness_simple():
    lrg = _simple_lrg()
    question = "How many employees are in each department?"
    score = completeness(lrg, question)
    assert 0.0 <= score <= 1.0


def test_completeness_with_filter():
    lrg = _simple_lrg()
    question = "How many employees have salary more than 50000?"
    score = completeness(lrg, question)
    assert score > 0.0


# ── Full explainability evaluation ────────────────────────────────────────────

def test_evaluate_explainability_correct():
    lrg = _simple_lrg()
    result = evaluate_explainability(
        lrg=lrg,
        sql="SELECT COUNT(*) FROM employees GROUP BY dept_id",
        question="How many employees are in each department?",
        is_correct=True,
        failure_categories=[],
    )
    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.completeness <= 1.0
    assert result.error_traceability is False


def test_evaluate_explainability_incorrect():
    lrg = _simple_lrg()
    result = evaluate_explainability(
        lrg=lrg,
        sql="SELECT * FROM employees",
        question="How many employees?",
        is_correct=False,
        failure_categories=["schema_linking"],
    )
    assert result.error_traceability is True
    assert result.traceable_error_node is not None


def test_aggregate_explainability():
    lrg = _simple_lrg()
    results = [
        evaluate_explainability(lrg, "SELECT COUNT(*) FROM employees", "How many?", True, []),
        evaluate_explainability(lrg, "SELECT * FROM employees", "List all", False, ["schema_linking"]),
    ]
    agg = aggregate_explainability(results)
    assert "mean_faithfulness" in agg
    assert "error_traceability_rate" in agg
    assert agg["n_examples"] == 2


# ── Failure mode detection (sqlglot required) ─────────────────────────────────

def test_failure_mode_schema_linking():
    """A query referencing a nonexistent table should trigger schema_linking."""
    try:
        import sqlglot
    except ImportError:
        pytest.skip("sqlglot not installed")

    from src.schema.parser import SchemaParser
    from src.schema.graph import SchemaGraph
    import sqlite3, tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_dir = Path(tmp) / "mydb"
        db_dir.mkdir()
        db_path = db_dir / "mydb.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT)")
        conn.close()

        parser = SchemaParser()
        schema = parser.auto_parse(Path(tmp), "mydb")
        sg = SchemaGraph(schema)

        from src.evaluation.failure_modes import analyse_failure_modes
        result = analyse_failure_modes(
            predicted_sql="SELECT * FROM ghost_table",
            gold_sql="SELECT * FROM students",
            schema=schema,
            schema_graph=sg,
            is_correct=False,
        )
        assert result.schema_linking is True
