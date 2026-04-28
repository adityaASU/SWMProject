"""Explainability metrics for the LRG framework.

Three measures:
  - faithfulness   : fraction of SQL clauses traceable to an LRG node
  - completeness   : fraction of logical operations in the question covered by LRG nodes
  - error_traceability: whether an incorrect query's failure can be localised to a node
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.lrg.graph import LRGGraph
from src.lrg.nodes import EdgeType, NodeType


@dataclass
class ExplainabilityResult:
    faithfulness: float = 0.0       # 0.0 – 1.0
    completeness: float = 0.0       # 0.0 – 1.0
    error_traceability: bool = False
    traceable_error_node: Optional[str] = None
    details: dict = field(default_factory=dict)


def faithfulness(lrg: LRGGraph, sql: str) -> float:
    """Measure fraction of SQL structural clauses that map to an LRG node.

    A clause is considered 'traceable' if there exists at least one node of
    the corresponding type in the LRG.
    """
    sql_lower = sql.lower()

    clause_to_node_type = {
        "select": [NodeType.ATTRIBUTE, NodeType.AGGREGATION],
        "from": [NodeType.ENTITY, NodeType.ALIAS],
        "join": [NodeType.ENTITY, NodeType.ALIAS],
        "where": [NodeType.FILTER],
        "group by": [NodeType.GROUPING],
        "having": [NodeType.FILTER],
    }

    present_clauses = [c for c in clause_to_node_type if c in sql_lower]
    if not present_clauses:
        return 1.0

    traceable = 0
    for clause in present_clauses:
        node_types = clause_to_node_type[clause]
        for ntype in node_types:
            if lrg.nodes_of_type(ntype):
                traceable += 1
                break

    return round(traceable / len(present_clauses), 4)


def completeness(lrg: LRGGraph, question: str) -> float:
    """Estimate fraction of logical operations identified in *question* that are in LRG.

    Uses keyword heuristics to detect which operation types the question implies.
    """
    q = question.lower()

    expected: list[NodeType] = [NodeType.ENTITY]  # always need a table

    agg_keywords = ["how many", "count", "total", "sum", "average", "maximum", "minimum", "max", "min", "avg"]
    if any(kw in q for kw in agg_keywords):
        expected.append(NodeType.AGGREGATION)

    filter_keywords = ["where", "which", "that", "whose", "more than", "less than", "equal", "at least", "at most", "between", "with", "has", "have", "not", "no"]
    if any(kw in q for kw in filter_keywords):
        expected.append(NodeType.FILTER)

    group_keywords = ["each", "per", "group", "every", "by"]
    if any(kw in q for kw in group_keywords):
        expected.append(NodeType.GROUPING)

    subq_keywords = ["who have", "that have", "except", "not in", "exists", "subquery"]
    if any(kw in q for kw in subq_keywords):
        expected.append(NodeType.SUBGRAPH)

    if not expected:
        return 1.0

    found = sum(1 for ntype in expected if lrg.nodes_of_type(ntype))
    return round(found / len(expected), 4)


def error_traceability(
    lrg: LRGGraph,
    is_correct: bool,
    failure_categories: list[str],
) -> tuple[bool, Optional[str]]:
    """Determine whether a prediction failure can be localised to a specific LRG node.

    Returns (is_traceable, node_description).
    """
    if is_correct:
        return False, None

    category_to_node_type = {
        "schema_linking": NodeType.ENTITY,
        "join_hallucination": NodeType.ENTITY,
        "nested_subquery": NodeType.SUBGRAPH,
        "self_join": NodeType.ALIAS,
        "context_drift": NodeType.FILTER,
    }

    for cat in failure_categories:
        ntype = category_to_node_type.get(cat)
        if ntype:
            nodes = lrg.nodes_of_type(ntype)
            if nodes:
                node = nodes[0]
                desc = f"{ntype.value} node '{node.label or node.node_id}' (category: {cat})"
                return True, desc

    return False, None


def evaluate_explainability(
    lrg: LRGGraph,
    sql: str,
    question: str,
    is_correct: bool,
    failure_categories: Optional[list[str]] = None,
) -> ExplainabilityResult:
    """Compute all explainability metrics for one prediction."""
    faith = faithfulness(lrg, sql)
    compl = completeness(lrg, question)
    traceable, trace_node = error_traceability(
        lrg, is_correct, failure_categories or []
    )
    return ExplainabilityResult(
        faithfulness=faith,
        completeness=compl,
        error_traceability=traceable,
        traceable_error_node=trace_node,
        details={
            "faithfulness": faith,
            "completeness": compl,
            "error_traceable": traceable,
            "trace_node": trace_node,
        },
    )


def aggregate_explainability(results: list[ExplainabilityResult]) -> dict:
    """Aggregate explainability scores across a list of results."""
    if not results:
        return {}
    return {
        "mean_faithfulness": round(sum(r.faithfulness for r in results) / len(results), 4),
        "mean_completeness": round(sum(r.completeness for r in results) / len(results), 4),
        "error_traceability_rate": round(
            sum(1 for r in results if r.error_traceability) / len(results), 4
        ),
        "n_examples": len(results),
    }
