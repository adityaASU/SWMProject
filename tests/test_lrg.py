"""Tests for LRG nodes, graph, synthesizer, and (mock) builder."""
from __future__ import annotations

import pytest

from src.lrg.nodes import (
    AggregationFunction,
    AggregationNode,
    AliasNode,
    AttributeNode,
    EdgeType,
    EntityNode,
    FilterNode,
    FilterOperator,
    GroupingNode,
    LRGEdge,
    NodeType,
    node_from_dict,
)
from src.lrg.graph import LRGGraph
from src.lrg.synthesizer import SQLSynthesizer


# ── Node serialisation ────────────────────────────────────────────────────────

def test_entity_node_roundtrip():
    node = EntityNode(table_name="employees", is_main_entity=True, label="employees")
    d = node.model_dump()
    restored = node_from_dict(d)
    assert isinstance(restored, EntityNode)
    assert restored.table_name == "employees"


def test_alias_node_roundtrip():
    node = AliasNode(table_name="employees", alias="e1", role_description="manager")
    d = node.model_dump()
    restored = node_from_dict(d)
    assert isinstance(restored, AliasNode)
    assert restored.alias == "e1"


def test_filter_node_roundtrip():
    node = FilterNode(
        table_name="orders", column_name="amount", operator=FilterOperator.GT, value=100
    )
    d = node.model_dump()
    restored = node_from_dict(d)
    assert isinstance(restored, FilterNode)
    assert restored.operator == FilterOperator.GT


# ── LRGGraph structure ────────────────────────────────────────────────────────

def _build_simple_lrg() -> LRGGraph:
    """Build a minimal LRG: employees JOIN departments, COUNT(*), GROUP BY dept."""
    g = LRGGraph()

    emp = EntityNode(table_name="employees", is_main_entity=True, label="employees")
    dept = EntityNode(table_name="departments", label="departments")
    emp_id = g.add_node(emp)
    dept_id = g.add_node(dept)

    g.add_edge(LRGEdge(
        source_id=emp_id,
        target_id=dept_id,
        edge_type=EdgeType.JOIN,
        join_left_col="dept_id",
        join_right_col="id",
        label="JOIN ON dept_id=id",
    ))

    attr = AttributeNode(table_name="departments", column_name="name", in_select=True, label="departments.name")
    attr_id = g.add_node(attr)
    g.add_edge(LRGEdge(source_id=dept_id, target_id=attr_id, edge_type=EdgeType.SELECTS))

    agg = AggregationNode(
        function=AggregationFunction.COUNT,
        table_name="employees",
        column_name="*",
        alias="cnt",
        label="COUNT(*)",
    )
    agg_id = g.add_node(agg)
    g.add_edge(LRGEdge(source_id=emp_id, target_id=agg_id, edge_type=EdgeType.AGG_OF))

    gb = GroupingNode(columns=[{"table": "departments", "column": "name"}], label="GROUP BY departments.name")
    gb_id = g.add_node(gb)
    g.add_edge(LRGEdge(source_id=gb_id, target_id=agg_id, edge_type=EdgeType.GROUP_BY))

    return g


def test_lrg_node_count():
    g = _build_simple_lrg()
    assert len(list(g.all_nodes())) == 5


def test_lrg_edge_count():
    g = _build_simple_lrg()
    assert len(g.all_edges()) == 4


def test_lrg_nodes_of_type():
    g = _build_simple_lrg()
    entities = g.nodes_of_type(NodeType.ENTITY)
    assert len(entities) == 2


def test_lrg_edges_of_type():
    g = _build_simple_lrg()
    joins = g.edges_of_type(EdgeType.JOIN)
    assert len(joins) == 1


def test_lrg_serialisation_roundtrip():
    g = _build_simple_lrg()
    d = g.to_dict()
    restored = LRGGraph.from_dict(d)
    assert len(list(restored.all_nodes())) == len(list(g.all_nodes()))
    assert len(restored.all_edges()) == len(g.all_edges())


def test_lrg_validation_no_errors():
    g = _build_simple_lrg()
    errors = g.validate()
    assert errors == []


def test_lrg_validation_detects_cycle():
    g = LRGGraph()
    a = EntityNode(table_name="a", label="a")
    b = EntityNode(table_name="b", label="b")
    aid = g.add_node(a)
    bid = g.add_node(b)
    g.add_edge(LRGEdge(source_id=aid, target_id=bid, edge_type=EdgeType.JOIN))
    # Manually add reverse edge to create a cycle (bypass add_edge validation)
    import networkx as nx
    g._g.add_edge(bid, aid)
    g._edges.append(LRGEdge(source_id=bid, target_id=aid, edge_type=EdgeType.JOIN))
    errors = g.validate()
    assert any("cycle" in e.lower() for e in errors)


# ── SQL Synthesizer ───────────────────────────────────────────────────────────

def test_synthesizer_basic_join():
    g = _build_simple_lrg()
    synth = SQLSynthesizer()
    sql = synth.synthesize(g)
    sql_lower = sql.lower()
    assert "select" in sql_lower
    assert "from employees" in sql_lower
    assert "join departments" in sql_lower
    assert "count(" in sql_lower
    assert "group by" in sql_lower


def test_synthesizer_produces_valid_string():
    g = _build_simple_lrg()
    synth = SQLSynthesizer()
    sql = synth.synthesize(g)
    assert isinstance(sql, str)
    assert len(sql) > 10


def test_synthesizer_filter():
    g = LRGGraph()
    emp = EntityNode(table_name="employees", is_main_entity=True, label="employees")
    emp_id = g.add_node(emp)
    filt = FilterNode(
        table_name="employees", column_name="salary",
        operator=FilterOperator.GT, value=50000, label="salary > 50000"
    )
    filt_id = g.add_node(filt)
    g.add_edge(LRGEdge(source_id=emp_id, target_id=filt_id, edge_type=EdgeType.FILTER_OF))

    synth = SQLSynthesizer()
    sql = synth.synthesize(g)
    assert "where" in sql.lower()
    assert "50000" in sql


def test_synthesizer_self_join():
    g = LRGGraph()
    e1 = AliasNode(table_name="employees", alias="e1", role_description="employee")
    e2 = AliasNode(table_name="employees", alias="e2", role_description="manager")
    e1_id = g.add_node(e1)
    e2_id = g.add_node(e2)
    g.add_edge(LRGEdge(
        source_id=e1_id, target_id=e2_id,
        edge_type=EdgeType.JOIN,
        join_left_col="manager_id", join_right_col="id",
        label="self join",
    ))
    synth = SQLSynthesizer()
    sql = synth.synthesize(g)
    assert "e1" in sql
    assert "e2" in sql
    assert "join" in sql.lower()
