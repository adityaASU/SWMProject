"""Logical Reasoning Graph node and edge definitions.

All nodes are Pydantic models so they can be serialised to JSON easily
and used as structured LLM output schemas.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    ENTITY = "entity"
    ATTRIBUTE = "attribute"
    FILTER = "filter"
    AGGREGATION = "aggregation"
    GROUPING = "grouping"
    ALIAS = "alias"
    SUBGRAPH = "subgraph"


class AggregationFunction(str, Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT_DISTINCT = "COUNT DISTINCT"


class FilterOperator(str, Enum):
    EQ = "="
    NEQ = "!="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    BETWEEN = "BETWEEN"


class EdgeType(str, Enum):
    JOIN = "join"
    FILTER_OF = "filter_of"
    AGG_OF = "agg_of"
    GROUP_BY = "group_by"
    HAVING = "having"
    SUBQUERY_OF = "subquery_of"
    SELECTS = "selects"
    # ── Professor feedback: explicit inter-graph binding edges ──
    CONTEXT_PASS = "context_pass"   # outer entity referenced inside inner subgraph
    BINDING = "binding"             # correlated binding: inner col = outer col


class SubqueryType(str, Enum):
    """Classify nested queries per professor feedback."""
    UNNESTED = "unnested"                       # no subquery at all
    NESTED_UNCORRELATED = "nested_uncorrelated" # inner is fully independent
    NESTED_CORRELATED = "nested_correlated"     # inner references outer table/col


# ── Nodes ─────────────────────────────────────────────────────────────────────

class BaseNode(BaseModel):
    """Shared fields for every LRG node."""

    node_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    node_type: NodeType
    label: str = ""          # Human-readable label for visualisation


class EntityNode(BaseNode):
    """Represents a table referenced in the query (maps to FROM clause)."""

    node_type: Literal[NodeType.ENTITY] = NodeType.ENTITY
    table_name: str
    is_main_entity: bool = False   # True for the primary table in FROM


class AliasNode(BaseNode):
    """Represents one semantic role of a table in a self-join.

    Wraps an EntityNode with a distinct alias identity so that conditions
    attached to one role do not bleed into another.
    """

    node_type: Literal[NodeType.ALIAS] = NodeType.ALIAS
    table_name: str
    alias: str                     # e.g. "e1", "e2" for self-joined employees
    role_description: str = ""     # e.g. "manager", "subordinate"


class AttributeNode(BaseNode):
    """Represents a column referenced in SELECT, WHERE, or aggregation."""

    node_type: Literal[NodeType.ATTRIBUTE] = NodeType.ATTRIBUTE
    table_name: str
    column_name: str
    alias: Optional[str] = None    # table alias when inside self-join
    in_select: bool = False        # True if this column appears in SELECT


class FilterNode(BaseNode):
    """Represents a WHERE / HAVING condition."""

    node_type: Literal[NodeType.FILTER] = NodeType.FILTER
    table_name: str
    column_name: str
    operator: FilterOperator
    value: Optional[Any] = None    # None for IS NULL / IS NOT NULL
    value2: Optional[Any] = None   # Second bound for BETWEEN
    alias: Optional[str] = None    # table alias for self-join filters
    is_having: bool = False        # True if this is a HAVING clause filter


class AggregationNode(BaseNode):
    """Represents an aggregate function (COUNT, SUM, …)."""

    node_type: Literal[NodeType.AGGREGATION] = NodeType.AGGREGATION
    function: AggregationFunction
    table_name: str
    column_name: str               # "*" for COUNT(*)
    alias: Optional[str] = None    # output alias in SELECT clause
    table_alias: Optional[str] = None  # table alias for self-join contexts


class GroupingNode(BaseNode):
    """Represents GROUP BY columns."""

    node_type: Literal[NodeType.GROUPING] = NodeType.GROUPING
    columns: list[dict[str, str]]  # [{"table": ..., "column": ...}, ...]


class SubgraphNode(BaseNode):
    """Represents a nested subquery modeled as a scoped sub-LRG.

    Professor feedback: each SELECT must be its own distinct graph.
    Inter-graph connections (context_pass, binding edges) are stored
    explicitly so the synthesizer can reconstruct correlated subqueries.
    """

    node_type: Literal[NodeType.SUBGRAPH] = NodeType.SUBGRAPH
    role: str = ""                      # "IN subquery", "EXISTS subquery", etc.
    subquery_type: str = "unnested"     # unnested / nested_uncorrelated / nested_correlated

    # The inner LRG is stored as a nested dict (avoids circular import).
    # Synthesizer calls LRGGraph.from_dict() to reconstruct it.
    inner_lrg: Optional[dict] = None

    # Correlated binding: outer table/col that inner WHERE references.
    # e.g. [{"outer_table": "employees", "outer_col": "dept_id",
    #         "inner_table": "dept_budget", "inner_col": "dept_id"}]
    correlated_bindings: list = []

    # Operator wrapping this subquery in the outer WHERE/HAVING
    # e.g. "IN", "NOT IN", "EXISTS", "NOT EXISTS", "="
    operator: str = "IN"


# ── Edges ─────────────────────────────────────────────────────────────────────

class LRGEdge(BaseModel):
    """A directed edge in the Logical Reasoning Graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    # JOIN-specific metadata (populated for EdgeType.JOIN)
    join_left_col: Optional[str] = None
    join_right_col: Optional[str] = None
    label: str = ""


# ── Union type for all nodes ──────────────────────────────────────────────────

LRGNode = Union[EntityNode, AliasNode, AttributeNode, FilterNode, AggregationNode, GroupingNode, SubgraphNode]


def node_from_dict(data: dict) -> LRGNode:
    """Deserialise a node from a plain dict (uses node_type discriminator)."""
    _map = {
        NodeType.ENTITY: EntityNode,
        NodeType.ALIAS: AliasNode,
        NodeType.ATTRIBUTE: AttributeNode,
        NodeType.FILTER: FilterNode,
        NodeType.AGGREGATION: AggregationNode,
        NodeType.GROUPING: GroupingNode,
        NodeType.SUBGRAPH: SubgraphNode,
    }
    ntype = NodeType(data["node_type"])
    return _map[ntype](**data)
