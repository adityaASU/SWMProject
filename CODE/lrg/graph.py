"""LRG DiGraph wrapper with validation."""
from __future__ import annotations

from typing import Iterator, Optional

import networkx as nx

from src.lrg.nodes import (
    AggregationNode,
    AliasNode,
    AttributeNode,
    EdgeType,
    EntityNode,
    FilterNode,
    GroupingNode,
    LRGEdge,
    LRGNode,
    NodeType,
    SubgraphNode,
    node_from_dict,
)
from src.schema.graph import SchemaGraph


class LRGGraph:
    """Directed graph representing the Logical Reasoning Graph for one query."""

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, LRGNode] = {}
        self._edges: list[LRGEdge] = []
        self.metadata: dict = {}  # stores order_by, limit, distinct

    # ── Mutation API ──────────────────────────────────────────────────────────

    def add_node(self, node: LRGNode) -> str:
        """Add *node* to the graph. Returns its node_id."""
        self._nodes[node.node_id] = node
        self._g.add_node(node.node_id, node_type=node.node_type, label=node.label)
        return node.node_id

    def add_edge(self, edge: LRGEdge) -> None:
        """Add a directed edge. Both source and target must already exist."""
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node '{edge.source_id}' not in graph")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node '{edge.target_id}' not in graph")
        self._edges.append(edge)
        self._g.add_edge(edge.source_id, edge.target_id, edge_type=edge.edge_type)

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> LRGNode:
        return self._nodes[node_id]

    def nodes_of_type(self, ntype: NodeType) -> list[LRGNode]:
        return [n for n in self._nodes.values() if n.node_type == ntype]

    def edges_of_type(self, etype: EdgeType) -> list[LRGEdge]:
        return [e for e in self._edges if e.edge_type == etype]

    def successors(self, node_id: str) -> list[LRGNode]:
        return [self._nodes[nid] for nid in self._g.successors(node_id)]

    def predecessors(self, node_id: str) -> list[LRGNode]:
        return [self._nodes[nid] for nid in self._g.predecessors(node_id)]

    def all_nodes(self) -> Iterator[LRGNode]:
        return iter(self._nodes.values())

    def all_edges(self) -> list[LRGEdge]:
        return list(self._edges)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, schema_graph: Optional[SchemaGraph] = None) -> list[str]:
        """Run structural and optionally schema-grounded validation.

        Returns a list of error strings (empty means valid).
        """
        errors: list[str] = []

        # 1. No cycles (LRG must be a DAG)
        if not nx.is_directed_acyclic_graph(self._g):
            errors.append("LRG contains a cycle — it must be a DAG.")

        # 2. At least one entity node
        entities = self.nodes_of_type(NodeType.ENTITY) + self.nodes_of_type(NodeType.ALIAS)
        if not entities:
            errors.append("LRG has no EntityNode or AliasNode — at least one table must be referenced.")

        # 3. Schema-grounded checks
        if schema_graph is not None:
            for node in self._nodes.values():
                if isinstance(node, (EntityNode, AliasNode)):
                    tname = node.table_name
                    if not schema_graph.has_table(tname):
                        errors.append(f"Table '{tname}' referenced in LRG does not exist in schema.")

            for edge in self._edges:
                if edge.edge_type == EdgeType.JOIN:
                    src = self._nodes[edge.source_id]
                    tgt = self._nodes[edge.target_id]
                    src_table = getattr(src, "table_name", None)
                    tgt_table = getattr(tgt, "table_name", None)
                    if src_table and tgt_table:
                        if not schema_graph.has_direct_fk(src_table, tgt_table):
                            path = schema_graph.join_path(src_table, tgt_table)
                            if path is None:
                                errors.append(
                                    f"JOIN edge from '{src_table}' to '{tgt_table}' has no FK path in schema."
                                )

        return errors

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LRGGraph":
        g = cls()
        for ndata in data.get("nodes", []):
            g.add_node(node_from_dict(ndata))
        for edata in data.get("edges", []):
            g.add_edge(LRGEdge(**edata))
        return g

    def summary(self) -> str:
        """Short human-readable description of the graph."""
        counts = {}
        for n in self._nodes.values():
            counts[n.node_type] = counts.get(n.node_type, 0) + 1
        lines = [f"LRGGraph: {len(self._nodes)} nodes, {len(self._edges)} edges"]
        for ntype, cnt in counts.items():
            lines.append(f"  {ntype.value}: {cnt}")
        return "\n".join(lines)
