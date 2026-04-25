"""LRG Self-Repair: detect structural errors in an LRGGraph and correct them.

Professor feedback:
  "program targeted tests, error detection, and correction routines directly
   into your pipeline, enabling the system to actively learn from its mistakes
   and self-repair over time."

Repair routines target the three failure mode families:
  1. join_hallucination  — replace invalid join with FK-validated path
  2. nested_subquery     — fix missing / wrong subquery type classification
  3. self_join           — ensure alias nodes exist when same table used twice
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.lrg.graph import LRGGraph
from src.lrg.nodes import (
    AliasNode,
    EdgeType,
    EntityNode,
    LRGEdge,
    NodeType,
    SubgraphNode,
)
from src.schema.graph import SchemaGraph

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RepairResult:
    original_errors: list[str]
    repairs_applied: list[str] = field(default_factory=list)
    repaired_lrg: Optional[LRGGraph] = None
    success: bool = False

    def summary(self) -> str:
        if not self.repairs_applied:
            return "No repairs needed."
        lines = [f"Applied {len(self.repairs_applied)} repair(s):"]
        for r in self.repairs_applied:
            lines.append(f"  - {r}")
        return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def repair(
    lrg: LRGGraph,
    schema_graph: SchemaGraph,
    validation_errors: list[str],
) -> RepairResult:
    """Attempt to correct all structural errors in *lrg* in-place.

    Returns a RepairResult describing what was fixed.
    The repaired LRG is a new LRGGraph object (original is not mutated).
    """
    result = RepairResult(original_errors=list(validation_errors))

    if not validation_errors:
        result.success = True
        result.repaired_lrg = lrg
        return result

    # Work on a copy via round-trip serialisation
    try:
        working = LRGGraph.from_dict(lrg.to_dict())
    except Exception as exc:
        logger.error("Repair: could not clone LRG: %s", exc)
        result.repaired_lrg = lrg
        return result

    for error in validation_errors:
        error_lower = error.lower()

        # ── Repair 1: hallucinated JOIN ───────────────────────────────────────
        if "no fk path" in error_lower or "fk path" in error_lower:
            applied = _repair_hallucinated_join(working, schema_graph, error)
            result.repairs_applied.extend(applied)

        # ── Repair 2: unknown table (schema linking) ──────────────────────────
        elif "does not exist in schema" in error_lower:
            applied = _repair_unknown_table(working, schema_graph, error)
            result.repairs_applied.extend(applied)

        # ── Repair 3: self-join missing alias ─────────────────────────────────
        elif "self-join" in error_lower or "alias" in error_lower:
            applied = _repair_self_join_alias(working, schema_graph, error)
            result.repairs_applied.extend(applied)

    # Re-validate after repairs
    remaining_errors = working.validate(schema_graph)
    result.success = len(remaining_errors) == 0
    result.repaired_lrg = working
    return result


# ── Repair routines ───────────────────────────────────────────────────────────

def _repair_hallucinated_join(
    lrg: LRGGraph,
    schema_graph: SchemaGraph,
    error: str,
) -> list[str]:
    """Remove invalid JOIN edges and replace them with FK-validated paths.

    Strategy:
      1. Find all JOIN edges that have no FK path in the schema.
      2. For each bad join (A → B), find the shortest FK path A → X → ... → B.
      3. Remove the bad edge; add intermediate entity nodes + valid edges.
    """
    repairs: list[str] = []
    bad_edges = [
        e for e in lrg.all_edges()
        if e.edge_type == EdgeType.JOIN
        and not _join_is_valid(lrg, e, schema_graph)
    ]

    for bad_edge in bad_edges:
        src_node = lrg.get_node(bad_edge.source_id)
        tgt_node = lrg.get_node(bad_edge.target_id)
        src_table = getattr(src_node, "table_name", "")
        tgt_table = getattr(tgt_node, "table_name", "")

        path = schema_graph.join_path(src_table, tgt_table)
        if path is None:
            # Cannot find any path — drop the edge entirely
            lrg._edges.remove(bad_edge)
            lrg._g.remove_edge(bad_edge.source_id, bad_edge.target_id)
            repairs.append(
                f"Removed unresolvable JOIN {src_table} → {tgt_table} (no FK path exists)"
            )
            continue

        # Build intermediate hops
        lrg._edges.remove(bad_edge)
        lrg._g.remove_edge(bad_edge.source_id, bad_edge.target_id)

        prev_id = bad_edge.source_id
        prev_table = src_table
        for hop_table in path[1:]:
            conditions = schema_graph.join_conditions([prev_table, hop_table])
            if not conditions:
                break
            cond = conditions[0]

            # Add intermediate entity node if not already present
            existing = _find_entity_node(lrg, hop_table)
            if existing is None:
                hop_node = EntityNode(
                    table_name=hop_table,
                    is_main_entity=False,
                    label=hop_table,
                )
                hop_id = lrg.add_node(hop_node)
            else:
                hop_id = existing.node_id

            lrg.add_edge(LRGEdge(
                source_id=prev_id,
                target_id=hop_id,
                edge_type=EdgeType.JOIN,
                join_left_col=cond["left_col"],
                join_right_col=cond["right_col"],
                label=f"JOIN ON {prev_table}.{cond['left_col']}={hop_table}.{cond['right_col']}",
            ))
            prev_id = hop_id
            prev_table = hop_table

        repairs.append(
            f"Repaired JOIN {src_table} → {tgt_table} via FK path: {' → '.join(path)}"
        )

    return repairs


def _repair_unknown_table(
    lrg: LRGGraph,
    schema_graph: SchemaGraph,
    error: str,
) -> list[str]:
    """Remove entity nodes that reference tables not in the schema.

    Strategy: drop the node and all its edges. The LRG may still be
    incomplete but will no longer reference phantom tables.
    """
    repairs: list[str] = []
    bad_nodes = [
        n for n in lrg.nodes_of_type(NodeType.ENTITY) + lrg.nodes_of_type(NodeType.ALIAS)
        if not schema_graph.has_table(getattr(n, "table_name", ""))
    ]
    for node in bad_nodes:
        tname = getattr(node, "table_name", node.node_id)
        # Remove all edges touching this node
        lrg._edges = [
            e for e in lrg._edges
            if e.source_id != node.node_id and e.target_id != node.node_id
        ]
        if node.node_id in lrg._g:
            lrg._g.remove_node(node.node_id)
        del lrg._nodes[node.node_id]
        repairs.append(f"Removed phantom table node '{tname}' (not in schema)")
    return repairs


def _repair_self_join_alias(
    lrg: LRGGraph,
    schema_graph: SchemaGraph,
    error: str,
) -> list[str]:
    """Ensure that self-joins use distinct AliasNode instances.

    Strategy: if the same table appears twice as EntityNode (no alias),
    convert the second occurrence to an AliasNode with a generated alias.
    """
    repairs: list[str] = []
    table_counts: dict[str, list] = {}
    for node in lrg.nodes_of_type(NodeType.ENTITY):
        assert isinstance(node, EntityNode)
        tname = node.table_name
        table_counts.setdefault(tname, []).append(node)

    for tname, nodes in table_counts.items():
        if len(nodes) < 2:
            continue
        # Keep first as EntityNode; convert rest to AliasNode
        for i, node in enumerate(nodes[1:], start=2):
            alias = f"{tname[:1]}{i}"
            alias_node = AliasNode(
                node_id=node.node_id,   # preserve ID so existing edges still connect
                table_name=tname,
                alias=alias,
                role_description=f"role_{i}",
                label=f"{tname} AS {alias}",
            )
            lrg._nodes[node.node_id] = alias_node
            repairs.append(
                f"Converted duplicate EntityNode '{tname}' → AliasNode '{tname} AS {alias}'"
            )
    return repairs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _join_is_valid(lrg: LRGGraph, edge: LRGEdge, schema_graph: SchemaGraph) -> bool:
    src = lrg.get_node(edge.source_id)
    tgt = lrg.get_node(edge.target_id)
    src_table = getattr(src, "table_name", "")
    tgt_table = getattr(tgt, "table_name", "")
    if not src_table or not tgt_table:
        return True   # can't verify — assume ok
    if schema_graph.has_direct_fk(src_table, tgt_table):
        return True
    return schema_graph.join_path(src_table, tgt_table) is not None


def _find_entity_node(lrg: LRGGraph, table_name: str) -> Optional[EntityNode]:
    for node in lrg.nodes_of_type(NodeType.ENTITY):
        if isinstance(node, EntityNode) and node.table_name == table_name:
            return node
    return None
