"""Deterministic SQL Synthesizer: LRGGraph -> SQL string.

No LLM is involved in this step — the graph structure drives generation.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.lrg.graph import LRGGraph
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
    NodeType,
    SubgraphNode,
)


class SQLSynthesizer:
    """Converts an LRGGraph into an executable SQL string deterministically."""

    def synthesize(self, lrg: LRGGraph) -> str:
        """Return a SQL query string from *lrg*."""
        try:
            return self._build_select(lrg)
        except Exception as exc:
            logger.error("SQL synthesis failed: %s", exc, exc_info=True)
            return f"-- Synthesis error: {exc}"

    def _build_select(self, lrg: LRGGraph) -> str:
        select_clause = self._select(lrg)
        from_clause = self._from(lrg)
        join_clause = self._joins(lrg)
        where_clause = self._where(lrg)
        group_clause = self._group_by(lrg)
        having_clause = self._having(lrg)

        parts = [f"SELECT {select_clause}", f"FROM {from_clause}"]
        if join_clause:
            parts.append(join_clause)
        if where_clause:
            parts.append(f"WHERE {where_clause}")
        if group_clause:
            parts.append(f"GROUP BY {group_clause}")
        if having_clause:
            parts.append(f"HAVING {having_clause}")

        # ── Professor feedback: each SubgraphNode = its own SELECT block ──
        for sq_node in lrg.nodes_of_type(NodeType.SUBGRAPH):
            assert isinstance(sq_node, SubgraphNode)
            sq_sql = self._subquery_sql(sq_node, lrg)
            if sq_sql:
                # Attach to WHERE if no WHERE yet, else append with AND
                subq_expr = f"{sq_node.operator} (\n  {sq_sql}\n)"
                if "WHERE" not in "\n".join(parts):
                    parts.append(f"WHERE {subq_expr}")
                else:
                    parts = [
                        p + f"\n  AND {subq_expr}" if p.startswith("WHERE") else p
                        for p in parts
                    ]

        return "\n".join(parts)

    def _subquery_sql(self, sq_node: SubgraphNode, outer_lrg: LRGGraph) -> str:
        """Synthesize SQL for an inner scoped subgraph.

        Professor feedback:
          - Scoped subgraph has its own entity/filter/agg nodes
          - Correlated bindings produce 'WHERE inner.col = outer.col'
          - subquery_type determines whether outer context is injected
        """
        if not sq_node.inner_lrg:
            return ""

        try:
            inner_lrg = LRGGraph.from_dict(sq_node.inner_lrg)
        except Exception as exc:
            logger.warning("Failed to reconstruct inner LRG: %s", exc)
            return ""

        # Build inner SELECT
        inner_select = self._select(inner_lrg) or "*"
        inner_from = self._from(inner_lrg)
        if not inner_from or inner_from == "unknown_table":
            return ""

        inner_joins = self._joins(inner_lrg)
        inner_where = self._where(inner_lrg)
        inner_group = self._group_by(inner_lrg)
        inner_having = self._having(inner_lrg)

        # Correlated bindings → extra WHERE conditions (nested_correlated only)
        binding_conditions: list[str] = []
        if sq_node.subquery_type == "nested_correlated":
            for b in sq_node.correlated_bindings:
                outer_t = b.get("outer_table", "")
                outer_c = b.get("outer_col", "")
                inner_t = b.get("inner_table", "")
                inner_c = b.get("inner_col", "")
                if outer_t and outer_c and inner_t and inner_c:
                    binding_conditions.append(f"{inner_t}.{inner_c} = {outer_t}.{outer_c}")

        all_where_parts = [p for p in [inner_where] + binding_conditions if p]
        combined_where = " AND ".join(all_where_parts)

        parts = [f"SELECT {inner_select}", f"FROM {inner_from}"]
        if inner_joins:
            parts.append(inner_joins)
        if combined_where:
            parts.append(f"WHERE {combined_where}")
        if inner_group:
            parts.append(f"GROUP BY {inner_group}")
        if inner_having:
            parts.append(f"HAVING {inner_having}")

        return "\n  ".join(parts)

    # ── SELECT ────────────────────────────────────────────────────────────────

    def _select(self, lrg: LRGGraph) -> str:
        items: list[str] = []
        multi_table = self._is_multi_table(lrg)

        # Aggregation expressions take priority
        for node in lrg.nodes_of_type(NodeType.AGGREGATION):
            assert isinstance(node, AggregationNode)
            # COUNT(*) — never qualify the wildcard with a table prefix
            if node.column_name == "*":
                inner = "*"
            else:
                inner = _col_ref(node.table_name, node.column_name, node.table_alias, multi_table)
            if node.function == AggregationFunction.COUNT_DISTINCT:
                expr = f"COUNT(DISTINCT {inner})"
            else:
                expr = f"{node.function.value}({inner})"
            # Only add alias if it looks like a meaningful user-specified name
            # (not generic auto-names like 'cnt', 'count', 'agg')
            _generic = {"cnt", "count", "agg", "total", "num", "sum", "avg", "min", "max"}
            if node.alias and node.alias.lower() not in _generic:
                expr += f" AS {node.alias}"
            items.append(expr)

        # Plain attribute nodes marked for SELECT
        for node in lrg.nodes_of_type(NodeType.ATTRIBUTE):
            assert isinstance(node, AttributeNode)
            if node.in_select:
                ref = _col_ref(node.table_name, node.column_name, None, multi_table)
                items.append(ref)

        return ", ".join(items) if items else "*"

    def _is_multi_table(self, lrg: LRGGraph) -> bool:
        """Return True if the query involves more than one table (needs qualification)."""
        tables = set()
        for n in lrg.nodes_of_type(NodeType.ENTITY) + lrg.nodes_of_type(NodeType.ALIAS):
            tables.add(getattr(n, "table_name", ""))
        return len(tables) > 1

    # ── FROM ──────────────────────────────────────────────────────────────────

    def _from(self, lrg: LRGGraph) -> str:
        """Return the primary FROM table (first main entity, no alias or with alias)."""
        mains = [n for n in lrg.nodes_of_type(NodeType.ENTITY) if isinstance(n, EntityNode) and n.is_main_entity]
        if not mains:
            entities = lrg.nodes_of_type(NodeType.ENTITY) + lrg.nodes_of_type(NodeType.ALIAS)
            if not entities:
                return "unknown_table"
            mains = [entities[0]]

        node = mains[0]
        if isinstance(node, AliasNode):
            return f"{node.table_name} {node.alias}"
        return node.table_name

    # ── JOIN ──────────────────────────────────────────────────────────────────

    def _joins(self, lrg: LRGGraph) -> str:
        join_edges = lrg.edges_of_type(EdgeType.JOIN)
        if not join_edges:
            return ""
        lines: list[str] = []
        seen_targets: set[str] = set()

        for edge in join_edges:
            tgt_node = lrg.get_node(edge.target_id)
            tgt_table = getattr(tgt_node, "table_name", "")
            tgt_alias = getattr(tgt_node, "alias", "") or ""

            table_expr = f"{tgt_table} {tgt_alias}" if tgt_alias else tgt_table

            # De-duplicate join targets
            key = f"{tgt_table}_{tgt_alias}"
            if key in seen_targets:
                continue
            seen_targets.add(key)

            src_node = lrg.get_node(edge.source_id)
            src_table = getattr(src_node, "table_name", "")
            src_alias = getattr(src_node, "alias", "") or ""

            left_ref = f"{src_alias or src_table}.{edge.join_left_col}"
            right_ref = f"{tgt_alias or tgt_table}.{edge.join_right_col}"
            lines.append(f"JOIN {table_expr} ON {left_ref} = {right_ref}")

        return "\n".join(lines)

    # ── WHERE ─────────────────────────────────────────────────────────────────

    def _where(self, lrg: LRGGraph) -> str:
        filters = [
            n for n in lrg.nodes_of_type(NodeType.FILTER)
            if isinstance(n, FilterNode) and not n.is_having
        ]
        conditions = [self._filter_expr(f, lrg) for f in filters]
        return " AND ".join(c for c in conditions if c)

    def _filter_expr(self, node: FilterNode, lrg: LRGGraph) -> str:
        multi_table = self._is_multi_table(lrg)
        col = _col_ref(node.table_name, node.column_name, node.alias, multi_table)
        op = node.operator

        if op == FilterOperator.IS_NULL:
            return f"{col} IS NULL"
        if op == FilterOperator.IS_NOT_NULL:
            return f"{col} IS NOT NULL"
        if op == FilterOperator.BETWEEN:
            return f"{col} BETWEEN {_quote(node.value)} AND {_quote(node.value2)}"
        if op in (FilterOperator.IN, FilterOperator.NOT_IN):
            if isinstance(node.value, list):
                vals = ", ".join(_quote(v) for v in node.value)
                return f"{col} {op.value} ({vals})"
            # Could be a subquery reference — emit placeholder
            return f"{col} {op.value} (/* subquery */)"

        return f"{col} {op.value} {_quote(node.value)}"

    # ── GROUP BY ──────────────────────────────────────────────────────────────

    def _group_by(self, lrg: LRGGraph) -> str:
        gb_nodes = lrg.nodes_of_type(NodeType.GROUPING)
        if not gb_nodes:
            return ""
        node = gb_nodes[0]
        assert isinstance(node, GroupingNode)
        multi_table = self._is_multi_table(lrg)
        return ", ".join(
            _col_ref(col["table"], col["column"], None, multi_table) for col in node.columns
        )

    # ── HAVING ────────────────────────────────────────────────────────────────

    def _having(self, lrg: LRGGraph) -> str:
        having_filters = [
            n for n in lrg.nodes_of_type(NodeType.FILTER)
            if isinstance(n, FilterNode) and n.is_having
        ]
        conditions = [self._filter_expr(f, lrg) for f in having_filters]
        return " AND ".join(c for c in conditions if c)



# ── Helpers ───────────────────────────────────────────────────────────────────

def _col_ref(table: str, column: str, alias: Optional[str] = None, qualify: bool = True) -> str:
    """Build a column reference.

    - If column is '*', always returns bare '*' (never 'table.*').
    - If qualify is False (single-table query), returns bare column name.
    - Otherwise uses alias if given, else table name as prefix.
    """
    if column == "*":
        return "*"
    if not qualify:
        return column
    prefix = alias or table
    return f"{prefix}.{column}" if prefix else column


def _quote(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return str(value)
