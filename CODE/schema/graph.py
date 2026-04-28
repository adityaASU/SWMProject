"""NetworkX-based schema graph with FK path queries."""
from __future__ import annotations

from typing import Optional

import networkx as nx

from src.schema.parser import ForeignKey, SchemaInfo


class SchemaGraph:
    """Represents a relational database schema as a directed graph.

    Nodes: table names (str)
    Edges: directed FK relationships with metadata
    """

    def __init__(self, schema: SchemaInfo) -> None:
        self.schema = schema
        self._graph: nx.DiGraph = nx.DiGraph()
        self._build()

    def _build(self) -> None:
        for tname in self.schema.tables:
            self._graph.add_node(tname)

        for fk in self.schema.foreign_keys:
            self._graph.add_edge(
                fk.from_table,
                fk.to_table,
                from_col=fk.from_column,
                to_col=fk.to_column,
                label=f"{fk.from_column} -> {fk.to_column}",
            )
            # Add reverse edge so traversal is bidirectional
            if not self._graph.has_edge(fk.to_table, fk.from_table):
                self._graph.add_edge(
                    fk.to_table,
                    fk.from_table,
                    from_col=fk.to_column,
                    to_col=fk.from_column,
                    label=f"{fk.to_column} -> {fk.from_column}",
                    reverse=True,
                )

    # ── Queries ────────────────────────────────────────────────────────────────

    def has_table(self, table: str) -> bool:
        return self._graph.has_node(table)

    def has_direct_fk(self, table_a: str, table_b: str) -> bool:
        return self._graph.has_edge(table_a, table_b) or self._graph.has_edge(table_b, table_a)

    def join_path(self, from_table: str, to_table: str) -> Optional[list[str]]:
        """Return the shortest join path between two tables, or None if unreachable."""
        try:
            path = nx.shortest_path(self._graph.to_undirected(), from_table, to_table)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def join_conditions(self, path: list[str]) -> list[dict]:
        """Return JOIN ON conditions for consecutive pairs in *path*.

        Each dict has keys: left_table, left_col, right_table, right_col.
        """
        conditions = []
        undirected = self._graph.to_undirected(as_view=True)
        for i in range(len(path) - 1):
            left, right = path[i], path[i + 1]
            edge_data = self._get_edge_data(left, right)
            conditions.append(
                {
                    "left_table": left,
                    "left_col": edge_data["from_col"],
                    "right_table": right,
                    "right_col": edge_data["to_col"],
                }
            )
        return conditions

    def _get_edge_data(self, a: str, b: str) -> dict:
        """Get edge data between *a* and *b*, checking both directions."""
        if self._graph.has_edge(a, b):
            return self._graph.edges[a, b]
        if self._graph.has_edge(b, a):
            d = self._graph.edges[b, a]
            return {"from_col": d["to_col"], "to_col": d["from_col"]}
        raise ValueError(f"No FK edge between {a} and {b}")

    def validate_join_path(self, tables: list[str]) -> tuple[bool, Optional[str]]:
        """Check if a list of tables forms a valid join path via FK edges.

        Returns (is_valid, error_message).
        """
        for table in tables:
            if not self.has_table(table):
                return False, f"Table '{table}' does not exist in schema"

        for i in range(len(tables) - 1):
            a, b = tables[i], tables[i + 1]
            if self.join_path(a, b) is None:
                return False, f"No FK path between '{a}' and '{b}'"

        return True, None

    def neighbors(self, table: str) -> list[str]:
        """Return tables directly connected to *table* via a FK."""
        undirected = self._graph.to_undirected()
        if table not in undirected:
            return []
        return list(undirected.neighbors(table))

    def to_dict(self) -> dict:
        """Serialize schema graph for JSON output / API responses."""
        return {
            "db_id": self.schema.db_id,
            "tables": {
                tname: {
                    "columns": [
                        {
                            "name": c.name,
                            "dtype": c.dtype,
                            "is_primary_key": c.is_primary_key,
                        }
                        for c in tinfo.columns
                    ]
                }
                for tname, tinfo in self.schema.tables.items()
            },
            "foreign_keys": [
                {
                    "from_table": fk.from_table,
                    "from_column": fk.from_column,
                    "to_table": fk.to_table,
                    "to_column": fk.to_column,
                }
                for fk in self.schema.foreign_keys
            ],
            "edges": [
                {"from": u, "to": v, **data}
                for u, v, data in self._graph.edges(data=True)
                if not data.get("reverse", False)
            ],
        }
