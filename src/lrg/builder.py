"""LRG Builder: NL + SchemaGraph -> LRGGraph via two-stage structured LLM extraction."""
from __future__ import annotations

import logging
from typing import Any, Optional

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
    LRGEdge,
    NodeType,
    SubgraphNode,
)
from src.llm.base import BaseLLM
from src.schema.graph import SchemaGraph
from src.schema.parser import SchemaInfo

logger = logging.getLogger(__name__)

# ── Extraction schema sent to the LLM ─────────────────────────────────────────

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "main_entities": {
            "type": "array",
            "description": "Primary tables needed to answer the question",
            "items": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "alias": {"type": "string", "description": "Leave empty unless self-join"},
                    "role": {"type": "string", "description": "Role description for self-join"},
                    "is_main": {"type": "boolean"},
                },
                "required": ["table", "is_main"],
            },
        },
        "select_attributes": {
            "type": "array",
            "description": "Columns that appear in the SELECT clause",
            "items": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "column": {"type": "string"},
                    "alias": {"type": "string"},
                },
                "required": ["table", "column"],
            },
        },
        "filters": {
            "type": "array",
            "description": "WHERE / HAVING conditions",
            "items": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "column": {"type": "string"},
                    "operator": {"type": "string"},
                    "value": {},
                    "value2": {},
                    "alias": {"type": "string"},
                    "is_having": {"type": "boolean"},
                },
                "required": ["table", "column", "operator"],
            },
        },
        "aggregations": {
            "type": "array",
            "description": "Aggregate functions",
            "items": {
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "enum": ["COUNT", "SUM", "AVG", "MIN", "MAX", "COUNT DISTINCT"],
                    },
                    "table": {"type": "string"},
                    "column": {"type": "string"},
                    "output_alias": {"type": "string"},
                    "table_alias": {"type": "string"},
                },
                "required": ["function", "table", "column"],
            },
        },
        "group_by": {
            "type": "array",
            "description": "GROUP BY columns",
            "items": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "column": {"type": "string"},
                },
                "required": ["table", "column"],
            },
        },
        "subqueries": {
            "type": "array",
            "description": "Nested subqueries needed",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["role", "description"],
            },
        },
        "join_hints": {
            "type": "array",
            "description": "Explicit join paths between tables (table names only)",
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "is_self_join": {"type": "boolean"},
        "has_subquery": {"type": "boolean"},
    },
    "required": ["main_entities", "select_attributes"],
}

_BUILDER_PROMPT_TEMPLATE = """\
You are a semantic parser. Extract logical components from the question and return ONLY a JSON object.

{schema_section}

{history_section}
Question: {question}

Rules:
- Only use tables and columns that exist in the schema above.
- operator must be one of: =, !=, <, <=, >, >=, LIKE, NOT LIKE, IN, NOT IN, IS NULL, IS NOT NULL, BETWEEN
- Return ONLY the JSON object below, with no explanation or extra text.

You MUST respond with a JSON object using EXACTLY these keys:
{{
  "main_entities": [
    {{"table": "<table_name>", "alias": "", "role": "", "is_main": true}}
  ],
  "select_attributes": [
    {{"table": "<table_name>", "column": "<column_name>", "alias": ""}}
  ],
  "aggregations": [
    {{"function": "COUNT", "table": "<table_name>", "column": "*", "output_alias": "cnt", "table_alias": ""}}
  ],
  "group_by": [
    {{"table": "<table_name>", "column": "<column_name>"}}
  ],
  "filters": [
    {{"table": "<table_name>", "column": "<column_name>", "operator": "=", "value": "<val>", "is_having": false}}
  ],
  "join_hints": [],
  "subqueries": [],
  "is_self_join": false,
  "has_subquery": false
}}

Example — "How many students are in each department?":
{{
  "main_entities": [{{"table": "student", "alias": "", "role": "", "is_main": true}}],
  "select_attributes": [{{"table": "student", "column": "dept_name", "alias": ""}}],
  "aggregations": [{{"function": "COUNT", "table": "student", "column": "*", "output_alias": "cnt", "table_alias": ""}}],
  "group_by": [{{"table": "student", "column": "dept_name"}}],
  "filters": [],
  "join_hints": [],
  "subqueries": [],
  "is_self_join": false,
  "has_subquery": false
}}

Now answer for the question above using the schema provided. Return ONLY the JSON object.
"""


def _build_extraction_prompt(
    question: str,
    schema: SchemaInfo,
    conversation_history: Optional[list[dict]],
) -> str:
    schema_section = "Database Schema:\n" + schema.format_for_prompt()
    history_section = ""
    if conversation_history:
        lines = ["Conversation History:"]
        for turn in conversation_history:
            lines.append(f"  Q: {turn['question']}")
            if turn.get("sql"):
                lines.append(f"  SQL: {turn['sql']}")
        history_section = "\n".join(lines) + "\n"
    return _BUILDER_PROMPT_TEMPLATE.format(
        schema_section=schema_section,
        history_section=history_section,
        question=question,
    )


class LRGBuilder:
    """Builds a Logical Reasoning Graph from a natural language question and schema."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def build(
        self,
        question: str,
        schema: SchemaInfo,
        schema_graph: SchemaGraph,
        conversation_history: Optional[list[dict]] = None,
    ) -> tuple[LRGGraph, list[str]]:
        """Extract logical components from *question* and assemble a validated LRGGraph.

        Returns:
            (lrg, validation_errors) — errors is empty if the LRG is valid.
        """
        prompt = _build_extraction_prompt(question, schema, conversation_history)
        try:
            extracted = self._llm.generate_structured(prompt, _EXTRACTION_SCHEMA)
            logger.debug("LLM extracted: %s", extracted)
        except Exception as exc:
            logger.error("LLM structured extraction failed: %s", exc)
            print(f"\n  [LRG Builder] ERROR: LLM structured extraction failed: {exc}")
            print("  This usually means the model does not support JSON output.")
            print("  For Ollama, use a general-purpose model (llama3.2, mistral, qwen2.5).")
            print("  For Gemini, check your API key and model name in .env\n")
            extracted = {}

        if not extracted.get("main_entities"):
            print(f"  [LRG Builder] WARNING: No entities extracted from LLM response.")
            print(f"  LLM backend: {self._llm.name}")
            print("  Check that your model is running and supports instruction-following.\n")

        lrg = self._assemble(extracted, schema, schema_graph)
        errors = lrg.validate(schema_graph)
        if errors:
            logger.warning("LRG validation errors: %s", errors)

        return lrg, errors

    # ── Assembly ──────────────────────────────────────────────────────────────

    def _assemble(self, raw: dict, schema: SchemaInfo, schema_graph: SchemaGraph) -> LRGGraph:
        lrg = LRGGraph()
        # Maps (table, alias_or_empty) -> node_id
        entity_map: dict[tuple[str, str], str] = {}

        # 1. Entity / alias nodes
        for ent in raw.get("main_entities", []):
            table = ent.get("table", "")
            alias = ent.get("alias", "") or ""
            role = ent.get("role", "") or ""
            is_main = ent.get("is_main", False)

            if not schema_graph.has_table(table):
                logger.warning("Unknown table '%s' from LLM — skipping", table)
                continue

            if alias:
                node = AliasNode(
                    table_name=table,
                    alias=alias,
                    role_description=role,
                    label=f"{table} AS {alias}",
                )
            else:
                node = EntityNode(
                    table_name=table,
                    is_main_entity=is_main,
                    label=table,
                )
            nid = lrg.add_node(node)
            entity_map[(table, alias)] = nid

        # 2. Determine join edges from schema traversal
        self._add_join_edges(lrg, entity_map, raw, schema_graph)

        # 3. SELECT attribute nodes
        for attr in raw.get("select_attributes", []):
            table = attr.get("table", "")
            column = attr.get("column", "")
            alias = attr.get("alias", "") or ""
            if not table or not column:
                continue
            attr_node = AttributeNode(
                table_name=table,
                column_name=column,
                alias=alias or None,
                in_select=True,
                label=f"{table}.{column}",
            )
            attr_id = lrg.add_node(attr_node)
            # Connect to owning entity
            owner_id = self._find_entity(entity_map, table)
            if owner_id:
                lrg.add_edge(LRGEdge(
                    source_id=owner_id,
                    target_id=attr_id,
                    edge_type=EdgeType.SELECTS,
                    label="selects",
                ))

        # 4. Aggregation nodes
        for agg in raw.get("aggregations", []):
            func_str = agg.get("function", "COUNT").upper()
            try:
                func = AggregationFunction(func_str)
            except ValueError:
                func = AggregationFunction.COUNT

            agg_node = AggregationNode(
                function=func,
                table_name=agg.get("table", ""),
                column_name=agg.get("column", "*"),
                alias=agg.get("output_alias") or None,
                table_alias=agg.get("table_alias") or None,
                label=f"{func.value}({agg.get('table', '')}.{agg.get('column', '*')})",
            )
            agg_id = lrg.add_node(agg_node)
            owner_id = self._find_entity(entity_map, agg.get("table", ""))
            if owner_id:
                lrg.add_edge(LRGEdge(
                    source_id=owner_id,
                    target_id=agg_id,
                    edge_type=EdgeType.AGG_OF,
                    label="aggregates",
                ))

        # 5. Filter nodes
        for filt in raw.get("filters", []):
            operator_str = filt.get("operator", "=")
            try:
                op = FilterOperator(operator_str)
            except ValueError:
                op = FilterOperator.EQ

            filter_node = FilterNode(
                table_name=filt.get("table", ""),
                column_name=filt.get("column", ""),
                operator=op,
                value=filt.get("value"),
                value2=filt.get("value2"),
                alias=filt.get("alias") or None,
                is_having=filt.get("is_having", False),
                label=f"{filt.get('table', '')}.{filt.get('column', '')} {operator_str} {filt.get('value', '')}",
            )
            filt_id = lrg.add_node(filter_node)
            owner_id = self._find_entity(entity_map, filt.get("table", ""))
            if owner_id:
                lrg.add_edge(LRGEdge(
                    source_id=owner_id,
                    target_id=filt_id,
                    edge_type=EdgeType.HAVING if filt.get("is_having") else EdgeType.FILTER_OF,
                    label="filters",
                ))

        # 6. Group by node
        group_cols = raw.get("group_by", [])
        if group_cols:
            gb_node = GroupingNode(
                columns=[{"table": g["table"], "column": g["column"]} for g in group_cols],
                label="GROUP BY " + ", ".join(f"{g['table']}.{g['column']}" for g in group_cols),
            )
            gb_id = lrg.add_node(gb_node)
            # Connect group_by to aggregations
            for agg_node in lrg.nodes_of_type(NodeType.AGGREGATION):
                lrg.add_edge(LRGEdge(
                    source_id=gb_id,
                    target_id=agg_node.node_id,
                    edge_type=EdgeType.GROUP_BY,
                    label="groups",
                ))

        # 7. Subquery nodes
        for sq in raw.get("subqueries", []):
            sq_node = SubgraphNode(
                role=sq.get("role", ""),
                label=f"Subquery: {sq.get('role', '')}",
            )
            sq_id = lrg.add_node(sq_node)
            # Connect to first entity as placeholder — builder caller can recurse
            if entity_map:
                first_entity_id = next(iter(entity_map.values()))
                lrg.add_edge(LRGEdge(
                    source_id=first_entity_id,
                    target_id=sq_id,
                    edge_type=EdgeType.SUBQUERY_OF,
                    label=sq.get("role", "subquery"),
                ))

        return lrg

    def _add_join_edges(
        self,
        lrg: LRGGraph,
        entity_map: dict[tuple[str, str], str],
        raw: dict,
        schema_graph: SchemaGraph,
    ) -> None:
        """Add JOIN edges based on LLM-provided join_hints, validated through schema_graph."""
        entities = list(entity_map.items())
        if len(entities) < 2:
            return

        # Use explicit join hints if provided
        join_hints: list[list[str]] = raw.get("join_hints", [])

        if join_hints:
            for path in join_hints:
                for i in range(len(path) - 1):
                    left_table, right_table = path[i], path[i + 1]
                    left_id = self._find_entity(entity_map, left_table)
                    right_id = self._find_entity(entity_map, right_table)
                    if not (left_id and right_id):
                        continue
                    join_conds = schema_graph.join_conditions(
                        schema_graph.join_path(left_table, right_table) or [left_table, right_table]
                    )
                    left_col = join_conds[0]["left_col"] if join_conds else ""
                    right_col = join_conds[0]["right_col"] if join_conds else ""
                    lrg.add_edge(LRGEdge(
                        source_id=left_id,
                        target_id=right_id,
                        edge_type=EdgeType.JOIN,
                        join_left_col=left_col,
                        join_right_col=right_col,
                        label=f"JOIN ON {left_table}.{left_col}={right_table}.{right_col}",
                    ))
        else:
            # Auto-infer joins between consecutive entities via FK path
            tables = [k[0] for k in entities]
            for i in range(len(tables) - 1):
                left_table, right_table = tables[i], tables[i + 1]
                path = schema_graph.join_path(left_table, right_table)
                if not path:
                    continue
                conditions = schema_graph.join_conditions(path)
                if not conditions:
                    continue
                cond = conditions[0]
                left_id = self._find_entity(entity_map, left_table)
                right_id = self._find_entity(entity_map, right_table)
                if left_id and right_id:
                    lrg.add_edge(LRGEdge(
                        source_id=left_id,
                        target_id=right_id,
                        edge_type=EdgeType.JOIN,
                        join_left_col=cond["left_col"],
                        join_right_col=cond["right_col"],
                        label=f"JOIN ON {left_table}.{cond['left_col']}={right_table}.{cond['right_col']}",
                    ))

    def _find_entity(self, entity_map: dict[tuple[str, str], str], table: str) -> Optional[str]:
        """Find entity node_id for *table* (matches with or without alias)."""
        # Exact match (no alias)
        if (table, "") in entity_map:
            return entity_map[(table, "")]
        # Match any alias variant
        for (t, a), nid in entity_map.items():
            if t == table:
                return nid
        return None
