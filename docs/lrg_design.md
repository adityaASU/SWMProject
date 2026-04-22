# Logical Reasoning Graph — Design Specification

## Motivation

Standard Text2SQL models generate SQL directly from natural language via token-level prediction. This produces opaque outputs where errors are hard to localise. The LRG introduces an explicit intermediate representation that models the semantic operations a query requires, grounded in the database schema.

## Node Types

| Node Type | Maps to SQL | Description |
|-----------|-------------|-------------|
| `EntityNode` | `FROM` | A table reference. `is_main_entity=True` marks the primary FROM table. |
| `AliasNode` | `FROM ... AS alias` | One semantic role of a table in a self-join. Each role gets a distinct node even though both refer to the same schema table. |
| `AttributeNode` | `SELECT col` / filter target | A column reference. `in_select=True` means it appears in the SELECT clause. |
| `FilterNode` | `WHERE` / `HAVING` | A condition: table.column OP value. `is_having=True` places it in HAVING. |
| `AggregationNode` | `SELECT AGG(col)` | An aggregate function with table and column scope. |
| `GroupingNode` | `GROUP BY` | A list of (table, column) pairs for grouping. |
| `SubgraphNode` | Nested `SELECT` | Placeholder for a nested subquery. Contains a reference to an inner LRGGraph. |

## Edge Types

| Edge Type | Description |
|-----------|-------------|
| `JOIN` | Connects two entity/alias nodes. Carries `join_left_col` and `join_right_col` validated against the schema FK graph. |
| `SELECTS` | Entity → AttributeNode (column selected from this table). |
| `FILTER_OF` | Entity → FilterNode (WHERE condition on this table). |
| `HAVING` | Entity → FilterNode where `is_having=True`. |
| `AGG_OF` | Entity → AggregationNode. |
| `GROUP_BY` | GroupingNode → AggregationNode (groups scope an aggregation). |
| `SUBQUERY_OF` | Entity → SubgraphNode (inner query depends on outer entity). |

## Graph Invariants

1. **DAG**: The LRGGraph must be a directed acyclic graph. Cycles indicate circular dependencies which are invalid for SQL generation.
2. **Schema-grounded tables**: Every EntityNode and AliasNode must reference a table that exists in the schema.
3. **Valid join paths**: Every JOIN edge must correspond to a path in the schema FK graph (shortest-path via NetworkX).
4. **At least one entity**: A valid LRG must have at least one EntityNode or AliasNode (the FROM clause must reference something).

## Self-Join Handling

When a query requires two references to the same table (e.g., finding employees and their managers from the same `employees` table), the LRG uses two `AliasNode` instances:

```
AliasNode(table="employees", alias="e1", role="employee")
    ──JOIN(manager_id = e2.id)──►
AliasNode(table="employees", alias="e2", role="manager")
```

The SQL synthesizer generates `employees e1 JOIN employees e2 ON e1.manager_id = e2.id`. This avoids the common baseline failure of omitting aliases or creating ambiguous column references.

## Nested Subquery Handling

Nested subqueries are represented as `SubgraphNode` instances. The node carries a reference to an inner `LRGGraph`. During synthesis:

1. The inner LRG is synthesized first into a SELECT statement
2. The outer query embeds the inner SQL inside an IN/EXISTS/NOT IN clause

This preserves logical scope boundaries that flat token generation often violates.

## LRG Construction — Two-Stage Pipeline

**Stage 1 — LLM Structured Extraction**

The LRGBuilder sends the schema and question to the LLM with a fixed JSON schema. The LLM returns:
- `main_entities`: tables needed
- `select_attributes`: columns in SELECT
- `filters`: WHERE conditions
- `aggregations`: aggregate functions
- `group_by`: GROUP BY columns
- `subqueries`: nested query descriptions
- `join_hints`: explicit join table order
- `is_self_join`: boolean flag
- `has_subquery`: boolean flag

**Stage 2 — Graph Assembly with Schema Validation**

Python takes the extracted JSON and:
1. Creates Entity/AliasNode for each table (skipping unknown tables)
2. Derives JOIN edges via `SchemaGraph.join_path()` — FK traversal, not LLM guessing
3. Creates AttributeNode, FilterNode, AggregationNode, GroupingNode
4. Runs `LRGGraph.validate(schema_graph)` to catch remaining issues

Only the join structure in Stage 2 is fully deterministic. The semantic decomposition in Stage 1 still uses the LLM, but the LLM only needs to identify logical operations, not produce valid SQL.
