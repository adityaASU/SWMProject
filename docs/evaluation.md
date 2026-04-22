# Evaluation Methodology

## Standard Accuracy Metrics

### Exact Match (EM)

Two SQL queries are considered an exact match after normalisation:
- Lowercase everything
- Collapse all whitespace to single spaces
- Strip trailing semicolons

EM is strict: even semantically equivalent queries with different clause ordering are counted as mismatches. This makes it suitable for structural correctness analysis.

**Implementation**: `src/evaluation/metrics.exact_match(predicted, gold) -> bool`

### Execution Accuracy (EX)

The predicted SQL and gold SQL are both executed against the actual SQLite database. The prediction is correct if the result sets are identical (row order ignored).

EX is more permissive than EM: it accepts alternative formulations that produce the same answer. This is the primary metric used in Spider leaderboard evaluation.

**Implementation**: `src/evaluation/metrics.execution_accuracy(predicted, gold, db_path) -> bool`

### Component-Level Scores

Each SQL clause (SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT) is extracted heuristically and compared independently. This supports targeted failure diagnosis.

**Implementation**: `src/evaluation/metrics.component_match(predicted, gold) -> dict[str, bool]`

## Failure Mode Categories

### 1. Schema Linking Errors

The predicted SQL references tables or columns that do not exist in the database schema. This indicates that the LLM hallucinated schema elements.

**Detection**: Parse the predicted SQL with sqlglot, extract all Table and Column references, check against the known schema.

### 2. Join Hallucination

The predicted SQL contains a JOIN between two tables for which no FK path exists in the schema graph. This indicates unconstrained join generation.

**Detection**: For each JOIN ... ON clause, extract the table pair and query `SchemaGraph.join_path()`. If no path exists, the join is hallucinated.

### 3. Nested Subquery Errors

The predicted SQL has a different number of nested subqueries than the gold SQL. This captures both missing nesting (collapsing hierarchical logic into flat queries) and over-nesting.

**Detection**: Count `sqlglot.expressions.Subquery` nodes in both parsed ASTs.

### 4. Self-Join Ambiguity

The gold SQL requires the same table to appear multiple times with different roles, but the predicted SQL omits one of the references (fails to generate correct aliases).

**Detection**: Count table name occurrences in both parsed ASTs. If a table appears multiple times in gold but only once in prediction, self-join aliasing failed.

### 5. Context Drift (CoSQL only)

In multi-turn dialogue, the gold SQL references tables from previous turns that the predicted SQL ignores entirely. This indicates the model failed to maintain conversational context.

**Detection**: Extract tables referenced in previous turns from conversation history. Check if tables present in both history and gold SQL are missing from the predicted SQL.

## Explainability Metrics

### Faithfulness

Measures the fraction of SQL clauses (SELECT, WHERE, JOIN, etc.) that correspond to at least one node in the LRG of the appropriate type.

- SELECT → AttributeNode or AggregationNode
- FROM / JOIN → EntityNode or AliasNode
- WHERE → FilterNode
- GROUP BY → GroupingNode
- HAVING → FilterNode (is_having=True)

**Range**: 0.0–1.0. A faithful LRG has all clauses traceable to graph nodes.

### Completeness

Estimates the fraction of logical operations implied by the natural language question that are represented in the LRG. Uses keyword heuristics:

- Aggregation keywords (how many, total, count, sum, average) → expect AggregationNode
- Filter keywords (where, more than, less than, with, has) → expect FilterNode
- Grouping keywords (each, per, every, group) → expect GroupingNode
- Subquery keywords (except, not in, who have) → expect SubgraphNode

**Range**: 0.0–1.0. A complete LRG covers all operations mentioned in the question.

### Error Traceability

For incorrect predictions, determines whether the LRG structure allows localisation of the failure to a specific node. This is the key advantage of LRG over black-box generation.

- Schema linking error → traceable to EntityNode with wrong table
- Join hallucination → traceable to JOIN edge between entity nodes
- Nested subquery error → traceable to SubgraphNode
- Self-join error → traceable to AliasNode
- Context drift → traceable to FilterNode carrying the drifted constraint

**Value**: boolean (true if the failure maps to a specific graph element).
