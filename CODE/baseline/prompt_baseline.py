"""Zero/few-shot prompt-only baseline: send schema + question directly to the LLM."""
from __future__ import annotations

import re
from typing import Optional

from src.baseline.base import BaseText2SQL, PredictionResult
from src.llm.base import BaseLLM
from src.schema.parser import SchemaInfo

_SYSTEM_PROMPT = """\
You are an expert SQL assistant. Given a database schema and a natural language question,
generate a single executable SQL query that answers the question.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences.
- Use only table and column names that exist in the provided schema.
- Use proper JOIN syntax based on foreign key relationships.
- For aggregations use GROUP BY when needed.
"""

_FEW_SHOT_EXAMPLES = """\
Example 1:
Schema: Table employees: [id (int*), name (text), dept_id (int)]
        Table departments: [id (int*), name (text)]
        FK: employees.dept_id -> departments.id
Question: How many employees are in each department?
SQL: SELECT d.name, COUNT(e.id) FROM employees e JOIN departments d ON e.dept_id = d.id GROUP BY d.name

Example 2:
Schema: Table orders: [id (int*), customer_id (int), amount (real), year (int)]
        Table customers: [id (int*), name (text)]
        FK: orders.customer_id -> customers.id
Question: Find customers who placed more than 5 orders.
SQL: SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name HAVING COUNT(o.id) > 5
"""


def _build_prompt(
    question: str,
    schema: SchemaInfo,
    conversation_history: Optional[list[dict]],
    use_few_shot: bool,
) -> str:
    parts = [_SYSTEM_PROMPT]

    if use_few_shot:
        parts.append(_FEW_SHOT_EXAMPLES)

    parts.append("--- Database Schema ---")
    parts.append(schema.format_for_prompt())

    if conversation_history:
        parts.append("\n--- Conversation History ---")
        for turn in conversation_history:
            parts.append(f"Q: {turn['question']}")
            if turn.get("sql"):
                parts.append(f"SQL: {turn['sql']}")

    parts.append("\n--- Current Question ---")
    parts.append(f"Question: {question}")
    parts.append("SQL:")

    return "\n".join(parts)


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and trim whitespace from a raw SQL response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence line
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    # Remove inline leading 'SQL:' prefix if model echoed it
    text = re.sub(r"^sql\s*:", "", text, flags=re.IGNORECASE).strip()
    return text


class PromptBaseline(BaseText2SQL):
    """Direct LLM prompting baseline — no intermediate representation."""

    def __init__(self, llm: BaseLLM, use_few_shot: bool = True) -> None:
        self._llm = llm
        self._use_few_shot = use_few_shot

    @property
    def model_name(self) -> str:
        suffix = "few_shot" if self._use_few_shot else "zero_shot"
        return f"prompt_baseline_{suffix}_{self._llm.name}"

    def predict(
        self,
        question: str,
        schema: SchemaInfo,
        conversation_history: Optional[list[dict]] = None,
    ) -> PredictionResult:
        prompt = _build_prompt(question, schema, conversation_history, self._use_few_shot)
        raw = self._llm.generate(prompt)
        sql = _clean_sql(raw)
        return PredictionResult(
            question=question,
            db_id=schema.db_id,
            predicted_sql=sql,
            model_name=self.model_name,
            raw_response=raw,
            conversation_history=conversation_history,
        )
