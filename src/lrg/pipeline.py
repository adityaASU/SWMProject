"""LRG Text2SQL pipeline — implements BaseText2SQL using the full LRG framework."""
from __future__ import annotations

from typing import Optional

from src.baseline.base import BaseText2SQL, PredictionResult
from src.llm.base import BaseLLM
from src.lrg.builder import LRGBuilder
from src.lrg.graph import LRGGraph
from src.lrg.synthesizer import SQLSynthesizer
from src.schema.graph import SchemaGraph
from src.schema.parser import SchemaInfo


class LRGText2SQL(BaseText2SQL):
    """End-to-end Text2SQL using Logical Reasoning Graphs.

    Pipeline:
      1. Build SchemaGraph from SchemaInfo
      2. LRGBuilder extracts logical components via LLM structured output
      3. SQLSynthesizer deterministically converts LRG -> SQL
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm
        self._builder = LRGBuilder(llm)
        self._synthesizer = SQLSynthesizer()

    @property
    def model_name(self) -> str:
        return f"lrg_{self._llm.name}"

    def predict(
        self,
        question: str,
        schema: SchemaInfo,
        conversation_history: Optional[list[dict]] = None,
    ) -> PredictionResult:
        schema_graph = SchemaGraph(schema)
        lrg, validation_errors = self._builder.build(
            question, schema, schema_graph, conversation_history
        )
        sql = self._synthesizer.synthesize(lrg)

        return PredictionResult(
            question=question,
            db_id=schema.db_id,
            predicted_sql=sql,
            model_name=self.model_name,
            raw_response="",
            conversation_history=conversation_history,
            metadata={
                "lrg": lrg.to_dict(),
                "lrg_summary": lrg.summary(),
                "validation_errors": validation_errors,
            },
        )

    def predict_with_lrg(
        self,
        question: str,
        schema: SchemaInfo,
        conversation_history: Optional[list[dict]] = None,
    ) -> tuple[PredictionResult, LRGGraph, list[str]]:
        """Like :meth:`predict` but also returns the LRGGraph and validation errors."""
        schema_graph = SchemaGraph(schema)
        lrg, validation_errors = self._builder.build(
            question, schema, schema_graph, conversation_history
        )
        sql = self._synthesizer.synthesize(lrg)
        result = PredictionResult(
            question=question,
            db_id=schema.db_id,
            predicted_sql=sql,
            model_name=self.model_name,
            raw_response="",
            conversation_history=conversation_history,
            metadata={
                "lrg": lrg.to_dict(),
                "lrg_summary": lrg.summary(),
                "validation_errors": validation_errors,
            },
        )
        return result, lrg, validation_errors
