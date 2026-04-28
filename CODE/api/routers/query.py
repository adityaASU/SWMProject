"""POST /query — NL -> SQL (baseline or LRG)."""
from __future__ import annotations

import base64
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    db_id: str
    model: str = "lrg"  # "lrg" | "prompt_zero_shot" | "prompt_few_shot"
    data_dir: Optional[str] = None
    conversation_history: Optional[list[dict]] = None
    return_lrg: bool = False
    return_lrg_image: bool = False


class QueryResponse(BaseModel):
    question: str
    db_id: str
    model: str
    predicted_sql: str
    validation_errors: list[str] = []
    lrg_summary: Optional[str] = None
    lrg_graph: Optional[dict] = None
    lrg_image_b64: Optional[str] = None


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest):
    from src.config import load_config
    from src.llm.factory import create_llm
    from src.baseline.registry import create_baseline
    from src.schema.parser import SchemaParser
    from pathlib import Path

    cfg = load_config()
    data_dir = Path(req.data_dir) if req.data_dir else cfg.paths.spider_data

    # Parse schema
    parser = SchemaParser()
    try:
        schema = parser.auto_parse(data_dir, req.db_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Create model
    try:
        llm = create_llm(cfg.llm)
        model = create_baseline(req.model, llm)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Predict
    try:
        if req.model == "lrg":
            from src.lrg.pipeline import LRGText2SQL
            from src.schema.graph import SchemaGraph
            lrg_model = LRGText2SQL(llm)
            result, lrg, errors = lrg_model.predict_with_lrg(
                req.question, schema, req.conversation_history
            )
        else:
            result = model.predict(req.question, schema, req.conversation_history)
            lrg = None
            errors = []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    response = QueryResponse(
        question=req.question,
        db_id=req.db_id,
        model=req.model,
        predicted_sql=result.predicted_sql,
        validation_errors=errors,
    )

    if lrg is not None:
        response.lrg_summary = lrg.summary()
        if req.return_lrg:
            response.lrg_graph = lrg.to_dict()
        if req.return_lrg_image:
            from src.lrg.visualizer import render_lrg
            img_bytes = render_lrg(lrg, title=req.question[:80])
            response.lrg_image_b64 = base64.b64encode(img_bytes).decode()

    return response
