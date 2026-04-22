"""GET /schema/{db_id} — introspect a database schema."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/schema", tags=["schema"])


@router.get("/{db_id}")
def get_schema(
    db_id: str,
    data_dir: Optional[str] = Query(None, description="Override the data directory path"),
):
    from pathlib import Path
    from src.config import load_config
    from src.schema.parser import SchemaParser
    from src.schema.graph import SchemaGraph

    cfg = load_config()
    base = Path(data_dir) if data_dir else cfg.paths.spider_data

    parser = SchemaParser()
    try:
        schema = parser.auto_parse(base, db_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    sg = SchemaGraph(schema)
    return sg.to_dict()
