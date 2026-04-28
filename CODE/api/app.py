"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import query, schema, benchmark


def create_app() -> FastAPI:
    app = FastAPI(
        title="Text2SQL LRG API",
        description=(
            "Explainable Text-to-SQL with Logical Reasoning Graphs. "
            "Supports baseline and LRG-enhanced prediction, schema introspection, "
            "and benchmarking over Spider / CoSQL / custom datasets."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query.router)
    app.include_router(schema.router)
    app.include_router(benchmark.router)

    @app.get("/", tags=["health"])
    def root():
        return {"status": "ok", "service": "text2sql-lrg"}

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "healthy"}

    return app


app = create_app()
