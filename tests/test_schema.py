"""Tests for schema parsing and schema graph."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.schema.parser import SchemaParser, SchemaInfo, ForeignKey
from src.schema.graph import SchemaGraph


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_sqlite_db(tmp_path: Path) -> Path:
    """Create a minimal SQLite database for testing."""
    db_dir = tmp_path / "test_db"
    db_dir.mkdir()
    db_path = db_dir / "test_db.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            dept_id INTEGER,
            FOREIGN KEY (dept_id) REFERENCES departments(id)
        );
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            title TEXT,
            dept_id INTEGER,
            FOREIGN KEY (dept_id) REFERENCES departments(id)
        );
    """)
    conn.close()
    return tmp_path


@pytest.fixture()
def simple_schema(simple_sqlite_db: Path) -> SchemaInfo:
    parser = SchemaParser()
    return parser.auto_parse(simple_sqlite_db, "test_db")


@pytest.fixture()
def simple_graph(simple_schema: SchemaInfo) -> SchemaGraph:
    return SchemaGraph(simple_schema)


# ── SchemaParser tests ─────────────────────────────────────────────────────────

def test_parser_reads_tables(simple_schema: SchemaInfo):
    assert set(simple_schema.table_names()) == {"departments", "employees", "projects"}


def test_parser_reads_columns(simple_schema: SchemaInfo):
    emp_cols = simple_schema.tables["employees"].column_names()
    assert "name" in emp_cols
    assert "dept_id" in emp_cols


def test_parser_reads_primary_keys(simple_schema: SchemaInfo):
    pk_cols = [c for c in simple_schema.tables["departments"].columns if c.is_primary_key]
    assert len(pk_cols) == 1
    assert pk_cols[0].name == "id"


def test_parser_reads_foreign_keys(simple_schema: SchemaInfo):
    fk_pairs = {(fk.from_table, fk.to_table) for fk in simple_schema.foreign_keys}
    assert ("employees", "departments") in fk_pairs


def test_parser_spider_json(tmp_path: Path):
    import json
    entry = {
        "db_id": "test_spider",
        "table_names_original": ["orders", "customers"],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "customer_id"], [1, "id"], [1, "name"]],
        "column_types": ["text", "number", "number", "number", "text"],
        "primary_keys": [1, 3],
        "foreign_keys": [[2, 3]],
    }
    json_file = tmp_path / "tables.json"
    json_file.write_text(json.dumps([entry]))

    parser = SchemaParser()
    schema = parser.from_spider_json(json_file, "test_spider")
    assert "orders" in schema.tables
    assert len(schema.foreign_keys) == 1
    assert schema.foreign_keys[0].from_column == "customer_id"


# ── SchemaGraph tests ─────────────────────────────────────────────────────────

def test_graph_has_all_tables(simple_graph: SchemaGraph):
    assert simple_graph.has_table("employees")
    assert simple_graph.has_table("departments")
    assert not simple_graph.has_table("nonexistent")


def test_graph_direct_fk(simple_graph: SchemaGraph):
    assert simple_graph.has_direct_fk("employees", "departments")
    assert not simple_graph.has_direct_fk("employees", "projects")


def test_graph_join_path_direct(simple_graph: SchemaGraph):
    path = simple_graph.join_path("employees", "departments")
    assert path is not None
    assert path[0] == "employees"
    assert path[-1] == "departments"


def test_graph_join_path_multi_hop(simple_graph: SchemaGraph):
    # employees -> departments -> projects (2 hops)
    path = simple_graph.join_path("employees", "projects")
    assert path is not None
    assert len(path) >= 3


def test_graph_join_path_unreachable(simple_graph: SchemaGraph):
    path = simple_graph.join_path("employees", "nonexistent_table")
    assert path is None


def test_graph_validate_valid_path(simple_graph: SchemaGraph):
    ok, err = simple_graph.validate_join_path(["employees", "departments"])
    assert ok
    assert err is None


def test_graph_validate_invalid_table(simple_graph: SchemaGraph):
    ok, err = simple_graph.validate_join_path(["employees", "ghost_table"])
    assert not ok
    assert "ghost_table" in err


def test_graph_join_conditions(simple_graph: SchemaGraph):
    path = ["employees", "departments"]
    conds = simple_graph.join_conditions(path)
    assert len(conds) == 1
    assert conds[0]["left_table"] == "employees"
    assert conds[0]["right_table"] == "departments"


def test_graph_to_dict(simple_graph: SchemaGraph):
    d = simple_graph.to_dict()
    assert d["db_id"] == "test_db"
    assert "employees" in d["tables"]
    assert len(d["foreign_keys"]) >= 1
