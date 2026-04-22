"""Schema parser: reads SQLite databases and Spider-format JSON into a unified SchemaInfo."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    is_primary_key: bool = False
    table: str = ""


@dataclass
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class SchemaInfo:
    db_id: str
    tables: dict[str, TableInfo] = field(default_factory=dict)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    def format_for_prompt(self) -> str:
        """Return a compact text representation suitable for LLM prompts."""
        lines: list[str] = [f"Database: {self.db_id}"]
        for tname, tinfo in self.tables.items():
            cols = ", ".join(
                f"{c.name} ({c.dtype}{'*' if c.is_primary_key else ''})"
                for c in tinfo.columns
            )
            lines.append(f"  Table {tname}: [{cols}]")
        if self.foreign_keys:
            lines.append("  Foreign Keys:")
            for fk in self.foreign_keys:
                lines.append(
                    f"    {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}"
                )
        return "\n".join(lines)


class SchemaParser:
    """Parses schemas from SQLite files or Spider-format JSON tables."""

    def from_sqlite(self, db_path: Path) -> SchemaInfo:
        """Extract schema directly from a SQLite database file."""
        db_id = db_path.stem
        schema = SchemaInfo(db_id=db_id)

        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            table_names = [row[0] for row in cursor.fetchall()]

            for tname in table_names:
                tinfo = TableInfo(name=tname)
                cursor.execute(f"PRAGMA table_info('{tname}')")
                for col_row in cursor.fetchall():
                    # (cid, name, type, notnull, dflt_value, pk)
                    col = ColumnInfo(
                        name=col_row[1],
                        dtype=col_row[2] or "TEXT",
                        is_primary_key=bool(col_row[5]),
                        table=tname,
                    )
                    tinfo.columns.append(col)
                schema.tables[tname] = tinfo

            for tname in table_names:
                cursor.execute(f"PRAGMA foreign_key_list('{tname}')")
                for fk_row in cursor.fetchall():
                    # (id, seq, table, from, to, ...)
                    fk = ForeignKey(
                        from_table=tname,
                        from_column=fk_row[3],
                        to_table=fk_row[2],
                        to_column=fk_row[4],
                    )
                    schema.foreign_keys.append(fk)
        finally:
            conn.close()

        return schema

    def from_spider_json(self, tables_json_path: Path, db_id: str) -> SchemaInfo:
        """Parse Spider-format tables.json for a specific db_id."""
        with open(tables_json_path) as f:
            all_schemas = json.load(f)

        entry = next((s for s in all_schemas if s["db_id"] == db_id), None)
        if entry is None:
            raise ValueError(f"db_id '{db_id}' not found in {tables_json_path}")

        return self._parse_spider_entry(entry)

    def from_spider_entry(self, entry: dict) -> SchemaInfo:
        """Parse a single Spider tables.json entry dict."""
        return self._parse_spider_entry(entry)

    def _parse_spider_entry(self, entry: dict) -> SchemaInfo:
        schema = SchemaInfo(db_id=entry["db_id"])
        table_names = entry["table_names_original"]
        col_names = entry["column_names_original"]  # [[table_idx, col_name], ...]
        col_types = entry["column_types"]
        pk_indices = set(entry.get("primary_keys", []))
        fk_pairs = entry.get("foreign_keys", [])

        for tname in table_names:
            schema.tables[tname] = TableInfo(name=tname)

        for col_idx, (tbl_idx, cname) in enumerate(col_names):
            if tbl_idx == -1:
                continue  # skip the wildcard (*) column at index 0
            tname = table_names[tbl_idx]
            col = ColumnInfo(
                name=cname,
                dtype=col_types[col_idx] if col_idx < len(col_types) else "text",
                is_primary_key=col_idx in pk_indices,
                table=tname,
            )
            schema.tables[tname].columns.append(col)

        for from_idx, to_idx in fk_pairs:
            from_tbl_idx, from_col = col_names[from_idx]
            to_tbl_idx, to_col = col_names[to_idx]
            fk = ForeignKey(
                from_table=table_names[from_tbl_idx],
                from_column=from_col,
                to_table=table_names[to_tbl_idx],
                to_column=to_col,
            )
            schema.foreign_keys.append(fk)

        return schema

    def auto_parse(self, db_dir: Path, db_id: str) -> SchemaInfo:
        """Try SQLite first, then fall back to tables.json in the same directory."""
        sqlite_path = db_dir / db_id / f"{db_id}.sqlite"
        if sqlite_path.exists():
            return self.from_sqlite(sqlite_path)

        tables_json = db_dir / "tables.json"
        if tables_json.exists():
            return self.from_spider_json(tables_json, db_id)

        raise FileNotFoundError(
            f"Cannot find schema for db_id='{db_id}'. "
            f"Expected SQLite at {sqlite_path} or tables.json at {tables_json}."
        )
