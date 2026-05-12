from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from fashion_trend.foundation.io import remove_file_if_exists
from fashion_trend.presentation.contracts import (
    CORE_TREND_ATTR_TYPES,
    PRESENTATION_SCHEMA_VERSION,
)
from fashion_trend.presentation.schema import apply_schema, read_schema_version

REQUIRED_TABLES = (
    "app_metadata",
    "demo_users",
    "user_profile_attributes",
    "recommendation_items",
    "recommendation_score_components",
    "articles",
    "article_attributes",
    "trend_attributes",
    "attribute_heat_series",
    "attribute_hierarchy_edges",
    "metrics_summary",
    "report_assets",
)

REQUIRED_NON_EMPTY_TABLES = (
    "demo_users",
    "recommendation_items",
    "recommendation_score_components",
    "articles",
    "trend_attributes",
    "metrics_summary",
)

TOP12_RANKS = set(range(1, 13))

RECOMMENDATION_RANKS_SQL = """
SELECT case_id, rank
FROM recommendation_items
ORDER BY case_id, rank
"""

MISSING_RECOMMENDATIONS_SQL = """
SELECT case_id
FROM demo_users
WHERE case_id NOT IN (
    SELECT case_id FROM recommendation_items
)
"""

MISSING_SCORE_COMPONENTS_SQL = """
SELECT items.case_id, items.article_id
FROM recommendation_items AS items
LEFT JOIN recommendation_score_components AS scores
    ON scores.case_id = items.case_id
   AND scores.article_id = items.article_id
WHERE scores.case_id IS NULL
ORDER BY items.case_id, items.rank
"""

DUPLICATE_SCORE_COMPONENTS_SQL = """
SELECT case_id, article_id, COUNT(*) AS row_count
FROM recommendation_score_components
GROUP BY case_id, article_id
HAVING row_count > 1
ORDER BY case_id, article_id
"""


def write_presentation_database(
    tables: Mapping[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Write presentation tables to SQLite and atomically publish the database."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    remove_file_if_exists(tmp_path)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(tmp_path)
        apply_schema(connection)
        for table_name, frame in tables.items():
            frame.to_sql(table_name, connection, if_exists="append", index=False)
        validate_database(connection)
        connection.close()
        tmp_path.replace(output_path)
    except Exception:
        if connection is not None:
            connection.close()
        remove_file_if_exists(tmp_path)
        raise
    finally:
        remove_file_if_exists(tmp_path)


def validate_database(connection: sqlite3.Connection) -> None:
    """Validate the defense app SQLite schema and critical data invariants."""
    schema_version = read_schema_version(connection)
    if schema_version != PRESENTATION_SCHEMA_VERSION:
        raise ValueError(
            "presentation schema version mismatch: "
            f"found {schema_version}, expected {PRESENTATION_SCHEMA_VERSION}"
        )

    existing_tables = _existing_tables(connection)
    missing_tables = sorted(set(REQUIRED_TABLES) - existing_tables)
    if missing_tables:
        raise ValueError(f"presentation database missing tables: {missing_tables}")

    empty_tables = [
        table_name
        for table_name in REQUIRED_NON_EMPTY_TABLES
        if _table_count(connection, table_name) == 0
    ]
    if empty_tables:
        raise ValueError(
            "presentation database required tables are empty: " f"{empty_tables}"
        )

    _validate_recommendation_top12(connection)
    _validate_recommendation_score_components(connection)
    _validate_core_trend_attr_types(connection)


def _existing_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _table_count(connection: sqlite3.Connection, table_name: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0])


def _validate_recommendation_top12(connection: sqlite3.Connection) -> None:
    rows = connection.execute(RECOMMENDATION_RANKS_SQL).fetchall()
    ranks_by_case: dict[str, set[int]] = {}
    for case_id, rank in rows:
        ranks_by_case.setdefault(str(case_id), set()).add(int(rank))

    invalid = {
        case_id: sorted(ranks)
        for case_id, ranks in ranks_by_case.items()
        if ranks != TOP12_RANKS
    }
    if invalid:
        raise ValueError(f"Top-12 recommendation ranks invalid: {invalid}")

    missing_cases = [
        str(row[0]) for row in connection.execute(MISSING_RECOMMENDATIONS_SQL)
    ]
    if missing_cases:
        raise ValueError(
            "Top-12 recommendation items missing for demo cases: " f"{missing_cases}"
        )


def _validate_recommendation_score_components(
    connection: sqlite3.Connection,
) -> None:
    missing_components = [
        {"case_id": str(row[0]), "article_id": str(row[1])}
        for row in connection.execute(MISSING_SCORE_COMPONENTS_SQL)
    ]
    if missing_components:
        raise ValueError(
            "recommendation score components missing for items: "
            f"{missing_components[:3]}"
        )

    duplicate_components = [
        {"case_id": str(row[0]), "article_id": str(row[1]), "row_count": int(row[2])}
        for row in connection.execute(DUPLICATE_SCORE_COMPONENTS_SQL)
    ]
    if duplicate_components:
        raise ValueError(
            "recommendation score components contain duplicate keys: "
            f"{duplicate_components[:3]}"
        )


def _validate_core_trend_attr_types(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT DISTINCT attr_type FROM trend_attributes"
    ).fetchall()
    actual = {str(row[0]) for row in rows}
    missing = sorted(set(CORE_TREND_ATTR_TYPES) - actual)
    if missing:
        raise ValueError(f"trend_attributes missing core attr types: {missing}")
