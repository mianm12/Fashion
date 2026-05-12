from __future__ import annotations

import sqlite3

from fashion_trend.presentation.contracts import PRESENTATION_SCHEMA_VERSION


SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS demo_users (
    case_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    split TEXT NOT NULL,
    cutoff_week INTEGER NOT NULL,
    label_week INTEGER NOT NULL,
    hit_count INTEGER NOT NULL,
    primary_tags TEXT NOT NULL,
    profile_summary TEXT NOT NULL,
    recommendation_summary TEXT NOT NULL,
    UNIQUE(customer_id, split, cutoff_week, label_week)
);

CREATE TABLE IF NOT EXISTS user_profile_attributes (
    case_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    attr_id TEXT NOT NULL,
    attr_type TEXT NOT NULL,
    attr_value TEXT NOT NULL,
    preference_score REAL NOT NULL,
    purchase_count INTEGER NOT NULL,
    last_purchase_week INTEGER NOT NULL,
    PRIMARY KEY(case_id, attr_id)
);

CREATE TABLE IF NOT EXISTS recommendation_items (
    case_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    is_hit INTEGER NOT NULL,
    candidate_sources TEXT NOT NULL,
    PRIMARY KEY(case_id, rank)
);

CREATE TABLE IF NOT EXISTS recommendation_score_components (
    case_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    pop_score REAL NOT NULL,
    sim_score REAL NOT NULL,
    trend_score REAL NOT NULL,
    recent_score REAL NOT NULL,
    final_score REAL NOT NULL,
    PRIMARY KEY(case_id, article_id)
);

CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    prod_name TEXT,
    product_group_name TEXT,
    product_type_name TEXT,
    garment_group_name TEXT,
    colour_group_name TEXT,
    graphical_appearance_name TEXT,
    department_name TEXT,
    section_name TEXT,
    index_name TEXT,
    index_group_name TEXT
);

CREATE TABLE IF NOT EXISTS article_attributes (
    article_id TEXT NOT NULL,
    attr_id TEXT NOT NULL,
    attr_type TEXT NOT NULL,
    attr_value TEXT NOT NULL,
    PRIMARY KEY(article_id, attr_id)
);

CREATE TABLE IF NOT EXISTS trend_attributes (
    source_week INTEGER NOT NULL,
    target_week INTEGER NOT NULL,
    attr_id TEXT NOT NULL,
    attr_type TEXT NOT NULL,
    attr_value TEXT NOT NULL,
    rank INTEGER NOT NULL,
    heat_t REAL NOT NULL,
    pred_share_t1 REAL,
    pred_target_growth REAL,
    is_trend_eligible_t INTEGER NOT NULL,
    PRIMARY KEY(source_week, attr_type, rank)
);

CREATE TABLE IF NOT EXISTS attribute_heat_series (
    attr_id TEXT NOT NULL,
    attr_type TEXT NOT NULL,
    attr_value TEXT NOT NULL,
    week_id INTEGER NOT NULL,
    heat REAL NOT NULL,
    actual_target_growth REAL,
    pred_target_growth REAL,
    pred_share_t1 REAL,
    PRIMARY KEY(attr_id, week_id)
);

CREATE TABLE IF NOT EXISTS attribute_hierarchy_edges (
    parent_attr_id TEXT NOT NULL,
    child_attr_id TEXT NOT NULL,
    parent_attr_type TEXT NOT NULL,
    parent_attr_value TEXT NOT NULL,
    child_attr_type TEXT NOT NULL,
    child_attr_value TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    PRIMARY KEY(parent_attr_id, child_attr_id, relation_type)
);

CREATE TABLE IF NOT EXISTS metrics_summary (
    metric_domain TEXT NOT NULL,
    model_or_method TEXT NOT NULL,
    split TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    display_order INTEGER NOT NULL,
    PRIMARY KEY(metric_domain, model_or_method, split, metric_name)
);

CREATE TABLE IF NOT EXISTS report_assets (
    asset_name TEXT PRIMARY KEY,
    asset_type TEXT NOT NULL,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    static_url TEXT,
    description TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_demo_users_customer_window
    ON demo_users(customer_id, split, cutoff_week, label_week);
CREATE INDEX IF NOT EXISTS idx_recommendation_items_article
    ON recommendation_items(article_id);
CREATE INDEX IF NOT EXISTS idx_article_attributes_attr
    ON article_attributes(attr_id);
CREATE INDEX IF NOT EXISTS idx_trend_attributes_attr
    ON trend_attributes(attr_id);
CREATE INDEX IF NOT EXISTS idx_attribute_heat_series_week
    ON attribute_heat_series(week_id);
"""


def apply_schema(connection: sqlite3.Connection) -> None:
    """Create the presentation database schema and record its version."""
    existing_version = read_schema_version(connection)
    if (
        existing_version is not None
        and existing_version != PRESENTATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "presentation schema version mismatch: "
            f"found {existing_version}, expected {PRESENTATION_SCHEMA_VERSION}"
        )

    with connection:
        connection.executescript(SCHEMA_DDL)
        connection.execute(
            """
            INSERT INTO app_metadata(key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (PRESENTATION_SCHEMA_VERSION,),
        )


def read_schema_version(connection: sqlite3.Connection) -> str | None:
    """Return the recorded presentation schema version, if the table exists."""
    if not _metadata_table_exists(connection):
        return None
    row = connection.execute(
        "SELECT value FROM app_metadata WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def _metadata_table_exists(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'app_metadata'
        """
    ).fetchone()
    return row is not None
