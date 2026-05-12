from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fashion_trend.presentation.schema import apply_schema

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture()
def seeded_db_path(tmp_path: Path) -> Path:
    database_path = tmp_path / "fashion_demo.sqlite"
    with sqlite3.connect(database_path) as connection:
        apply_schema(connection)
        _seed_database(connection)
    return database_path


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, seeded_db_path: Path) -> TestClient:
    monkeypatch.setenv("DEFENSE_APP_DB_PATH", str(seeded_db_path))
    from app.main import app

    return TestClient(app)


@pytest.fixture()
def mismatch_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    database_path = tmp_path / "mismatch.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "create table app_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "insert into app_metadata(key, value) values ('schema_version', 'old')"
        )
    monkeypatch.setenv("DEFENSE_APP_DB_PATH", str(database_path))
    from app.main import app

    return TestClient(app)


@pytest.fixture()
def missing_db_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("DEFENSE_APP_DB_PATH", str(tmp_path / "missing.sqlite"))
    from app.main import app

    return TestClient(app)


def _seed_database(connection: sqlite3.Connection) -> None:
    connection.executemany(
        """
        insert into app_metadata(key, value)
        values (?, ?)
        on conflict(key) do update set value = excluded.value
        """,
        [
            ("generated_at", "2026-05-12T00:00:00+00:00"),
            ("default_source_week", "10"),
        ],
    )
    connection.executemany(
        """
        insert into demo_users(
            case_id, customer_id, split, cutoff_week, label_week, hit_count,
            primary_tags, profile_summary, recommendation_summary
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "case-001",
                "000customer001",
                "test",
                10,
                11,
                2,
                "trend,high_hit",
                "prefers black shirts",
                "trend-aware recommendations",
            ),
            (
                "case-002",
                "000customer002",
                "test",
                10,
                11,
                0,
                "cold_start",
                "prefers blue dresses",
                "popular recommendations",
            ),
        ],
    )
    connection.executemany(
        """
        insert into user_profile_attributes(
            case_id, customer_id, attr_id, attr_type, attr_value,
            preference_score, purchase_count, last_purchase_week
        )
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "case-001",
                "000customer001",
                "colour_group_name::Black",
                "colour_group_name",
                "Black",
                0.8,
                3,
                10,
            ),
            (
                "case-001",
                "000customer001",
                "product_type_name::Shirt",
                "product_type_name",
                "Shirt",
                0.7,
                2,
                9,
            ),
        ],
    )
    connection.executemany(
        """
        insert into articles(
            article_id, prod_name, product_group_name, product_type_name,
            garment_group_name, colour_group_name, graphical_appearance_name,
            department_name, section_name, index_name, index_group_name
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "0000000001",
                "Black Shirt",
                "Garment Upper body",
                "Shirt",
                "Jersey Basic",
                "Black",
                "Solid",
                "Ladieswear",
                "Womens Everyday Basics",
                "Ladieswear",
                "Ladieswear",
            ),
            (
                "0000000002",
                "Blue Dress",
                "Garment Full body",
                "Dress",
                "Dresses Ladies",
                "Blue",
                "Stripe",
                "Ladieswear",
                "Womens Trend",
                "Ladieswear",
                "Ladieswear",
            ),
        ],
    )
    connection.executemany(
        """
        insert into article_attributes(article_id, attr_id, attr_type, attr_value)
        values (?, ?, ?, ?)
        """,
        [
            ("0000000001", "colour_group_name::Black", "colour_group_name", "Black"),
            ("0000000001", "product_type_name::Shirt", "product_type_name", "Shirt"),
            (
                "0000000001",
                "graphical_appearance_name::Solid",
                "graphical_appearance_name",
                "Solid",
            ),
            (
                "0000000001",
                "garment_group_name::Jersey Basic",
                "garment_group_name",
                "Jersey Basic",
            ),
            ("0000000002", "colour_group_name::Blue", "colour_group_name", "Blue"),
            ("0000000002", "product_type_name::Dress", "product_type_name", "Dress"),
        ],
    )
    connection.executemany(
        """
        insert into trend_attributes(
            source_week, target_week, attr_id, attr_type, attr_value, rank,
            heat_t, pred_share_t1, pred_target_growth, is_trend_eligible_t
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                10,
                11,
                "colour_group_name::Black",
                "colour_group_name",
                "Black",
                1,
                100.0,
                0.3,
                0.5,
                1,
            ),
            (
                9,
                10,
                "colour_group_name::Black",
                "colour_group_name",
                "Black",
                1,
                90.0,
                0.2,
                0.2,
                1,
            ),
            (
                10,
                11,
                "product_type_name::Shirt",
                "product_type_name",
                "Shirt",
                1,
                80.0,
                0.2,
                0.4,
                1,
            ),
        ],
    )
    connection.executemany(
        """
        insert into attribute_heat_series(
            attr_id, attr_type, attr_value, week_id, heat,
            actual_target_growth, pred_target_growth, pred_share_t1
        )
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "colour_group_name::Black",
                "colour_group_name",
                "Black",
                9,
                90.0,
                0.1,
                0.2,
                0.2,
            ),
            (
                "colour_group_name::Black",
                "colour_group_name",
                "Black",
                10,
                100.0,
                0.2,
                0.5,
                0.3,
            ),
            (
                "product_type_name::Shirt",
                "product_type_name",
                "Shirt",
                10,
                80.0,
                0.1,
                0.4,
                0.2,
            ),
        ],
    )
    connection.executemany(
        """
        insert into attribute_hierarchy_edges(
            parent_attr_id, child_attr_id, parent_attr_type, parent_attr_value,
            child_attr_type, child_attr_value, relation_type
        )
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "garment_group_name::Jersey Basic",
                "product_type_name::Shirt",
                "garment_group_name",
                "Jersey Basic",
                "product_type_name",
                "Shirt",
                "contains",
            ),
            (
                "product_type_name::Shirt",
                "colour_group_name::Black",
                "product_type_name",
                "Shirt",
                "colour_group_name",
                "Black",
                "co_occurs",
            ),
        ],
    )
    connection.executemany(
        """
        insert into recommendation_items(
            case_id, customer_id, article_id, rank, score, is_hit, candidate_sources
        )
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "case-001",
                "000customer001",
                "0000000001",
                1,
                0.99,
                1,
                "popularity,trend_union",
            ),
            ("case-001", "000customer001", "0000000002", 2, 0.75, 0, "popularity"),
        ],
    )
    connection.executemany(
        """
        insert into recommendation_score_components(
            case_id, article_id, pop_score, sim_score, trend_score, recent_score, final_score
        )
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("case-001", "0000000001", 0.1, 0.4, 0.3, 0.2, 0.99),
            ("case-001", "0000000002", 0.2, 0.2, 0.1, 0.1, 0.75),
        ],
    )
    connection.executemany(
        """
        insert into metrics_summary(
            metric_domain, model_or_method, split, metric_name, metric_value, display_order
        )
        values (?, ?, ?, ?, ?, ?)
        """,
        [
            ("trend", "lightgbm", "test", "ndcg_at_10", 0.5, 1),
            ("trend", "last_week", "valid", "ndcg_at_10", 0.3, 2),
            ("trend", "lightgbm", "all", "mape", 0.4, 3),
            ("recommendation", "pop_similarity_trend", "test", "map_at_12", 0.12, 1),
            ("recommendation", "recent_popularity", "valid", "map_at_12", 0.05, 2),
            ("recommendation", "pop_similarity_trend", "all", "hit_rate_at_12", 0.2, 3),
            ("data", "presentation", "all", "demo_user_count", 2.0, 1),
        ],
    )
    connection.commit()
