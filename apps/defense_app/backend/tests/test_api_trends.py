from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fashion_trend.presentation.schema import apply_schema


def test_get_trends_returns_ranked_attributes(client: TestClient) -> None:
    response = client.get(
        "/api/trends",
        params={"source_week": 10, "attr_type": "colour_group_name", "limit": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_week"] == 10
    assert payload["target_week"] == 11
    assert payload["items"][0]["attr_id"] == "colour_group_name::Black"
    assert payload["items"][0]["rank"] == 1
    assert payload["items"][0]["pred_target_growth"] == 0.5


def test_get_trends_defaults_to_metadata_source_week(client: TestClient) -> None:
    response = client.get("/api/trends", params={"limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_week"] == 10
    assert payload["target_week"] == 11
    assert {item["source_week"] for item in payload["items"]} == {10}


def test_get_trends_without_attr_type_returns_core_types_with_per_type_limit(
    client: TestClient,
) -> None:
    response = client.get("/api/trends", params={"source_week": 10, "limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert [item["attr_type"] for item in payload["items"]] == [
        "colour_group_name",
        "product_type_name",
        "graphical_appearance_name",
        "garment_group_name",
    ]
    assert {item["rank"] for item in payload["items"]} == {1}


def test_get_trend_source_weeks_returns_default_and_sorted_weeks(
    client: TestClient,
) -> None:
    response = client.get("/api/trends/source-weeks")

    assert response.status_code == 200
    assert response.json() == {"default_source_week": 10, "items": [9, 10]}


def test_get_trend_summary_returns_visual_metric_payload(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/trends/summary",
        params={"source_week": 10, "limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_week"] == 10
    assert payload["target_week"] == 11
    assert payload["rising_attribute_count"] == 4
    assert payload["high_confidence_attribute_count"] == 4
    assert payload["top_k_average_pred_target_growth"] == pytest.approx(0.35)
    assert payload["covered_article_count"] == 1
    assert payload["model_status"] == "LightGBM stable"


def test_get_trend_evidence_returns_distribution_history_and_new_rows(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/trends/evidence",
        params={"source_week": 10, "limit": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_week"] == 10
    assert payload["target_week"] == 11
    assert sum(bucket["count"] for bucket in payload["distribution"]) == 4
    assert {
        (point["attr_type"], point["week_id"])
        for point in payload["top_history"]
    } >= {("colour_group_name", 9), ("colour_group_name", 10)}
    new_attr_ids = {item["attr_id"] for item in payload["new_high_potential"]}
    assert "product_type_name::Shirt" in new_attr_ids


def test_get_trend_detail_returns_core_rows_with_per_type_limit(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/trends/detail",
        params={"source_week": 10, "limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["attr_type"] for item in payload["items"]] == [
        "colour_group_name",
        "product_type_name",
        "graphical_appearance_name",
        "garment_group_name",
    ]
    assert {item["rank"] for item in payload["items"]} == {1}


def test_schema_version_mismatch_returns_structured_error(
    mismatch_client: TestClient,
) -> None:
    response = mismatch_client.get("/api/trends")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "schema_version_mismatch",
            "message": "展示库 schema_version 与应用不兼容",
        }
    }


def test_missing_database_returns_structured_error(
    missing_db_client: TestClient,
) -> None:
    response = missing_db_client.get("/api/trends")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "database_unavailable",
            "message": "展示库不存在，请先运行 src/18_build_defense_app_db.py",
        }
    }


def test_database_path_with_uri_characters_opens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "fashion?demo#1.sqlite"
    with sqlite3.connect(database_path) as connection:
        apply_schema(connection)

    monkeypatch.setenv("DEFENSE_APP_DB_PATH", str(database_path))
    from app.main import app

    response = TestClient(app).get("/api/trends")

    assert response.status_code == 200


def test_database_connection_can_be_used_from_worker_thread(
    seeded_db_path: Path,
) -> None:
    from app.core.database import open_database

    connection = open_database(seeded_db_path)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                lambda: connection.execute(
                    "select value from app_metadata where key = ?",
                    ("schema_version",),
                ).fetchone()[0]
            ).result()
    finally:
        connection.close()

    assert result == "defense_app_v1"
