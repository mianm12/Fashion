from __future__ import annotations

import sqlite3
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
