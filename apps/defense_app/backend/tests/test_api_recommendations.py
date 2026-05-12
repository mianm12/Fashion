from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_get_demo_user_recommendations_returns_ranked_items(
    client: TestClient,
) -> None:
    response = client.get("/api/demo-users/case-001/recommendations")

    assert response.status_code == 200
    payload = response.json()
    assert [item["rank"] for item in payload["items"]] == [1, 2]
    assert payload["items"][0]["article"]["article_id"] == "0000000001"
    assert payload["items"][0]["is_hit"] is True


def test_get_recommendation_explanation_returns_profile_item_and_scores(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/demo-users/case-001/recommendations/0000000001/explanation"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"] == "case-001"
    assert payload["article"]["article_id"] == "0000000001"
    assert payload["score_components"]["final_score"] == 0.99
    assert {item["attr_id"] for item in payload["matching_trend_attributes"]} == {
        "colour_group_name::Black",
        "product_type_name::Shirt",
    }


def test_missing_recommendation_user_returns_not_found(client: TestClient) -> None:
    response = client.get("/api/demo-users/missing/recommendations")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"code": "not_found", "message": "未找到指定演示用户"}
    }


def test_missing_recommendation_article_returns_not_found(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/demo-users/case-001/recommendations/0000009999/explanation"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"code": "not_found", "message": "未找到指定推荐商品"}
    }


def test_missing_score_components_returns_not_found(
    monkeypatch: pytest.MonkeyPatch,
    seeded_db_path: Path,
) -> None:
    with sqlite3.connect(seeded_db_path) as connection:
        connection.execute(
            """
            delete from recommendation_score_components
            where case_id = ? and article_id = ?
            """,
            ("case-001", "0000000002"),
        )
        connection.commit()

    monkeypatch.setenv("DEFENSE_APP_DB_PATH", str(seeded_db_path))
    from app.main import app

    response = TestClient(app).get(
        "/api/demo-users/case-001/recommendations/0000000002/explanation"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"code": "not_found", "message": "未找到指定推荐商品"}
    }
