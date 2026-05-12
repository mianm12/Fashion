from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_demo_users_sorts_by_hit_count(client: TestClient) -> None:
    response = client.get("/api/demo-users", params={"limit": 20})

    assert response.status_code == 200
    payload = response.json()
    assert [item["case_id"] for item in payload["items"]] == ["case-001", "case-002"]


def test_list_demo_users_filters_by_query_and_tag(client: TestClient) -> None:
    response = client.get(
        "/api/demo-users",
        params={"q": "000customer001", "tag": "trend", "limit": 20},
    )

    assert response.status_code == 200
    assert [item["case_id"] for item in response.json()["items"]] == ["case-001"]


def test_list_demo_users_treats_like_wildcards_as_literals(
    client: TestClient,
) -> None:
    response = client.get("/api/demo-users", params={"q": "%", "limit": 20})

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_get_demo_user_detail_returns_case_payload(client: TestClient) -> None:
    response = client.get("/api/demo-users/case-001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"] == "case-001"
    assert payload["customer_id"] == "000customer001"


def test_get_demo_user_profile_returns_profile_attributes(client: TestClient) -> None:
    response = client.get("/api/demo-users/case-001/profile")

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"] == "case-001"
    assert payload["items"][0]["preference_score"] == 0.8


def test_missing_demo_user_returns_not_found(client: TestClient) -> None:
    response = client.get("/api/demo-users/missing")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"code": "not_found", "message": "未找到指定演示用户"}
    }
