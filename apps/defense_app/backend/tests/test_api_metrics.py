from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_metrics_summary_groups_metrics(client: TestClient) -> None:
    response = client.get("/api/metrics/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["groups"]["trend"][0]["metric_name"] == "ndcg_at_10"
    assert payload["groups"]["recommendation"][0]["metric_name"] == "map_at_12"


def test_get_trend_metrics_filters_by_split(client: TestClient) -> None:
    response = client.get("/api/metrics/trend", params={"split": "test"})

    assert response.status_code == 200
    payload = response.json()
    assert [item["model_or_method"] for item in payload["items"]] == ["lightgbm"]


def test_get_trend_metrics_defaults_to_valid_and_test(client: TestClient) -> None:
    response = client.get("/api/metrics/trend")

    assert response.status_code == 200
    assert {item["split"] for item in response.json()["items"]} == {"valid", "test"}


def test_get_trend_metrics_treats_empty_split_as_default(
    client: TestClient,
) -> None:
    response = client.get("/api/metrics/trend", params={"split": " "})

    assert response.status_code == 200
    assert {item["split"] for item in response.json()["items"]} == {"valid", "test"}


def test_get_recommendation_metrics_filters_by_split(client: TestClient) -> None:
    response = client.get("/api/metrics/recommendation", params={"split": "test"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["model_or_method"] == "pop_similarity_trend"


def test_get_recommendation_metrics_defaults_to_valid_and_test(
    client: TestClient,
) -> None:
    response = client.get("/api/metrics/recommendation")

    assert response.status_code == 200
    assert {item["split"] for item in response.json()["items"]} == {"valid", "test"}


def test_get_recommendation_metrics_treats_empty_split_as_default(
    client: TestClient,
) -> None:
    response = client.get("/api/metrics/recommendation", params={"split": " "})

    assert response.status_code == 200
    assert {item["split"] for item in response.json()["items"]} == {"valid", "test"}
