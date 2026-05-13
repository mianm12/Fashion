from __future__ import annotations

from urllib.parse import quote

from fastapi.testclient import TestClient


def test_get_attribute_detail_returns_latest_trend_and_heat(
    client: TestClient,
) -> None:
    response = client.get("/api/attributes/colour_group_name::Black")

    assert response.status_code == 200
    payload = response.json()
    assert payload["attr_id"] == "colour_group_name::Black"
    assert payload["attr_value"] == "Black"
    assert payload["latest_trend"]["source_week"] == 10
    assert payload["latest_heat"]["week_id"] == 10


def test_get_attribute_detail_respects_source_week(client: TestClient) -> None:
    response = client.get(
        "/api/attributes/colour_group_name::Black",
        params={"source_week": 9},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["latest_trend"]["source_week"] == 9
    assert payload["latest_heat"]["week_id"] == 9


def test_get_attribute_heat_series_limits_recent_weeks(client: TestClient) -> None:
    response = client.get(
        "/api/attributes/colour_group_name::Black/heat-series",
        params={"source_week": 10, "weeks": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["week_id"] for item in payload["points"]] == [10]


def test_get_attribute_articles_returns_related_articles(client: TestClient) -> None:
    response = client.get(
        "/api/attributes/colour_group_name::Black/articles",
        params={"limit": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["article_id"] == "0000000001"
    assert payload["items"][0]["prod_name"] == "Black Shirt"


def test_get_attribute_supports_double_encoded_slash_id(client: TestClient) -> None:
    attr_id = "index_group_name::Baby/Children"
    encoded_attr_id = quote(quote(attr_id, safe=""), safe="")

    response = client.get(f"/api/attributes/{encoded_attr_id}")

    assert response.status_code == 200
    assert response.json()["attr_id"] == attr_id


def test_get_attribute_graph_returns_parent_and_child_edges(client: TestClient) -> None:
    response = client.get("/api/attributes/product_type_name::Shirt/graph")

    assert response.status_code == 200
    payload = response.json()
    node_ids = {node["id"] for node in payload["nodes"]}
    assert "product_type_name::Shirt" in node_ids
    assert "garment_group_name::Jersey Basic" in node_ids
    assert "colour_group_name::Black" in node_ids
    assert payload["edges"][0]["relation_type"] == "contains"


def test_missing_attribute_returns_not_found(client: TestClient) -> None:
    response = client.get("/api/attributes/colour_group_name::Missing")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"code": "not_found", "message": "未找到指定属性"}
    }
