from __future__ import annotations

from fastapi.testclient import TestClient


def test_search_articles_prioritizes_prefix_matches(client: TestClient) -> None:
    response = client.get("/api/articles/search", params={"q": "Black", "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["article_id"] == "0000000001"
    assert payload["items"][0]["prod_name"] == "Black Shirt"


def test_search_articles_requires_query(client: TestClient) -> None:
    response = client.get("/api/articles/search")

    assert response.status_code == 422
    assert response.json() == {
        "detail": {"code": "validation_error", "message": "请求参数不合法"}
    }


def test_search_articles_rejects_blank_query(client: TestClient) -> None:
    response = client.get("/api/articles/search", params={"q": "   "})

    assert response.status_code == 422
    assert response.json() == {
        "detail": {"code": "validation_error", "message": "请求参数不合法"}
    }


def test_search_articles_rejects_limit_over_max(client: TestClient) -> None:
    response = client.get(
        "/api/articles/search",
        params={"q": "Black", "limit": 51},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {"code": "validation_error", "message": "请求参数不合法"}
    }


def test_search_articles_treats_like_wildcards_as_literals(
    client: TestClient,
) -> None:
    response = client.get("/api/articles/search", params={"q": "%"})

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_get_article_detail_returns_attributes(client: TestClient) -> None:
    response = client.get("/api/articles/0000000001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["article"]["article_id"] == "0000000001"
    assert {item["attr_id"] for item in payload["attributes"]} >= {
        "colour_group_name::Black",
        "product_type_name::Shirt",
    }


def test_get_article_graph_returns_article_attribute_graph(client: TestClient) -> None:
    response = client.get("/api/articles/0000000001/graph")

    assert response.status_code == 200
    payload = response.json()
    assert payload["article"]["id"] == "article::0000000001"
    assert any(node["id"] == "colour_group_name::Black" for node in payload["nodes"])
    assert any(edge["source"] == "article::0000000001" for edge in payload["edges"])


def test_missing_article_returns_not_found(client: TestClient) -> None:
    response = client.get("/api/articles/9999999999")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"code": "not_found", "message": "未找到指定商品"}
    }
