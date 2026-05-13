from pathlib import Path


FRONTEND_SRC = Path("apps/defense_app/frontend/src")


def _read(path: str) -> str:
    return (FRONTEND_SRC / path).read_text(encoding="utf-8")


def test_attribute_detail_uses_materialized_eight_week_window() -> None:
    source = _read("views/AttributeDetailView.vue")

    assert "最近周数" not in source
    assert "weeksInput" not in source
    assert "weeks: 8" in source


def test_graph_search_selection_clears_stale_attribute_query() -> None:
    source = _read("views/AttributeGraphView.vue")

    assert "delete nextQuery.attr_id" in source
    assert "query: nextQuery" in source


def test_recommendation_reason_preserves_return_query_context() -> None:
    source = _read("views/UserProfileReasonView.vue")

    assert "returnQuery" in source
    assert "...route.query" in source


def test_primary_demo_labels_are_chinese() -> None:
    recommendation_view = _read("views/RecommendationView.vue")
    week_table = _read("components/attribute/AttributeWeekDetailTable.vue")
    edge_table = _read("components/graph/GraphEdgeTable.vue")

    assert "<span>case_id</span>" not in recommendation_view
    assert "<span>customer</span>" not in recommendation_view
    assert "<span>hit_count</span>" not in recommendation_view
    assert "<span>window</span>" not in recommendation_view
    assert "<span>案例 ID</span>" in recommendation_view
    assert "<span>用户 ID</span>" in recommendation_view
    assert "<span>命中数</span>" in recommendation_view
    assert "<span>评估窗口</span>" in recommendation_view

    assert "<th>week_id</th>" not in week_table
    assert "<th>周</th>" in week_table
    assert "<th>预测增长</th>" in week_table

    assert "<span>source</span>" not in edge_table
    assert "<span>源节点</span>" in edge_table
    assert "<span>关系类型</span>" in edge_table
