from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.presentation.builders import (
    build_article_attributes,
    build_articles,
    build_attribute_heat_series,
    build_attribute_hierarchy_edges,
    build_demo_users,
    build_metrics_summary,
    build_presentation_tables,
    build_recommendation_items,
    build_recommendation_score_components,
    build_trend_attributes,
)
from fashion_trend.presentation.contracts import CORE_TREND_ATTR_TYPES
from fashion_trend.presentation.extractors import PresentationSources
from fashion_trend.presentation.source_artifacts import collect_source_artifact_metadata


def _case_payload(
    customer_id: str = "000000abcdef123456",
    *,
    split: str = "test",
    cutoff_week: int = 10,
    label_week: int = 11,
) -> dict[str, object]:
    return {
        "customer_id": customer_id,
        "split": split,
        "cutoff_week": cutoff_week,
        "label_week": label_week,
        "hit_count": 2,
        "profile": [
            {
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "preference_score": 0.9,
            }
        ],
        "recommendations": [
            {
                "article_id": "0000000001",
                "rank": 1,
                "candidate_sources": "popularity,trend_union",
            }
        ],
    }


def _recommendation_rows(
    count: int = 12,
    *,
    cutoff_week: int = 10,
    label_week: int = 11,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "000000abcdef123456",
                "split": "test",
                "cutoff_week": cutoff_week,
                "label_week": label_week,
                "method": "pop_similarity_trend",
                "article_id": f"00000000{rank:02d}",
                "rank": rank,
                "score": 1.0 / rank,
                "pop_score": 0.1 * rank,
                "sim_score": 0.2 * rank,
                "trend_score": 0.3 * rank,
                "recent_score": 0.4 * rank,
                "candidate_sources": "popularity,trend_union",
            }
            for rank in range(1, count + 1)
        ]
    )


def test_build_demo_users_preserves_string_ids_and_stable_case_id() -> None:
    result = build_demo_users([_case_payload()])

    assert result.loc[0, "case_id"] == "demo_test_10_11_000000abcdef"
    assert result.loc[0, "customer_id"] == "000000abcdef123456"
    assert isinstance(result.loc[0, "customer_id"], str)
    tags = json.loads(result.loc[0, "primary_tags"])
    assert "命中用户" in tags
    assert "colour_group_name: Black" in tags


def test_build_recommendation_items_computes_hits_from_labels() -> None:
    labels = pd.DataFrame(
        [
            {
                "customer_id": "000000abcdef123456",
                "split": "test",
                "cutoff_week": 10,
                "label_week": 11,
                "article_id": "0000000005",
            }
        ]
    )

    result = build_recommendation_items(
        [_case_payload()],
        _recommendation_rows(),
        labels,
    )

    assert len(result) == 12
    assert result.loc[result["rank"].eq(5), "article_id"].item() == "0000000005"
    assert isinstance(result.loc[result["rank"].eq(5), "article_id"].item(), str)
    assert result.loc[result["rank"].eq(5), "is_hit"].item() == 1
    assert result.loc[result["rank"].eq(4), "is_hit"].item() == 0


def test_build_recommendation_items_rejects_incomplete_top12() -> None:
    with pytest.raises(ValueError, match="Top-12.*incomplete"):
        build_recommendation_items(
            [_case_payload()],
            _recommendation_rows(count=11),
            pd.DataFrame(
                columns=[
                    "customer_id",
                    "split",
                    "cutoff_week",
                    "label_week",
                    "article_id",
                ]
            ),
        )


def test_build_recommendation_items_rejects_non_contiguous_top12_ranks() -> None:
    rows = _recommendation_rows()
    rows["rank"] = list(range(2, 14))

    with pytest.raises(ValueError, match="Top-12.*rank"):
        build_recommendation_items(
            [_case_payload()],
            rows,
            pd.DataFrame(
                columns=[
                    "customer_id",
                    "split",
                    "cutoff_week",
                    "label_week",
                    "article_id",
                ]
            ),
        )


def test_build_recommendation_score_components_maps_final_score() -> None:
    result = build_recommendation_score_components(
        [_case_payload()],
        _recommendation_rows(),
    )

    first = result.loc[result["article_id"].eq("0000000001")].iloc[0]
    assert first["final_score"] == 1.0
    for column in [
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
        "final_score",
    ]:
        assert pd.Series(result[column]).map(pd.notna).all()


def test_build_trend_attributes_contains_core_attr_types_and_ranks() -> None:
    rows: list[dict[str, object]] = []
    for index, attr_type in enumerate(CORE_TREND_ATTR_TYPES):
        rows.extend(
            [
                {
                    "week_id": 10,
                    "attr_id": f"{attr_type}::low",
                    "attr_type": attr_type,
                    "attr_value": "low",
                    "split": "test",
                    "heat_t": 10 + index,
                    "pred_share_t1": 0.1,
                    "pred_target_growth": 0.2,
                    "is_trend_eligible_t": 1,
                },
                {
                    "week_id": 10,
                    "attr_id": f"{attr_type}::high",
                    "attr_type": attr_type,
                    "attr_value": "high",
                    "split": "test",
                    "heat_t": 20 + index,
                    "pred_share_t1": 0.2,
                    "pred_target_growth": 0.8,
                    "is_trend_eligible_t": 1,
                },
            ]
        )

    result = build_trend_attributes(pd.DataFrame(rows), source_week=10)

    assert set(result["attr_type"]) == set(CORE_TREND_ATTR_TYPES)
    assert set(result["source_week"]) == {10}
    assert set(result["target_week"]) == {11}
    for attr_type in CORE_TREND_ATTR_TYPES:
        typed = result[result["attr_type"].eq(attr_type)].sort_values("rank")
        assert typed["rank"].tolist() == [1, 2]
        assert typed.iloc[0]["attr_value"] == "high"
        assert typed.iloc[0]["heat_t"] == 20 + CORE_TREND_ATTR_TYPES.index(attr_type)
        assert typed.iloc[0]["is_trend_eligible_t"] == 1


def test_build_attribute_heat_series_uses_recent_window_and_predictions() -> None:
    trend_attributes = pd.DataFrame(
        [
            {
                "source_week": 10,
                "target_week": 11,
                "attr_id": "colour_group_name::Black",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "rank": 1,
                "heat_t": 100,
                "pred_share_t1": 0.2,
                "pred_target_growth": 0.7,
                "is_trend_eligible_t": 1,
            }
        ]
    )
    heat = pd.DataFrame(
        [
            {
                "week_id": str(week),
                "attr_id": "colour_group_name::Black",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_cnt": week * 10,
                "heat_share": week / 100,
            }
            for week in range(2, 11)
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "week_id": 9,
                "attr_id": "colour_group_name::Black",
                "pred_target_growth": 0.5,
                "pred_share_t1": 0.18,
                "target_growth": 0.4,
            },
            {
                "week_id": 10,
                "attr_id": "colour_group_name::Black",
                "pred_target_growth": 0.7,
                "pred_share_t1": 0.2,
                "target_growth": 0.6,
            },
        ]
    )

    result = build_attribute_heat_series(
        trend_attributes,
        heat,
        predictions,
        weeks=8,
    )

    assert result["week_id"].tolist() == list(range(3, 11))
    assert len(result) <= 8
    previous_week = result[result["week_id"].eq(9)].iloc[0]
    assert previous_week["pred_target_growth"] == 0.5
    assert previous_week["pred_share_t1"] == 0.18
    assert previous_week["actual_target_growth"] == 0.4
    source_week = result[result["week_id"].eq(10)].iloc[0]
    assert source_week["pred_target_growth"] == 0.7
    assert source_week["pred_share_t1"] == 0.2
    assert source_week["actual_target_growth"] == 0.6


def test_build_attribute_hierarchy_edges_joins_attr_values() -> None:
    hierarchy = pd.DataFrame(
        [
            {
                "parent_attr_id": "colour::Black",
                "child_attr_id": "product::Dress",
                "parent_attr_type": "colour_group_name",
                "child_attr_type": "product_type_name",
                "relation_type": "co_occurs",
                "edge_weight": 0.8,
            }
        ]
    )
    nodes = pd.DataFrame(
        [
            {
                "attr_id": "colour::Black",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
            },
            {
                "attr_id": "product::Dress",
                "attr_type": "product_type_name",
                "attr_value": "Dress",
            },
        ]
    )

    result = build_attribute_hierarchy_edges(hierarchy, nodes)

    assert result.loc[0, "parent_attr_value"] == "Black"
    assert result.loc[0, "child_attr_value"] == "Dress"


def test_build_metrics_summary_flattens_trend_and_recommendation_metrics() -> None:
    trend_payloads = [
        {
            "model_name": "lightgbm",
            "overall": {
                "valid": {
                    "mae": 0.1,
                    "rmse": 0.2,
                    "spearman": 0.3,
                    "ndcg_at_k": {"10": 0.4},
                    "precision_at_k": {"10": 0.5},
                    "recall_at_k": {"10": 0.6},
                },
                "test": {
                    "mae": 0.11,
                    "rmse": 0.21,
                    "spearman": 0.31,
                    "ndcg_at_k": {"10": 0.41},
                    "precision_at_k": {"10": 0.51},
                    "recall_at_k": {"10": 0.61},
                },
            },
        }
    ]
    recommendation_payloads = [
        {
            "method": "pop_similarity_trend",
            "metrics": {
                "test": {
                    "map_at_12": 0.01,
                    "recall_at_12": 0.02,
                    "hit_rate_at_12": 0.03,
                    "ndcg_at_12": 0.04,
                    "coverage": 0.05,
                    "user_count": 2,
                    "missing_recommendation_user_count": 0,
                }
            },
        }
    ]

    result = build_metrics_summary(trend_payloads, recommendation_payloads)

    domains = set(result["metric_domain"])
    assert domains == {"trend", "recommendation"}
    assert {"mae", "ndcg_at_10", "map_at_12", "hit_rate_at_12"} <= set(
        result["metric_name"]
    )
    assert result["display_order"].tolist() == sorted(result["display_order"].tolist())


def test_build_articles_rejects_missing_article_id() -> None:
    with pytest.raises(ValueError, match="articles_clean.*article_id"):
        build_articles(pd.DataFrame([{"prod_name": "Dress"}]))


def test_build_article_attributes_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="article_attributes.*attr_id"):
        build_article_attributes(
            pd.DataFrame(
                [
                    {
                        "article_id": "0000000001",
                        "attr_type": "colour_group_name",
                        "attr_value": "Black",
                    }
                ]
            )
        )


def test_collect_source_artifact_metadata_counts_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")
    json_path = tmp_path / "payload.json"
    json_path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    parquet_path = tmp_path / "sample.parquet"
    pd.DataFrame({"id": ["001", "002", "003"]}).to_parquet(parquet_path, index=False)

    result = collect_source_artifact_metadata(
        {
            "csv": csv_path,
            "json": json_path,
            "parquet": parquet_path,
        },
        required=["csv", "json", "parquet"],
    )

    assert result["csv"]["path"] == str(csv_path)
    assert result["csv"]["row_count"] == 2
    assert result["parquet"]["row_count"] == 3
    assert result["json"]["row_count"] is None
    assert result["csv"]["mtime"] > 0
    assert result["csv"]["size"] > 0


def test_collect_source_artifact_metadata_fails_for_missing_required_source(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="required.*missing_source"):
        collect_source_artifact_metadata(
            {"missing_source": missing},
            required=["missing_source"],
        )


def test_build_presentation_tables_writes_source_artifacts_metadata() -> None:
    sources = _minimal_presentation_sources(
        source_artifacts={
            "reports_manifest": {
                "path": "outputs/reports/manifest.json",
                "mtime": 1.0,
                "size": 10,
                "row_count": None,
            },
            "lightgbm_predictions": {
                "path": "outputs/models/lightgbm/predictions.csv",
                "mtime": 2.0,
                "size": 20,
                "row_count": 1,
            },
        }
    )

    metadata = build_presentation_tables(sources)["app_metadata"]
    values = dict(zip(metadata["key"], metadata["value"], strict=True))

    source_artifacts = json.loads(values["source_artifacts"])
    assert source_artifacts["lightgbm_predictions"]["path"].endswith("predictions.csv")
    assert source_artifacts["lightgbm_predictions"]["row_count"] == 1
    assert values["source_manifest_path"] == "outputs/reports/manifest.json"


def test_build_presentation_tables_includes_default_and_demo_source_weeks() -> None:
    sources = _minimal_presentation_sources(
        report_cases=[
            _case_payload(cutoff_week=9, label_week=10),
            _case_payload(cutoff_week=10, label_week=11),
        ],
        prediction_rows=[
            _prediction_row(9, pred_target_growth=0.2),
            _prediction_row(10, pred_target_growth=0.4),
        ],
        sample_rows=[
            _sample_row(9, heat_t=90.0),
            _sample_row(10, heat_t=100.0),
        ],
        heat_rows=[_heat_row(week, heat_cnt=week * 10) for week in range(2, 11)],
    )

    tables = build_presentation_tables(sources)

    trend_attributes = tables["trend_attributes"]
    assert trend_attributes["source_week"].tolist() == [9, 10]
    assert trend_attributes["target_week"].tolist() == [10, 11]
    assert not trend_attributes.duplicated(["source_week", "attr_type", "rank"]).any()
    heat_series = tables["attribute_heat_series"]
    assert set(heat_series["week_id"]) == set(range(2, 11))
    assert heat_series.groupby("attr_id")["week_id"].max().item() == 10


def _minimal_presentation_sources(
    *,
    report_cases: list[dict[str, object]] | None = None,
    prediction_rows: list[dict[str, object]] | None = None,
    sample_rows: list[dict[str, object]] | None = None,
    heat_rows: list[dict[str, object]] | None = None,
    source_artifacts: dict[str, dict[str, object]] | None = None,
) -> PresentationSources:
    case_payloads = report_cases or [_case_payload()]
    predictions = pd.DataFrame(prediction_rows or [_prediction_row(10)])
    samples = pd.DataFrame(sample_rows or [_sample_row(10)])
    recommendation_items = pd.concat(
        [
            _recommendation_rows(
                cutoff_week=int(case["cutoff_week"]),
                label_week=int(case["label_week"]),
            )
            for case in case_payloads
        ],
        ignore_index=True,
    )
    return PresentationSources(
        manifest={"warnings": [], "output_artifacts": {"figures": []}},
        report_cases=case_payloads,
        report_tables={},
        predictions=predictions,
        feature_importance=pd.DataFrame(),
        trend_metrics={
            "lightgbm": {
                "model_name": "lightgbm",
                "overall": {
                    "test": {
                        "mae": 0.1,
                        "rmse": 0.2,
                        "spearman": 0.3,
                        "ndcg_at_k": {"10": 0.4},
                        "precision_at_k": {"10": 0.5},
                        "recall_at_k": {"10": 0.6},
                    }
                },
            }
        },
        recommendation_metrics={
            "pop_similarity_trend": {
                "method": "pop_similarity_trend",
                "metrics": {
                    "test": {
                        "map_at_12": 0.01,
                        "recall_at_12": 0.02,
                        "hit_rate_at_12": 0.03,
                        "ndcg_at_12": 0.04,
                        "coverage": 0.05,
                        "user_count": 1,
                        "missing_recommendation_user_count": 0,
                    }
                },
            }
        },
        recommendation_items=recommendation_items,
        experiment={},
        evaluation_labels=pd.DataFrame(
            [
                {
                    "customer_id": str(case["customer_id"]),
                    "split": str(case["split"]),
                    "cutoff_week": int(case["cutoff_week"]),
                    "label_week": int(case["label_week"]),
                    "article_id": "0000000001",
                }
                for case in case_payloads
            ]
        ),
        user_profile=pd.DataFrame(
            [
                {
                    "customer_id": str(case["customer_id"]),
                    "split": str(case["split"]),
                    "cutoff_week": int(case["cutoff_week"]),
                    "label_week": int(case["label_week"]),
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                    "preference_score": 0.9,
                    "purchase_count": 3,
                    "last_purchase_week": 9,
                }
                for case in case_payloads
            ]
        ),
        attribute_week_heat=pd.DataFrame(heat_rows or [_heat_row(10)]),
        trend_model_samples=samples,
        article_nodes=pd.DataFrame(),
        attribute_nodes=pd.DataFrame(
            [
                {
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                }
            ]
        ),
        article_attributes=pd.DataFrame(
            [
                {
                    "article_id": "0000000001",
                    "article_node_id": "article::0000000001",
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                    "edge_type": "has_attribute",
                    "edge_weight": 1.0,
                }
            ]
        ),
        attribute_hierarchy_edges=pd.DataFrame(
            [
                {
                    "parent_attr_id": "colour_group_name::Black",
                    "child_attr_id": "colour_group_name::Black",
                    "parent_attr_type": "colour_group_name",
                    "child_attr_type": "colour_group_name",
                    "relation_type": "self",
                    "edge_weight": 1.0,
                }
            ]
        ),
        articles=pd.DataFrame(
            [
                {
                    "article_id": "0000000001",
                    "prod_name": "Dress",
                    "product_group_name": "Garment",
                    "product_type_name": "Dress",
                    "garment_group_name": "Dresses",
                    "colour_group_name": "Black",
                    "graphical_appearance_name": "Solid",
                    "department_name": "Dress",
                    "section_name": "Womens",
                    "index_name": "Ladieswear",
                    "index_group_name": "Ladieswear",
                }
            ]
        ),
        source_artifacts=source_artifacts,
    )


def _prediction_row(
    week_id: int,
    *,
    pred_target_growth: float = 0.4,
) -> dict[str, object]:
    return {
        "week_id": week_id,
        "attr_id": "colour_group_name::Black",
        "attr_type": "colour_group_name",
        "attr_value": "Black",
        "model_name": "lightgbm",
        "split": "test",
        "share_t": 0.1,
        "pred_share_t1": 0.2,
        "target_growth": 0.3,
        "pred_target_growth": pred_target_growth,
        "target_rank_in_type_t1": 1,
    }


def _sample_row(
    week_id: int,
    *,
    heat_t: float = 100.0,
) -> dict[str, object]:
    return {
        "week_id": week_id,
        "attr_id": "colour_group_name::Black",
        "attr_type": "colour_group_name",
        "attr_value": "Black",
        "heat_t": heat_t,
        "share_t": 0.1,
        "target_growth": 0.3,
        "history_total_heat_t": 500.0,
        "history_active_weeks_t": 5,
        "is_trend_eligible_t": 1,
    }


def _heat_row(
    week_id: int,
    *,
    heat_cnt: int = 100,
) -> dict[str, object]:
    return {
        "week_id": week_id,
        "attr_id": "colour_group_name::Black",
        "attr_type": "colour_group_name",
        "attr_value": "Black",
        "heat_cnt": heat_cnt,
    }
