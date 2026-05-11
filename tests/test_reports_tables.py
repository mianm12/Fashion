from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.reports.tables import (
    REPORT_TABLE_COLUMNS,
    REPORT_TABLE_SORT_COLUMNS,
    RECOMMENDATION_METHOD_METRICS_COLUMNS,
    TREND_MODEL_METRICS_COLUMNS,
    build_report_table,
    build_recommendation_method_metrics_table,
    build_trend_model_metrics_table,
    write_report_table,
)


def test_report_table_contracts_cover_design_outputs() -> None:
    assert set(REPORT_TABLE_COLUMNS) == {
        "data_artifact_summary",
        "time_split_summary",
        "attribute_graph_summary",
        "trend_feature_summary",
        "trend_model_metrics",
        "trend_metrics_by_attr_type",
        "recommendation_method_metrics",
        "recommendation_experiment_summary",
    }
    assert set(REPORT_TABLE_SORT_COLUMNS) == set(REPORT_TABLE_COLUMNS)


def test_build_trend_model_metrics_table_uses_contract_order() -> None:
    rows = [
        {
            "model_name": "lightgbm",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-1",
        }
    ]

    table = build_trend_model_metrics_table(rows)

    assert tuple(table.columns) == TREND_MODEL_METRICS_COLUMNS
    assert table.loc[0, "model_name"] == "lightgbm"


def test_build_recommendation_method_metrics_table_uses_contract_order() -> None:
    rows = [
        {
            "method": "pop_similarity_trend",
            "split": "valid",
            "map_at_12": 0.1,
            "recall_at_12": 0.2,
            "hit_rate_at_12": 0.3,
            "ndcg_at_12": 0.4,
            "coverage": 0.5,
            "user_count": 10,
            "missing_recommendation_user_count": 0,
        }
    ]

    table = build_recommendation_method_metrics_table(rows)

    assert tuple(table.columns) == RECOMMENDATION_METHOD_METRICS_COLUMNS
    assert table.loc[0, "method"] == "pop_similarity_trend"


def test_build_report_table_selects_each_design_contract() -> None:
    samples = {
        "data_artifact_summary": {
            "section": "trend",
            "artifact": "trend_model_samples",
            "path": "data/processed/features/trend_model_samples.parquet",
            "row_count": 59200,
            "column_count": 22,
            "paper_usage": "趋势模型样本规模说明",
        },
        "time_split_summary": {
            "domain": "trend",
            "split": "test",
            "week_start": 96,
            "week_end": 104,
            "week_count": 8,
            "row_count": 5920,
            "attribute_count": 740,
            "user_count": 0,
        },
        "attribute_graph_summary": {
            "entity_type": "article",
            "attr_type": "",
            "relation_type": "article_attribute",
            "count": 105542,
            "path": "data/processed/graph/edges_article_attribute.csv",
            "paper_usage": "属性图规模说明",
        },
        "trend_feature_summary": {
            "feature_group": "lag",
            "feature_name": "lag_1_heat",
            "source_table": "trend_model_samples",
            "model_input": True,
            "description": "上一周属性热度",
        },
        "trend_model_metrics": {
            "model_name": "lightgbm",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-1",
        },
        "trend_metrics_by_attr_type": {
            "model_name": "lightgbm",
            "split": "test",
            "attr_type": "colour_group_name",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
        },
        "recommendation_method_metrics": {
            "method": "pop_similarity_trend",
            "split": "test",
            "map_at_12": 0.1,
            "recall_at_12": 0.2,
            "hit_rate_at_12": 0.3,
            "ndcg_at_12": 0.4,
            "coverage": 0.5,
            "user_count": 10,
            "missing_recommendation_user_count": 0,
        },
        "recommendation_experiment_summary": {
            "section": "search_results",
            "rank": 1,
            "method": "pop_similarity_trend",
            "split": "valid",
            "pop_score": 0.2,
            "sim_score": 0.2,
            "trend_score": 0.1,
            "recent_score": 0.5,
            "map_at_12": 0.1,
            "recall_at_12": 0.2,
            "hit_rate_at_12": 0.3,
            "ndcg_at_12": 0.4,
            "coverage": 0.5,
        },
    }

    for table_name, row in samples.items():
        table = build_report_table([row], table_name=table_name)
        assert tuple(table.columns) == REPORT_TABLE_COLUMNS[table_name]


def test_build_report_table_rejects_empty_rows_with_table_name() -> None:
    with pytest.raises(ValueError, match="trend_model_metrics.*无数据"):
        build_report_table([], table_name="trend_model_metrics")


def test_build_report_table_rejects_unknown_table_name() -> None:
    with pytest.raises(ValueError, match="unknown_table"):
        build_report_table([{"model_name": "lightgbm"}], table_name="unknown_table")


def test_build_report_table_reports_missing_columns_with_table_name() -> None:
    with pytest.raises(ValueError, match="trend_model_metrics.*run_id"):
        build_report_table(
            [
                {
                    "model_name": "lightgbm",
                    "split": "test",
                    "mae": 0.1,
                    "rmse": 0.2,
                    "spearman": 0.3,
                    "ndcg_at_10": 0.4,
                    "precision_at_10": 0.5,
                    "recall_at_10": 0.6,
                }
            ],
            table_name="trend_model_metrics",
        )


def test_build_report_table_sorts_by_contract_with_stable_tie_order() -> None:
    rows = [
        {
            "model_name": "lightgbm",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-b",
        },
        {
            "model_name": "last_week",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-last",
        },
        {
            "model_name": "lightgbm",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-a",
        },
    ]

    table = build_report_table(rows, table_name="trend_model_metrics")

    assert table["model_name"].tolist() == ["last_week", "lightgbm", "lightgbm"]
    assert table["run_id"].tolist() == ["run-last", "run-b", "run-a"]


def test_write_report_table_rejects_missing_columns(tmp_path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "split": "test",
                "mae": 0.1,
            }
        ]
    )

    with pytest.raises(ValueError, match="报告表格缺少列.*run_id"):
        write_report_table(
            dataframe,
            columns=TREND_MODEL_METRICS_COLUMNS,
            output_paths={
                "csv": tmp_path / "trend_model_metrics.csv",
                "markdown": tmp_path / "trend_model_metrics.md",
            },
        )


def test_write_report_table_writes_csv_and_markdown(tmp_path) -> None:
    columns = ("split", "model_name", "run_id")
    dataframe = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "split": "test",
                "mae": 0.1,
                "rmse": 0.2,
                "spearman": 0.3,
                "ndcg_at_10": 0.4,
                "precision_at_10": 0.5,
                "recall_at_10": 0.6,
                "run_id": "run-1",
            }
        ]
    )
    paths = {
        "csv": tmp_path / "trend_model_metrics.csv",
        "markdown": tmp_path / "trend_model_metrics.md",
    }

    written = write_report_table(
        dataframe,
        columns=columns,
        output_paths=paths,
    )

    assert written == [paths["csv"], paths["markdown"]]
    assert paths["csv"].read_text(encoding="utf-8").splitlines()[0] == (
        '"split","model_name","run_id"'
    )
    assert paths["markdown"].read_text(encoding="utf-8").splitlines()[0] == (
        "| split | model_name | run_id |"
    )
