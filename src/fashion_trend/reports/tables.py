from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fashion_trend.foundation.io import write_csv_atomic, write_text_atomic
from fashion_trend.reports.markdown import markdown_table

DATA_ARTIFACT_SUMMARY_COLUMNS = (
    "section",
    "artifact",
    "path",
    "row_count",
    "column_count",
    "paper_usage",
)
TIME_SPLIT_SUMMARY_COLUMNS = (
    "domain",
    "split",
    "week_start",
    "week_end",
    "week_count",
    "row_count",
    "attribute_count",
    "user_count",
)
ATTRIBUTE_GRAPH_SUMMARY_COLUMNS = (
    "entity_type",
    "attr_type",
    "relation_type",
    "count",
    "path",
    "paper_usage",
)
TREND_FEATURE_SUMMARY_COLUMNS = (
    "feature_group",
    "feature_name",
    "source_table",
    "model_input",
    "description",
)
TREND_MODEL_METRICS_COLUMNS = (
    "model_name",
    "split",
    "mae",
    "rmse",
    "spearman",
    "ndcg_at_10",
    "precision_at_10",
    "recall_at_10",
    "run_id",
)
TREND_METRICS_BY_ATTR_TYPE_COLUMNS = (
    "model_name",
    "split",
    "attr_type",
    "mae",
    "rmse",
    "spearman",
    "ndcg_at_10",
    "precision_at_10",
    "recall_at_10",
)
RECOMMENDATION_METHOD_METRICS_COLUMNS = (
    "method",
    "split",
    "map_at_12",
    "recall_at_12",
    "hit_rate_at_12",
    "ndcg_at_12",
    "coverage",
    "user_count",
    "missing_recommendation_user_count",
)
RECOMMENDATION_EXPERIMENT_SUMMARY_COLUMNS = (
    "section",
    "rank",
    "method",
    "split",
    "pop_score",
    "sim_score",
    "trend_score",
    "recent_score",
    "map_at_12",
    "recall_at_12",
    "hit_rate_at_12",
    "ndcg_at_12",
    "coverage",
)

REPORT_TABLE_COLUMNS = {
    "data_artifact_summary": DATA_ARTIFACT_SUMMARY_COLUMNS,
    "time_split_summary": TIME_SPLIT_SUMMARY_COLUMNS,
    "attribute_graph_summary": ATTRIBUTE_GRAPH_SUMMARY_COLUMNS,
    "trend_feature_summary": TREND_FEATURE_SUMMARY_COLUMNS,
    "trend_model_metrics": TREND_MODEL_METRICS_COLUMNS,
    "trend_metrics_by_attr_type": TREND_METRICS_BY_ATTR_TYPE_COLUMNS,
    "recommendation_method_metrics": RECOMMENDATION_METHOD_METRICS_COLUMNS,
    "recommendation_experiment_summary": RECOMMENDATION_EXPERIMENT_SUMMARY_COLUMNS,
}
REPORT_TABLE_SORT_COLUMNS = {
    "data_artifact_summary": ("section", "artifact"),
    "time_split_summary": ("domain", "split", "week_start"),
    "attribute_graph_summary": ("entity_type", "attr_type", "relation_type"),
    "trend_feature_summary": ("feature_group", "feature_name"),
    "trend_model_metrics": ("model_name", "split"),
    "trend_metrics_by_attr_type": ("model_name", "split", "attr_type"),
    "recommendation_method_metrics": ("method", "split"),
    "recommendation_experiment_summary": ("section", "rank", "split", "method"),
}


def build_report_table(rows: list[dict[str, Any]], *, table_name: str) -> pd.DataFrame:
    if table_name not in REPORT_TABLE_COLUMNS:
        raise ValueError(f"未知报告表格: {table_name}")
    if not rows:
        raise ValueError(f"{table_name} 表格无数据")
    dataframe = pd.DataFrame(rows)
    return _select_and_sort(
        dataframe,
        columns=REPORT_TABLE_COLUMNS[table_name],
        sort_columns=REPORT_TABLE_SORT_COLUMNS[table_name],
        table_name=table_name,
    )


def build_trend_model_metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return build_report_table(rows, table_name="trend_model_metrics")


def build_recommendation_method_metrics_table(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    return build_report_table(rows, table_name="recommendation_method_metrics")


def write_report_table(
    dataframe: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    output_paths: dict[str, Path],
) -> list[Path]:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"报告表格缺少列: {missing}")
    table = dataframe.loc[:, list(columns)]
    csv_path = output_paths["csv"]
    markdown_path = output_paths["markdown"]
    write_csv_atomic(table, csv_path)
    write_text_atomic(markdown_table(table, columns=columns), markdown_path)
    _validate_non_empty_file(csv_path)
    _validate_non_empty_file(markdown_path)
    return [csv_path, markdown_path]


def _select_and_sort(
    dataframe: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    sort_columns: tuple[str, ...],
    table_name: str,
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{table_name} 缺少列: {missing}")
    return (
        dataframe.loc[:, list(columns)]
        .sort_values(list(sort_columns), kind="mergesort")
        .reset_index(drop=True)
    )


def _validate_non_empty_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"报告表格输出为空: {path}")
