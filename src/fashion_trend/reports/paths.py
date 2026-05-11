from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fashion_trend.foundation.artifacts import validate_output_parent_dirs
from fashion_trend.foundation.paths import INTERIM_DIR, OUTPUT_DIR, PROCESSED_DIR

# 报告阶段输出根目录。
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

# 报告阶段图表、表格和案例输出位置。
OUTPUT_FIGURES_DIR = OUTPUT_REPORTS_DIR / "figures"
OUTPUT_TABLES_DIR = OUTPUT_REPORTS_DIR / "tables"
OUTPUT_CASE_STUDIES_DIR = OUTPUT_REPORTS_DIR / "case_studies"
OUTPUT_REPORTS_MANIFEST_PATH = OUTPUT_REPORTS_DIR / "manifest.json"


@dataclass(frozen=True)
class ReportInputPaths:
    trend_metrics: dict[str, Path]
    recommendation_metrics: dict[str, Path]
    recommendation_items: dict[str, Path]
    recommendation_items_csv: dict[str, Path]
    lightgbm_predictions: Path
    lightgbm_feature_importance: Path
    trend_model_samples: Path
    trend_split_samples: dict[str, Path]
    recommendation_experiment: Path
    time_windows: Path
    target_users: Path
    evaluation_labels: Path
    user_profile: Path
    article_attributes: Path
    graph_artifacts: dict[str, Path]
    data_artifacts: dict[str, Path]


def reports_output_dir(output_root: Path | None = None) -> Path:
    """返回本次报告导出的输出根目录。"""
    return output_root if output_root is not None else OUTPUT_REPORTS_DIR


def default_report_input_paths() -> ReportInputPaths:
    """Return read-only default inputs for the reports stage.

    These defaults live in reports so the runner does not import upstream path
    modules such as trend.paths, recommendation.paths, or catalog.paths.
    """
    graph_dir = PROCESSED_DIR / "graph"
    trend_dir = PROCESSED_DIR / "trend"
    features_dir = PROCESSED_DIR / "features"
    recommend_dir = PROCESSED_DIR / "recommend"
    output_models_dir = OUTPUT_DIR / "models"
    output_metrics_dir = OUTPUT_DIR / "metrics"
    output_recommendation_dir = OUTPUT_DIR / "recommendation"

    trend_models = ("last_week", "previous_growth", "moving_average", "lightgbm")
    recommendation_methods = (
        "global_popularity",
        "recent_popularity",
        "attribute_similarity",
        "pop_similarity",
        "pop_similarity_trend",
    )
    graph_artifacts = {
        "nodes_article": graph_dir / "nodes_article.csv",
        "nodes_attribute": graph_dir / "nodes_attribute.csv",
        "edges_article_attribute": graph_dir / "edges_article_attribute.csv",
        "edges_attribute_hierarchy": graph_dir / "edges_attribute_hierarchy.csv",
    }
    trend_split_samples = {
        "train": features_dir / "trend_model_samples_train.parquet",
        "valid": features_dir / "trend_model_samples_valid.parquet",
        "test": features_dir / "trend_model_samples_test.parquet",
    }
    data_artifacts = {
        "articles_clean": INTERIM_DIR / "articles_clean.csv",
        **graph_artifacts,
        "article_week_sales": trend_dir / "article_week_sales.csv",
        "attribute_week_heat": trend_dir / "attribute_week_heat.csv",
        "attribute_week_target": trend_dir / "attribute_week_target.csv",
        "trend_model_samples": features_dir / "trend_model_samples.parquet",
        "time_windows": recommend_dir / "time_windows.parquet",
        "target_users": recommend_dir / "target_users.parquet",
        "evaluation_labels": recommend_dir / "evaluation_labels.parquet",
        "user_profile": recommend_dir / "user_profile.parquet",
    }
    return ReportInputPaths(
        trend_metrics={
            model_name: output_metrics_dir / model_name / "trend_metrics.json"
            for model_name in trend_models
        },
        recommendation_metrics={
            method: output_recommendation_dir / method / "metrics.json"
            for method in recommendation_methods
        },
        recommendation_items={
            method: output_recommendation_dir / method / "recommendation_items.parquet"
            for method in recommendation_methods
        },
        recommendation_items_csv={
            method: output_recommendation_dir / method / "recommendation_items.csv"
            for method in recommendation_methods
        },
        lightgbm_predictions=output_models_dir / "lightgbm" / "predictions.csv",
        lightgbm_feature_importance=(
            output_models_dir / "lightgbm" / "feature_importance.csv"
        ),
        trend_model_samples=features_dir / "trend_model_samples.parquet",
        trend_split_samples=trend_split_samples,
        recommendation_experiment=(
            output_recommendation_dir / "experiments" / "main" / "experiment.json"
        ),
        time_windows=recommend_dir / "time_windows.parquet",
        target_users=recommend_dir / "target_users.parquet",
        evaluation_labels=recommend_dir / "evaluation_labels.parquet",
        user_profile=recommend_dir / "user_profile.parquet",
        article_attributes=graph_artifacts["edges_article_attribute"],
        graph_artifacts=graph_artifacts,
        data_artifacts=data_artifacts,
    )


def figure_output_paths(
    name: str,
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """返回同一图表的 SVG 和 PNG 输出路径。"""
    _validate_report_artifact_name(name)
    root = reports_output_dir(output_root)
    return {
        "svg": root / "figures" / f"{name}.svg",
        "png": root / "figures" / f"{name}.png",
    }


def table_output_paths(
    name: str,
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """返回同一表格的 CSV 和 Markdown 输出路径。"""
    _validate_report_artifact_name(name)
    root = reports_output_dir(output_root)
    return {
        "csv": root / "tables" / f"{name}.csv",
        "markdown": root / "tables" / f"{name}.md",
    }


def case_study_output_paths(
    case_id: str,
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """返回单个案例的 JSON 和 Markdown 输出路径。"""
    _validate_report_artifact_name(case_id)
    root = reports_output_dir(output_root)
    return {
        "json": root / "case_studies" / f"{case_id}.json",
        "markdown": root / "case_studies" / f"{case_id}.md",
    }


def manifest_output_path(output_root: Path | None = None) -> Path:
    """返回本次报告 manifest 输出路径。"""
    return reports_output_dir(output_root) / "manifest.json"


def validate_report_output_path(
    path: Path,
    *,
    output_root: Path | None = None,
) -> None:
    """确认报告产物仍写在本次 output root 内。"""
    validate_output_parent_dirs(path.parent, reports_output_dir(output_root))


def _validate_report_artifact_name(name: str) -> None:
    if not name:
        raise ValueError("报告产物名称不能为空。")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"报告产物名称不是安全路径片段: {name}")
