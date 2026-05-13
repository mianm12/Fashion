from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fashion_trend.presentation import paths
from fashion_trend.presentation.demo_cases import build_demo_case_payloads
from fashion_trend.presentation.source_artifacts import (
    collect_source_artifact_metadata,
)
from fashion_trend.recommendation.contracts import (
    EVALUATION_LABEL_COLUMNS,
    EVALUATION_LABEL_KEY_COLUMNS,
    RECOMMENDATION_ITEMS_COLUMNS,
    RECOMMENDATION_ITEMS_KEY_COLUMNS,
    RECOMMENDATION_TOP_K,
    USER_PROFILE_COLUMNS,
    USER_PROFILE_KEY_COLUMNS,
)
from fashion_trend.recommendation.readers import (
    coerce_article_id_string,
    reject_duplicate_key,
    validate_columns,
    validate_path_value_matches,
)
from fashion_trend.reports.loaders import (
    read_feature_importance,
    read_json_object,
    read_trend_samples,
)


@dataclass(frozen=True)
class PresentationSources:
    manifest: dict[str, Any]
    report_cases: list[dict[str, Any]]
    report_tables: dict[str, pd.DataFrame]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    trend_metrics: dict[str, dict[str, Any]]
    recommendation_metrics: dict[str, dict[str, Any]]
    recommendation_items: pd.DataFrame
    experiment: dict[str, Any]
    evaluation_labels: pd.DataFrame
    user_profile: pd.DataFrame
    attribute_week_heat: pd.DataFrame
    trend_model_samples: pd.DataFrame
    article_nodes: pd.DataFrame
    attribute_nodes: pd.DataFrame
    article_attributes: pd.DataFrame
    attribute_hierarchy_edges: pd.DataFrame
    articles: pd.DataFrame
    source_artifacts: dict[str, dict[str, object]] | None = None


def load_presentation_sources() -> PresentationSources:
    """Load stable artifacts needed to build the read-only defense app tables."""
    recommendation_items = _read_recommendation_items(
        paths.MAIN_RECOMMENDATION_ITEMS_PATH
    )
    evaluation_labels = _read_evaluation_labels(paths.EVALUATION_LABELS_PATH)
    user_profile = _read_user_profile(paths.USER_PROFILE_PATH)
    report_cases = build_demo_case_payloads(
        recommendation_items=recommendation_items,
        evaluation_labels=evaluation_labels,
        user_profile=user_profile,
    )
    return PresentationSources(
        manifest=read_json_object(
            paths.REPORTS_MANIFEST_PATH,
            artifact_name="reports manifest",
        ),
        report_cases=report_cases,
        report_tables=_read_report_tables(paths.REPORTS_TABLES_DIR),
        predictions=_read_predictions(paths.LIGHTGBM_PREDICTIONS_PATH),
        feature_importance=read_feature_importance(
            paths.LIGHTGBM_FEATURE_IMPORTANCE_PATH
        ),
        trend_metrics=_read_trend_metric_payloads(paths.TREND_METRICS_DIR),
        recommendation_metrics=_read_recommendation_metric_payloads(
            paths.RECOMMENDATION_OUTPUT_DIR
        ),
        recommendation_items=filter_frame_to_case_keys(
            recommendation_items,
            report_cases,
            columns=list(RECOMMENDATION_ITEMS_COLUMNS),
            top_k=RECOMMENDATION_TOP_K,
        ),
        experiment=read_json_object(
            paths.RECOMMENDATION_EXPERIMENT_PATH,
            artifact_name="recommendation experiment",
        ),
        evaluation_labels=filter_frame_to_case_keys(
            evaluation_labels,
            report_cases,
            columns=list(EVALUATION_LABEL_COLUMNS),
        ),
        user_profile=filter_frame_to_case_keys(
            user_profile,
            report_cases,
            columns=list(USER_PROFILE_COLUMNS),
        ),
        attribute_week_heat=_read_id_heavy_csv(paths.ATTRIBUTE_WEEK_HEAT_PATH),
        trend_model_samples=read_trend_samples(paths.TREND_MODEL_SAMPLES_PATH),
        article_nodes=_read_id_heavy_csv(paths.GRAPH_NODES_ARTICLE_PATH),
        attribute_nodes=_read_id_heavy_csv(paths.GRAPH_NODES_ATTRIBUTE_PATH),
        article_attributes=_read_id_heavy_csv(paths.GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
        attribute_hierarchy_edges=_read_id_heavy_csv(
            paths.GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH
        ),
        articles=_read_id_heavy_csv(paths.ARTICLES_CLEAN_PATH),
        source_artifacts=_collect_required_source_artifacts(),
    )


def _read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "attr_id": "string",
            "attr_type": "string",
            "attr_value": "string",
            "model_name": "string",
            "split": "string",
        },
        keep_default_na=False,
    )


def _read_id_heavy_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def filter_frame_to_case_keys(
    dataframe: pd.DataFrame,
    report_cases: list[dict[str, Any]],
    *,
    columns: list[str] | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Return rows matching demo case windows, optionally trimming columns/ranks."""
    frame = dataframe.copy()
    if columns is not None:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"case-key filter missing columns: {missing}")
        frame = frame.loc[:, columns].copy()
    if frame.empty or not report_cases:
        return frame.iloc[0:0].copy()

    keys = pd.DataFrame(
        [
            {
                "customer_id": str(case["customer_id"]),
                "split": str(case["split"]),
                "cutoff_week": int(case["cutoff_week"]),
                "label_week": int(case["label_week"]),
            }
            for case in report_cases
        ]
    )
    for column in ("customer_id", "split"):
        frame[column] = frame[column].astype(str)
    for column in ("cutoff_week", "label_week"):
        frame[column] = frame[column].astype(int)
    filtered = frame.merge(
        keys,
        on=["customer_id", "split", "cutoff_week", "label_week"],
        how="inner",
    )
    if top_k is not None and "rank" in filtered.columns:
        filtered["rank"] = filtered["rank"].astype(int)
        filtered = filtered.loc[filtered["rank"] <= top_k].copy()
    return filtered


def _read_recommendation_items(path: Path) -> pd.DataFrame:
    expected_method = path.parent.name
    dataframe = pd.read_parquet(path, columns=list(RECOMMENDATION_ITEMS_COLUMNS))
    validate_columns(dataframe, RECOMMENDATION_ITEMS_COLUMNS, "recommendation_items")
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(
        dataframe,
        RECOMMENDATION_ITEMS_KEY_COLUMNS,
        "recommendation_items",
    )
    invalid_rank = ~dataframe["rank"].between(1, RECOMMENDATION_TOP_K)
    if invalid_rank.any():
        sample = dataframe.loc[invalid_rank, ["customer_id", "article_id", "rank"]]
        raise ValueError(
            "recommendation_items rank 超出 Top-K 范围: "
            f"{sample.head(3).to_dict('records')}"
        )
    if not dataframe.empty:
        validate_path_value_matches(
            dataframe,
            "method",
            expected_method,
            "recommendation_items",
        )
    return dataframe


def _read_evaluation_labels(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path, columns=list(EVALUATION_LABEL_COLUMNS))
    validate_columns(dataframe, EVALUATION_LABEL_COLUMNS, "evaluation_labels")
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(dataframe, EVALUATION_LABEL_KEY_COLUMNS, "evaluation_labels")
    return dataframe


def _read_user_profile(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path, columns=list(USER_PROFILE_COLUMNS))
    validate_columns(dataframe, USER_PROFILE_COLUMNS, "user_profile")
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(dataframe, USER_PROFILE_KEY_COLUMNS, "user_profile")
    return dataframe


def _read_report_tables(table_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        path.stem: pd.read_csv(path, dtype="string", keep_default_na=False)
        for path in _required_matching_files(table_dir, "*.csv", "report tables")
    }


def _read_trend_metric_payloads(metrics_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in _required_matching_files(
        metrics_dir,
        "*/trend_metrics.json",
        "trend metrics",
    ):
        model_name = path.parent.name
        payloads[model_name] = read_json_object(
            path,
            artifact_name=f"trend metrics {model_name}",
        )
    return payloads


def _read_recommendation_metric_payloads(
    recommendation_dir: Path,
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in _required_matching_files(
        recommendation_dir,
        "*/metrics.json",
        "recommendation metrics",
    ):
        method = path.parent.name
        payloads[method] = read_json_object(
            path,
            artifact_name=f"recommendation metrics {method}",
        )
    return payloads


def _collect_required_source_artifacts() -> dict[str, dict[str, object]]:
    source_paths = _required_source_paths()
    return collect_source_artifact_metadata(
        source_paths,
        required=source_paths.keys(),
    )


def _required_source_paths() -> dict[str, Path]:
    source_paths: dict[str, Path] = {
        "reports_manifest": paths.REPORTS_MANIFEST_PATH,
        "lightgbm_predictions": paths.LIGHTGBM_PREDICTIONS_PATH,
        "lightgbm_feature_importance": paths.LIGHTGBM_FEATURE_IMPORTANCE_PATH,
        "main_recommendation_items": paths.MAIN_RECOMMENDATION_ITEMS_PATH,
        "recommendation_experiment": paths.RECOMMENDATION_EXPERIMENT_PATH,
        "evaluation_labels": paths.EVALUATION_LABELS_PATH,
        "user_profile": paths.USER_PROFILE_PATH,
        "attribute_week_heat": paths.ATTRIBUTE_WEEK_HEAT_PATH,
        "trend_model_samples": paths.TREND_MODEL_SAMPLES_PATH,
        "graph_nodes_article": paths.GRAPH_NODES_ARTICLE_PATH,
        "graph_nodes_attribute": paths.GRAPH_NODES_ATTRIBUTE_PATH,
        "graph_edges_article_attribute": paths.GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH,
        "graph_edges_attribute_hierarchy": paths.GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH,
        "articles_clean": paths.ARTICLES_CLEAN_PATH,
    }
    source_paths.update(
        _keyed_paths(
            "report_table",
            _required_matching_files(
                paths.REPORTS_TABLES_DIR,
                "*.csv",
                "report tables",
            ),
        )
    )
    source_paths.update(
        {
            f"trend_metrics_{path.parent.name}": path
            for path in _required_matching_files(
                paths.TREND_METRICS_DIR,
                "*/trend_metrics.json",
                "trend metrics",
            )
        }
    )
    source_paths.update(
        {
            f"recommendation_metrics_{path.parent.name}": path
            for path in _required_matching_files(
                paths.RECOMMENDATION_OUTPUT_DIR,
                "*/metrics.json",
                "recommendation metrics",
            )
        }
    )
    return source_paths


def _required_matching_files(
    directory: Path,
    pattern: str,
    artifact_name: str,
) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"{artifact_name} directory not found: {directory}")
    paths_found = sorted(directory.glob(pattern))
    if not paths_found:
        raise ValueError(
            f"{artifact_name} not found: path={directory}, pattern={pattern}"
        )
    return paths_found


def _keyed_paths(prefix: str, paths_found: list[Path]) -> dict[str, Path]:
    return {f"{prefix}_{path.stem}": path for path in paths_found}
