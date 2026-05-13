from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fashion_trend.catalog.readers import (
    read_article_attribute_edges,
    read_attribute_hierarchy_edges,
    read_attribute_nodes,
)
from fashion_trend.foundation.io import write_json_atomic, write_text_atomic
from fashion_trend.recommendation.readers import (
    read_evaluation_labels,
    read_recommendation_items,
    read_target_users,
    read_time_windows,
    read_user_profile,
)
from fashion_trend.reports.cases import (
    build_case_payload,
    render_case_markdown,
    select_recommendation_cases,
)
from fashion_trend.reports.figures import (
    build_attribute_graph_schema_figure,
    build_data_pipeline_figure,
    build_feature_importance_figure,
    build_recommendation_method_metrics_figure,
    build_recommendation_weight_analysis_figure,
    build_topk_trend_attributes_figure,
    build_trend_curve_examples_figure,
    build_trend_model_metrics_figure,
)
from fashion_trend.reports.loaders import (
    build_lightgbm_prediction_sample_view,
    flatten_recommendation_metrics,
    flatten_trend_metrics,
    flatten_trend_metrics_by_attr_type,
    read_feature_importance,
    read_json_object,
    read_trend_samples,
)
from fashion_trend.reports.manifest import build_manifest_payload, write_manifest
from fashion_trend.reports.paths import (
    ReportInputPaths,
    case_study_output_paths,
    default_report_input_paths,
    figure_output_paths,
    manifest_output_path,
    table_output_paths,
)
from fashion_trend.reports.plotting import (
    configure_matplotlib_for_reports,
    save_report_figure,
)
from fashion_trend.reports.tables import (
    REPORT_TABLE_COLUMNS,
    build_report_table,
    write_report_table,
)
from fashion_trend.trend.readers import read_trend_metrics, read_trend_model_predictions


@dataclass(frozen=True)
class PaperAssetsExportConfig:
    case_count: int = 3
    top_k: int = 10
    trend_week: int = 103
    figure_formats: tuple[str, ...] = ("svg", "png")
    output_dir: Path | None = None
    input_paths: ReportInputPaths | None = None


@dataclass(frozen=True)
class ReportInputs:
    input_artifacts: dict[str, str]
    report_table_rows: dict[str, list[dict[str, object]]]
    trend_metrics: pd.DataFrame
    recommendation_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    trend_view: pd.DataFrame
    search_results: pd.DataFrame
    recommendation_items: pd.DataFrame
    evaluation_labels: pd.DataFrame
    user_profile: pd.DataFrame
    article_attributes: pd.DataFrame
    representative_trends: pd.DataFrame
    best_weights: dict[str, float]
    warnings: list[str]


def run_paper_assets_export(config: PaperAssetsExportConfig) -> dict[str, Any]:
    """Export tables, figures, cases, and manifest from stable artifacts."""
    selected_font = configure_matplotlib_for_reports()
    inputs = _load_report_inputs(config)
    table_paths, row_counts = _write_tables(
        inputs.report_table_rows,
        output_root=config.output_dir,
    )
    figure_paths = _write_figures(
        inputs.trend_metrics,
        inputs.recommendation_metrics,
        inputs.feature_importance,
        inputs.trend_view,
        inputs.search_results,
        best_weights=inputs.best_weights,
        trend_week=config.trend_week,
        top_k=config.top_k,
        figure_formats=config.figure_formats,
        output_root=config.output_dir,
    )
    case_paths, case_user_ids = _write_cases(
        inputs.recommendation_items,
        inputs.evaluation_labels,
        inputs.user_profile,
        inputs.article_attributes,
        inputs.representative_trends,
        case_count=config.case_count,
        output_root=config.output_dir,
    )
    manifest_path = manifest_output_path(config.output_dir)
    payload = build_manifest_payload(
        parameters={
            "case_count": config.case_count,
            "top_k": config.top_k,
            "trend_week": config.trend_week,
            "figure_formats": list(config.figure_formats),
            "output_dir": str(manifest_path.parent),
            "selected_font": selected_font,
        },
        input_artifacts=inputs.input_artifacts,
        output_artifacts={
            "figures": figure_paths,
            "tables": table_paths,
            "case_studies": case_paths,
        },
        row_counts=row_counts,
        case_user_ids=case_user_ids,
        warnings=inputs.warnings,
    )
    payload["manifest_path"] = str(manifest_path)
    write_manifest(payload, manifest_path)
    return payload


def _load_report_inputs(config: PaperAssetsExportConfig) -> ReportInputs:
    input_paths = config.input_paths or default_report_input_paths()
    trend_metric_payloads = [
        read_trend_metrics(path) for path in input_paths.trend_metrics.values()
    ]
    trend_metric_rows, trend_attr_type_rows = _build_trend_metric_rows(
        trend_metric_payloads
    )
    recommendation_metric_rows = []
    for method, path in input_paths.recommendation_metrics.items():
        payload = read_json_object(path, artifact_name=f"{method} metrics")
        recommendation_metric_rows.extend(flatten_recommendation_metrics(payload))

    predictions = read_trend_model_predictions(input_paths.lightgbm_predictions)
    trend_samples = read_trend_samples(input_paths.trend_model_samples)
    trend_view = build_lightgbm_prediction_sample_view(predictions, trend_samples)
    feature_importance = read_feature_importance(
        input_paths.lightgbm_feature_importance
    )
    experiment_payload = read_json_object(
        input_paths.recommendation_experiment,
        artifact_name="main recommendation experiment",
    )
    best_weights = _extract_best_weights(experiment_payload)
    recommendation_items = read_recommendation_items(
        input_paths.recommendation_items["pop_similarity_trend"]
    )
    evaluation_labels = read_evaluation_labels(input_paths.evaluation_labels)
    user_profile = read_user_profile(input_paths.user_profile)
    article_attributes = read_article_attribute_edges(input_paths.article_attributes)
    case_cutoff_weeks = sorted(
        recommendation_items.loc[
            recommendation_items["split"].astype(str) == "test",
            "cutoff_week",
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    representative_trends = build_representative_trend_attributes(
        trend_view,
        week_ids=case_cutoff_weeks,
    )
    search_results = flatten_experiment_search_results(experiment_payload)
    report_table_rows = {
        "data_artifact_summary": build_data_artifact_summary_rows(
            input_paths.data_artifacts
        ),
        "time_split_summary": build_time_split_summary_rows(
            split_paths=input_paths.trend_split_samples,
            time_windows_path=input_paths.time_windows,
            target_users_path=input_paths.target_users,
        ),
        "attribute_graph_summary": build_attribute_graph_summary_rows(
            graph_paths=input_paths.graph_artifacts
        ),
        "trend_feature_summary": build_trend_feature_summary_rows(),
        "trend_model_metrics": trend_metric_rows,
        "trend_metrics_by_attr_type": trend_attr_type_rows,
        "recommendation_method_metrics": recommendation_metric_rows,
        "recommendation_experiment_summary": flatten_recommendation_experiment_rows(
            experiment_payload
        ),
    }
    return ReportInputs(
        input_artifacts=_build_input_artifacts(input_paths),
        report_table_rows=report_table_rows,
        trend_metrics=build_report_table(
            trend_metric_rows,
            table_name="trend_model_metrics",
        ),
        recommendation_metrics=build_report_table(
            recommendation_metric_rows,
            table_name="recommendation_method_metrics",
        ),
        feature_importance=feature_importance,
        trend_view=trend_view,
        search_results=search_results,
        recommendation_items=recommendation_items,
        evaluation_labels=evaluation_labels,
        user_profile=user_profile,
        article_attributes=article_attributes,
        representative_trends=representative_trends,
        best_weights=best_weights,
        warnings=_build_report_warnings(experiment_payload, input_paths),
    )


def build_data_artifact_summary_rows(
    artifacts: dict[str, Path | str] | None = None,
    *,
    sections: dict[str, str] | None = None,
    paper_usage: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    artifacts = artifacts or default_report_input_paths().data_artifacts
    section_map = {
        **_default_data_artifact_sections(),
        **(sections or {}),
    }
    usage_map = {
        **_default_data_artifact_usage(),
        **(paper_usage or {}),
    }
    rows: list[dict[str, object]] = []
    for artifact, path_value in artifacts.items():
        path = Path(path_value)
        row_count, column_count = _artifact_shape(path)
        rows.append(
            {
                "section": section_map.get(artifact, "other"),
                "artifact": artifact,
                "path": str(path),
                "row_count": row_count,
                "column_count": column_count,
                "paper_usage": usage_map.get(artifact, ""),
            }
        )
    return rows


def build_time_split_summary_rows(
    split_frames: dict[str, pd.DataFrame] | None = None,
    *,
    split_paths: dict[str, Path] | None = None,
    time_windows_path: Path | None = None,
    target_users_path: Path | None = None,
    domain: str = "trend",
) -> list[dict[str, object]]:
    injected_split_frames = split_frames is not None
    input_paths = default_report_input_paths()
    split_paths = split_paths or input_paths.trend_split_samples
    split_frames = split_frames or {
        split: pd.read_parquet(path) for split, path in split_paths.items()
    }
    rows = [
        _time_split_row(domain=domain, split=split, dataframe=dataframe)
        for split, dataframe in split_frames.items()
    ]
    if injected_split_frames:
        return rows

    time_windows = read_time_windows(time_windows_path or input_paths.time_windows)
    target_users = read_target_users(target_users_path or input_paths.target_users)
    for split, dataframe in time_windows.groupby("split", sort=True):
        split_users = target_users.loc[target_users["split"].astype(str) == str(split)]
        rows.append(
            _time_split_row(
                domain="recommendation",
                split=str(split),
                dataframe=dataframe,
                user_count=split_users["customer_id"].nunique(),
            )
        )
    return rows


def build_attribute_graph_summary_rows(
    graph_frames: dict[str, pd.DataFrame] | None = None,
    *,
    graph_paths: dict[str, Path | str] | None = None,
) -> list[dict[str, object]]:
    default_graph_paths = default_report_input_paths().graph_artifacts
    graph_paths = {**default_graph_paths, **(graph_paths or {})}
    graph_frames = graph_frames or {
        "nodes_article": pd.read_csv(graph_paths["nodes_article"]),
        "nodes_attribute": read_attribute_nodes(Path(graph_paths["nodes_attribute"])),
        "edges_article_attribute": read_article_attribute_edges(
            Path(graph_paths["edges_article_attribute"])
        ),
        "edges_attribute_hierarchy": read_attribute_hierarchy_edges(
            Path(graph_paths["edges_attribute_hierarchy"])
        ),
    }
    rows = [
        {
            "entity_type": "article",
            "attr_type": "",
            "relation_type": "node",
            "count": len(graph_frames["nodes_article"]),
            "path": str(graph_paths["nodes_article"]),
            "paper_usage": "商品节点规模",
        },
        {
            "entity_type": "attribute",
            "attr_type": "",
            "relation_type": "node",
            "count": len(graph_frames["nodes_attribute"]),
            "path": str(graph_paths["nodes_attribute"]),
            "paper_usage": "属性节点规模",
        },
        {
            "entity_type": "attribute",
            "attr_type": "",
            "relation_type": "hierarchy",
            "count": len(graph_frames["edges_attribute_hierarchy"]),
            "path": str(graph_paths["edges_attribute_hierarchy"]),
            "paper_usage": "属性层级边规模",
        },
    ]
    edge_frame = graph_frames["edges_article_attribute"]
    for attr_type, group in edge_frame.groupby("attr_type", sort=True):
        rows.append(
            {
                "entity_type": "article_attribute_edge",
                "attr_type": str(attr_type),
                "relation_type": "article_attribute",
                "count": len(group),
                "path": str(graph_paths["edges_article_attribute"]),
                "paper_usage": "商品-属性边规模",
            }
        )
    return rows


def build_trend_feature_summary_rows() -> list[dict[str, object]]:
    feature_specs = [
        ("level", "heat_t", "trend_model_samples", True, "当前周属性热度"),
        ("level", "share_t", "trend_model_samples", True, "当前周属性份额"),
        ("lag", "lag_1_heat", "trend_model_samples", True, "上一周属性热度"),
        ("lag", "lag_2_heat", "trend_model_samples", True, "前两周属性热度"),
        ("lag", "lag_4_heat", "trend_model_samples", True, "前四周属性热度"),
        ("growth", "growth_1w", "trend_model_samples", True, "一周热度变化"),
        ("growth", "growth_4w", "trend_model_samples", True, "四周热度变化"),
        (
            "history",
            "history_total_heat_t",
            "trend_model_samples",
            True,
            "历史累计热度",
        ),
        ("history", "history_active_weeks_t", "trend_model_samples", True, "活跃周数"),
        ("label", "target_growth", "trend_model_samples", False, "下一周趋势增长"),
    ]
    return [
        {
            "feature_group": feature_group,
            "feature_name": feature_name,
            "source_table": source_table,
            "model_input": model_input,
            "description": description,
        }
        for feature_group, feature_name, source_table, model_input, description in feature_specs
    ]


def build_representative_trend_attributes(
    trend_view: pd.DataFrame,
    *,
    week_id: int | None = None,
    week_ids: list[int] | None = None,
    top_n: int = 8,
) -> pd.DataFrame:
    required_columns = {
        "week_id",
        "attr_type",
        "attr_value",
        "pred_target_growth",
        "heat_t",
        "is_trend_eligible_t",
    }
    missing_columns = sorted(required_columns - set(trend_view.columns))
    if missing_columns:
        raise ValueError(f"代表趋势属性缺少字段: {missing_columns}")
    if week_id is not None and week_ids is not None:
        raise ValueError("week_id 和 week_ids 不能同时传入")
    selected_weeks: list[int] | None = None
    if week_id is not None:
        selected_weeks = [int(week_id)]
    if week_ids is not None:
        selected_weeks = sorted({int(value) for value in week_ids})

    filtered = trend_view.loc[trend_view["is_trend_eligible_t"].astype(bool)].copy()
    filtered["week_id"] = filtered["week_id"].astype(int)
    if selected_weeks is not None:
        filtered = filtered.loc[filtered["week_id"].isin(selected_weeks)].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "week_id",
                "attr_type",
                "attr_value",
                "pred_target_growth",
                "heat_t",
            ]
        )
    filtered = filtered.sort_values(
        ["week_id", "pred_target_growth", "heat_t"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    per_week = filtered.groupby("week_id", group_keys=False, sort=False).head(top_n)
    return per_week.sort_values(
        ["pred_target_growth", "heat_t"],
        ascending=[False, False],
        kind="mergesort",
    ).loc[:, ["week_id", "attr_type", "attr_value", "pred_target_growth", "heat_t"]]


def flatten_experiment_search_results(payload: dict[str, object]) -> pd.DataFrame:
    rows = []
    for rank, result in enumerate(payload.get("search_results", []), start=1):
        if not isinstance(result, dict):
            raise ValueError(f"search_results[{rank - 1}] 必须是对象")
        weights = _required_mapping(result, "weights", f"search_results[{rank - 1}]")
        metrics = _required_mapping(
            result,
            "valid_metrics",
            f"search_results[{rank - 1}]",
        )
        rows.append(
            _recommendation_experiment_row(
                section="search_results",
                rank=rank,
                method=str(result.get("method", "pop_similarity_trend")),
                split="valid",
                weights=weights,
                metrics=metrics,
            )
        )
    return pd.DataFrame(
        rows,
        columns=REPORT_TABLE_COLUMNS["recommendation_experiment_summary"],
    )


def flatten_recommendation_experiment_rows(
    payload: dict[str, object],
) -> list[dict[str, object]]:
    rows = flatten_experiment_search_results(payload).to_dict("records")
    best_weights = payload.get("best_weights", {})
    if not isinstance(best_weights, dict):
        raise ValueError("experiment.json best_weights 必须是对象")
    for rank, result in enumerate(payload.get("ablation", []), start=1):
        if not isinstance(result, dict):
            raise ValueError(f"ablation[{rank - 1}] 必须是对象")
        method = str(result.get("method", ""))
        weights: dict[str, object] = {}
        if method == "pop_similarity_trend":
            weights = best_weights
        rows.append(
            _recommendation_experiment_row(
                section="ablation",
                rank=rank,
                method=method,
                split=str(result.get("split", "test")),
                weights=weights,
                metrics=result,
                blank_missing_weights=method != "pop_similarity_trend",
            )
        )
    rows.extend(
        _flatten_named_experiment_rows(
            payload.get("named_ablation", []),
            section="named_ablation",
        )
    )
    rows.extend(
        _flatten_named_experiment_rows(
            payload.get("trend_bucket_best_by_valid", []),
            section="trend_bucket_best_by_valid",
        )
    )
    return rows


def _flatten_named_experiment_rows(
    values: object,
    *,
    section: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if values is None:
        return rows
    if not isinstance(values, list):
        raise ValueError(f"{section} 必须是列表")
    for rank, result in enumerate(values, start=1):
        if not isinstance(result, dict):
            raise ValueError(f"{section}[{rank - 1}] 必须是对象")
        weights = _required_mapping(result, "weights", f"{section}[{rank - 1}]")
        metrics_by_split = _required_mapping(
            result,
            "metrics",
            f"{section}[{rank - 1}]",
        )
        display_name = str(result.get("display_name") or result.get("variant_id") or "")
        blank_missing_weights = (
            str(result.get("weight_policy", "")) == "stable_method_baseline"
        )
        for split in ("valid", "test"):
            metrics = _required_mapping(
                metrics_by_split,
                split,
                f"{section}[{rank - 1}].metrics",
            )
            rows.append(
                _recommendation_experiment_row(
                    section=section,
                    rank=rank,
                    method=display_name,
                    split=split,
                    weights=weights,
                    metrics=metrics,
                    blank_missing_weights=blank_missing_weights,
                )
            )
    return rows


def _write_tables(
    report_table_rows: dict[str, list[dict[str, object]]],
    *,
    output_root: Path | None,
) -> tuple[list[str], dict[str, int]]:
    missing_tables = sorted(set(REPORT_TABLE_COLUMNS) - set(report_table_rows))
    if missing_tables:
        raise ValueError(f"报告表格缺少设计要求的表: {missing_tables}")
    output_paths: list[str] = []
    row_counts: dict[str, int] = {}
    for name, columns in REPORT_TABLE_COLUMNS.items():
        table = build_report_table(report_table_rows[name], table_name=name)
        written = write_report_table(
            table,
            columns=columns,
            output_paths=table_output_paths(name, output_root=output_root),
        )
        output_paths.extend(str(path) for path in written)
        row_counts[name] = len(table)
    return output_paths, row_counts


def _write_figures(
    trend_metrics: pd.DataFrame,
    recommendation_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    trend_view: pd.DataFrame,
    search_results: pd.DataFrame,
    *,
    best_weights: dict[str, float],
    trend_week: int,
    top_k: int,
    figure_formats: tuple[str, ...],
    output_root: Path | None,
) -> list[str]:
    builders = [
        ("data_pipeline", build_data_pipeline_figure),
        ("attribute_graph_schema", build_attribute_graph_schema_figure),
        (
            "trend_curve_examples",
            lambda: build_trend_curve_examples_figure(
                trend_view,
                week_id=trend_week,
                lookback_weeks=8,
                top_n=3,
            ),
        ),
        (
            "lightgbm_feature_importance",
            lambda: build_feature_importance_figure(feature_importance, top_n=15),
        ),
        (
            "trend_model_metrics",
            lambda: build_trend_model_metrics_figure(trend_metrics),
        ),
        (
            "recommendation_method_metrics",
            lambda: build_recommendation_method_metrics_figure(recommendation_metrics),
        ),
        (
            "topk_trend_attributes",
            lambda: build_topk_trend_attributes_figure(
                trend_view,
                week_id=trend_week,
                top_k=top_k,
            ),
        ),
        (
            "recommendation_weight_analysis",
            lambda: build_recommendation_weight_analysis_figure(
                search_results,
                best_weights=best_weights,
            ),
        ),
    ]
    output_paths: list[str] = []
    for name, build_figure in builders:
        written = save_report_figure(
            build_figure(),
            figure_output_paths(name, output_root=output_root),
            formats=figure_formats,
        )
        output_paths.extend(str(path) for path in written)
    return output_paths


def _write_cases(
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    user_profile: pd.DataFrame,
    article_attributes: pd.DataFrame,
    representative_trends: pd.DataFrame,
    *,
    case_count: int,
    output_root: Path | None,
) -> tuple[list[str], list[str]]:
    case_keys = select_recommendation_cases(
        recommendation_items=recommendation_items,
        evaluation_labels=evaluation_labels,
        user_profile=user_profile,
        case_count=case_count,
    )
    output_paths: list[str] = []
    case_user_ids: list[str] = []
    for index, case_key in enumerate(case_keys, start=1):
        payload = build_case_payload(
            case_key=case_key,
            recommendation_items=recommendation_items,
            evaluation_labels=evaluation_labels,
            user_profile=user_profile,
            article_attributes=article_attributes,
            representative_trends=representative_trends,
        )
        paths = case_study_output_paths(f"case_{index:02d}", output_root=output_root)
        write_json_atomic(payload, paths["json"])
        write_text_atomic(render_case_markdown(payload), paths["markdown"])
        output_paths.extend(str(path) for path in paths.values())
        case_user_ids.append(str(payload["customer_id"]))
    return output_paths, case_user_ids


def _build_input_artifacts(input_paths: ReportInputPaths) -> dict[str, str]:
    artifacts = {
        "lightgbm_predictions": str(input_paths.lightgbm_predictions),
        "trend_model_samples": str(input_paths.trend_model_samples),
        "feature_importance": str(input_paths.lightgbm_feature_importance),
        "recommendation_experiment": str(input_paths.recommendation_experiment),
        "evaluation_labels": str(input_paths.evaluation_labels),
        "user_profile": str(input_paths.user_profile),
        "article_attributes": str(input_paths.article_attributes),
    }
    _extend_prefixed_paths(artifacts, "data_artifact", input_paths.data_artifacts)
    _extend_prefixed_paths(
        artifacts, "trend_split_sample", input_paths.trend_split_samples
    )
    _extend_prefixed_paths(artifacts, "graph_artifact", input_paths.graph_artifacts)
    _extend_prefixed_paths(artifacts, "trend_metrics", input_paths.trend_metrics)
    _extend_prefixed_paths(
        artifacts,
        "recommendation_metrics",
        input_paths.recommendation_metrics,
    )
    _extend_prefixed_paths(
        artifacts,
        "recommendation_items",
        input_paths.recommendation_items,
    )
    _extend_prefixed_paths(
        artifacts,
        "recommendation_items_csv",
        input_paths.recommendation_items_csv,
    )
    return artifacts


def _build_report_warnings(
    experiment_payload: dict[str, object],
    input_paths: ReportInputPaths,
) -> list[str]:
    warnings: list[str] = []
    search_results = experiment_payload.get("search_results", [])
    if isinstance(search_results, list) and any(
        isinstance(row, dict) and "valid_metrics" in row and "test_metrics" not in row
        for row in search_results
    ):
        warnings.append(
            "recommendation grid search 只有 valid 指标，test 指标仅来自最终方法评价。"
        )
    if not _has_strict_without_recent_ablation(experiment_payload):
        warnings.append("recommendation ablation 缺少严格 w/o Recent 消融行。")

    legacy_csv_paths = [
        path
        for path in input_paths.recommendation_items_csv.values()
        if Path(path).exists()
    ]
    if legacy_csv_paths:
        warnings.append(
            "检测到历史 recommendation_items.csv，但报告读取 parquet 长表: "
            + ", ".join(str(path) for path in legacy_csv_paths)
        )
    return warnings


def _build_trend_metric_rows(
    trend_metric_payloads: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    attr_type_rows: list[dict[str, object]] = []
    for payload in trend_metric_payloads:
        metric_rows.extend(flatten_trend_metrics(payload))
        attr_type_rows.extend(flatten_trend_metrics_by_attr_type(payload))
    return metric_rows, attr_type_rows


def _artifact_shape(path: Path) -> tuple[int, int]:
    if path.suffix == ".csv":
        dataframe = pd.read_csv(path)
        return len(dataframe), len(dataframe.columns)
    if path.suffix == ".parquet":
        dataframe = pd.read_parquet(path)
        return len(dataframe), len(dataframe.columns)
    if path.suffix == ".json":
        payload = read_json_object(path, artifact_name=path.name)
        return 1, len(payload)
    raise ValueError(f"不支持统计的报告 artifact 类型: {path}")


def _time_split_row(
    *,
    domain: str,
    split: str,
    dataframe: pd.DataFrame,
    user_count: int = 0,
) -> dict[str, object]:
    if "week_id" in dataframe.columns:
        week_ids = dataframe["week_id"].dropna().astype(int)
    elif {"cutoff_week", "label_week"} <= set(dataframe.columns):
        week_ids = (
            pd.concat(
                [dataframe["cutoff_week"], dataframe["label_week"]],
                ignore_index=True,
            )
            .dropna()
            .astype(int)
        )
    else:
        raise ValueError("time split summary 缺少 week_id 或 cutoff_week/label_week")
    return {
        "domain": domain,
        "split": split,
        "week_start": int(week_ids.min()),
        "week_end": int(week_ids.max()),
        "week_count": int(week_ids.nunique()),
        "row_count": len(dataframe),
        "attribute_count": (
            int(dataframe["attr_id"].nunique()) if "attr_id" in dataframe.columns else 0
        ),
        "user_count": int(user_count),
    }


def _recommendation_experiment_row(
    *,
    section: str,
    rank: int,
    method: str,
    split: str,
    weights: dict[str, object],
    metrics: dict[str, object],
    blank_missing_weights: bool = False,
) -> dict[str, object]:
    return {
        "section": section,
        "rank": rank,
        "method": method,
        "split": split,
        "pop_score": _score_value(weights, "pop_score", blank=blank_missing_weights),
        "sim_score": _score_value(weights, "sim_score", blank=blank_missing_weights),
        "trend_score": _score_value(
            weights,
            "trend_score",
            blank=blank_missing_weights,
        ),
        "recent_score": _score_value(
            weights,
            "recent_score",
            blank=blank_missing_weights,
        ),
        "map_at_12": float(metrics["map_at_12"]),
        "recall_at_12": float(metrics["recall_at_12"]),
        "hit_rate_at_12": float(metrics["hit_rate_at_12"]),
        "ndcg_at_12": float(metrics["ndcg_at_12"]),
        "coverage": float(metrics["coverage"]),
    }


def _score_value(weights: dict[str, object], key: str, *, blank: bool) -> float | str:
    if key not in weights:
        if blank:
            return ""
        raise ValueError(f"experiment.json 权重缺少字段: {key}")
    return float(weights[key])


def _required_mapping(
    payload: dict[str, object],
    key: str,
    source_name: str,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source_name} {key} 必须是对象")
    return value


def _extract_best_weights(payload: dict[str, object]) -> dict[str, float]:
    weights = payload.get("best_weights")
    if not isinstance(weights, dict):
        raise ValueError("experiment.json best_weights 必须是对象")
    return {
        name: float(weights[name])
        for name in ("pop_score", "sim_score", "trend_score", "recent_score")
    }


def _has_strict_without_recent_ablation(payload: dict[str, object]) -> bool:
    for row in payload.get("named_ablation", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("display_name", "")).lower() != "w/o recent":
            continue
        if row.get("weight_policy") != "strict_drop_and_renormalize_from_full":
            continue
        weights = row.get("weights")
        metrics = row.get("metrics")
        if not isinstance(weights, dict) or not isinstance(metrics, dict):
            continue
        try:
            recent_score = float(weights.get("recent_score", -1.0))
        except (TypeError, ValueError):
            continue
        if recent_score == 0.0 and {"valid", "test"} <= set(metrics):
            return True
    return False


def _extend_prefixed_paths(
    target: dict[str, str],
    prefix: str,
    paths: dict[str, Path],
) -> None:
    for name, path in paths.items():
        target[f"{prefix}__{name}"] = str(path)


def _default_data_artifact_sections() -> dict[str, str]:
    return {
        "articles_clean": "catalog",
        "nodes_article": "attribute_graph",
        "nodes_attribute": "attribute_graph",
        "edges_article_attribute": "attribute_graph",
        "edges_attribute_hierarchy": "attribute_graph",
        "article_week_sales": "trend",
        "attribute_week_heat": "trend",
        "attribute_week_target": "trend",
        "trend_model_samples": "trend",
        "time_windows": "recommendation",
        "target_users": "recommendation",
        "evaluation_labels": "recommendation",
        "user_profile": "recommendation",
    }


def _default_data_artifact_usage() -> dict[str, str]:
    return {
        "articles_clean": "商品属性清洗规模",
        "nodes_article": "属性图商品节点规模",
        "nodes_attribute": "属性图属性节点规模",
        "edges_article_attribute": "商品-属性边规模",
        "edges_attribute_hierarchy": "属性层级边规模",
        "article_week_sales": "商品周销量聚合规模",
        "attribute_week_heat": "属性周热度规模",
        "attribute_week_target": "趋势标签规模",
        "trend_model_samples": "趋势模型样本规模",
        "time_windows": "推荐评价时间窗规模",
        "target_users": "推荐目标用户规模",
        "evaluation_labels": "推荐真实标签规模",
        "user_profile": "用户画像规模",
    }
