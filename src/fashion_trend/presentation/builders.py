from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from fashion_trend.presentation.contracts import CORE_TREND_ATTR_TYPES
from fashion_trend.presentation.extractors import PresentationSources
from fashion_trend.reports.loaders import (
    build_lightgbm_prediction_sample_view,
    flatten_recommendation_metrics,
    flatten_trend_metrics,
)

CASE_KEY_COLUMNS = ("customer_id", "split", "cutoff_week", "label_week")
TOP_K = 12


def build_case_id(
    customer_id: str,
    split: str,
    cutoff_week: int,
    label_week: int,
) -> str:
    return f"demo_{split}_{int(cutoff_week)}_{int(label_week)}_{str(customer_id)[:12]}"


def build_demo_users(report_cases: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in report_cases:
        case_key = _case_key(payload)
        hit_count = int(payload.get("hit_count", 0))
        rows.append(
            {
                "case_id": build_case_id(*case_key),
                "customer_id": case_key[0],
                "split": case_key[1],
                "cutoff_week": case_key[2],
                "label_week": case_key[3],
                "hit_count": hit_count,
                "primary_tags": json.dumps(
                    _primary_tags(payload, hit_count),
                    ensure_ascii=False,
                ),
                "profile_summary": _profile_summary(payload),
                "recommendation_summary": _recommendation_summary(payload, hit_count),
            }
        )
    return pd.DataFrame(rows)


def build_user_profile_attributes(
    report_cases: Sequence[Mapping[str, object]],
    user_profile: pd.DataFrame,
) -> pd.DataFrame:
    frame = _filter_to_cases(user_profile, report_cases).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "customer_id",
                "attr_id",
                "attr_type",
                "attr_value",
                "preference_score",
                "purchase_count",
                "last_purchase_week",
            ]
        )
    _add_case_id_column(frame)
    for column in ("customer_id", "attr_id", "attr_type", "attr_value"):
        frame[column] = frame[column].astype(str)
    return frame.loc[
        :,
        [
            "case_id",
            "customer_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "preference_score",
            "purchase_count",
            "last_purchase_week",
        ],
    ].sort_values(["case_id", "preference_score"], ascending=[True, False])


def build_recommendation_items(
    report_cases: Sequence[Mapping[str, object]],
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
) -> pd.DataFrame:
    items = _top12_items_for_cases(report_cases, recommendation_items)
    return _build_recommendation_items_from_top12(items, evaluation_labels)


def _build_recommendation_items_from_top12(
    items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
) -> pd.DataFrame:
    labels = _label_keys(evaluation_labels)
    items["is_hit"] = [
        1 if key in labels else 0 for key in _iter_item_label_keys(items)
    ]
    return items.loc[
        :,
        [
            "case_id",
            "customer_id",
            "article_id",
            "rank",
            "score",
            "is_hit",
            "candidate_sources",
        ],
    ].sort_values(["case_id", "rank"], kind="mergesort")


def build_recommendation_score_components(
    report_cases: Sequence[Mapping[str, object]],
    recommendation_items: pd.DataFrame,
) -> pd.DataFrame:
    items = _top12_items_for_cases(report_cases, recommendation_items)
    return _build_recommendation_score_components_from_top12(items)


def _build_recommendation_score_components_from_top12(
    items: pd.DataFrame,
) -> pd.DataFrame:
    result = items.loc[
        :,
        [
            "case_id",
            "article_id",
            "pop_score",
            "sim_score",
            "trend_score",
            "recent_score",
            "score",
        ],
    ].rename(columns={"score": "final_score"})
    for column in (
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
        "final_score",
    ):
        result[column] = _finite_float_series(result[column], column)
    return result.sort_values(["case_id", "article_id"], kind="mergesort")


def build_articles(articles_clean: pd.DataFrame) -> pd.DataFrame:
    _require_columns(articles_clean, "articles_clean", ["article_id"])
    columns = [
        "article_id",
        "prod_name",
        "product_group_name",
        "product_type_name",
        "garment_group_name",
        "colour_group_name",
        "graphical_appearance_name",
        "department_name",
        "section_name",
        "index_name",
        "index_group_name",
    ]
    frame = _select_with_defaults(articles_clean, columns)
    frame["article_id"] = frame["article_id"].astype(str)
    return frame.drop_duplicates("article_id")


def build_article_attributes(article_attributes: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        article_attributes,
        "article_attributes",
        ["article_id", "attr_id", "attr_type", "attr_value"],
    )
    frame = _select_with_defaults(
        article_attributes,
        ["article_id", "attr_id", "attr_type", "attr_value"],
    )
    for column in ("article_id", "attr_id", "attr_type", "attr_value"):
        frame[column] = frame[column].astype(str)
    return frame.drop_duplicates(["article_id", "attr_id"])


def build_trend_attributes(
    prediction_sample_view: pd.DataFrame,
    *,
    source_week: int | None = None,
    limit_per_type: int = 50,
) -> pd.DataFrame:
    if limit_per_type <= 0:
        raise ValueError("limit_per_type must be positive")
    if prediction_sample_view.empty:
        return pd.DataFrame(columns=_trend_attribute_columns())
    week = _default_source_week(prediction_sample_view, source_week)
    frame = prediction_sample_view.loc[
        prediction_sample_view["week_id"].astype(int).eq(week)
    ].copy()
    frame["pred_target_growth"] = _finite_float_series(
        frame["pred_target_growth"],
        "pred_target_growth",
    )
    frame = frame.sort_values(
        ["attr_type", "pred_target_growth", "attr_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    frame["rank"] = frame.groupby("attr_type").cumcount() + 1
    frame = frame.loc[frame["rank"] <= limit_per_type].copy()
    frame["source_week"] = week
    frame["target_week"] = week + 1
    for column in ("attr_id", "attr_type", "attr_value"):
        frame[column] = frame[column].astype(str)
    frame["heat_t"] = _finite_float_series(frame["heat_t"], "heat_t")
    frame["is_trend_eligible_t"] = frame["is_trend_eligible_t"].astype(int)
    return frame.loc[:, _trend_attribute_columns()].sort_values(
        ["attr_type", "rank"],
        kind="mergesort",
    )


def build_trend_attributes_for_source_weeks(
    prediction_sample_view: pd.DataFrame,
    source_weeks: Sequence[int],
    *,
    limit_per_type: int = 50,
) -> pd.DataFrame:
    if limit_per_type <= 0:
        raise ValueError("limit_per_type must be positive")
    if prediction_sample_view.empty:
        return pd.DataFrame(columns=_trend_attribute_columns())

    available_weeks = set(
        pd.to_numeric(prediction_sample_view["week_id"], errors="raise").astype(int)
    )
    frames: list[pd.DataFrame] = []
    for week in sorted({int(value) for value in source_weeks}):
        if week not in available_weeks:
            continue
        frame = build_trend_attributes(
            prediction_sample_view,
            source_week=week,
            limit_per_type=limit_per_type,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=_trend_attribute_columns())
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["source_week", "attr_type", "rank"],
        kind="mergesort",
    )
    duplicate_keys = result.duplicated(["source_week", "attr_type", "rank"])
    if duplicate_keys.any():
        sample = result.loc[
            duplicate_keys,
            ["source_week", "attr_type", "rank", "attr_id"],
        ]
        raise ValueError(
            "trend_attributes contain duplicate display rank keys: "
            f"{sample.head(3).to_dict('records')}"
        )
    return result.reset_index(drop=True)


def build_attribute_heat_series(
    trend_attributes: pd.DataFrame,
    attribute_week_heat: pd.DataFrame,
    prediction_sample_view: pd.DataFrame,
    *,
    weeks: int = 8,
) -> pd.DataFrame:
    if weeks <= 0:
        raise ValueError("weeks must be positive")
    rows: list[pd.DataFrame] = []
    predictions = _prediction_lookup(prediction_sample_view)
    for attr in trend_attributes.itertuples(index=False):
        source_week = int(attr.source_week)
        attr_id = str(attr.attr_id)
        history = attribute_week_heat.loc[
            (attribute_week_heat["attr_id"].astype(str) == attr_id)
            & (attribute_week_heat["week_id"].astype(int) <= source_week)
        ].copy()
        history["week_id"] = pd.to_numeric(history["week_id"], errors="raise").astype(
            int
        )
        history = history.sort_values("week_id", kind="mergesort").tail(weeks)
        if history.empty:
            continue
        history["attr_id"] = attr_id
        history["attr_type"] = history["attr_type"].astype(str)
        history["attr_value"] = history["attr_value"].astype(str)
        history["heat"] = pd.to_numeric(history["heat_cnt"], errors="raise")
        history["actual_target_growth"] = np.nan
        history["pred_target_growth"] = np.nan
        history["pred_share_t1"] = np.nan
        for row_index, week_id in history["week_id"].items():
            prediction = predictions.get((attr_id, int(week_id)))
            if prediction is None:
                continue
            history.at[row_index, "actual_target_growth"] = prediction["target_growth"]
            history.at[row_index, "pred_target_growth"] = prediction[
                "pred_target_growth"
            ]
            history.at[row_index, "pred_share_t1"] = prediction["pred_share_t1"]
        rows.append(
            history.loc[
                :,
                [
                    "attr_id",
                    "attr_type",
                    "attr_value",
                    "week_id",
                    "heat",
                    "actual_target_growth",
                    "pred_target_growth",
                    "pred_share_t1",
                ],
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "attr_id",
                "attr_type",
                "attr_value",
                "week_id",
                "heat",
                "actual_target_growth",
                "pred_target_growth",
                "pred_share_t1",
            ]
        )
    return (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(
            ["attr_id", "week_id"],
            keep="last",
        )
        .sort_values(
            ["attr_id", "week_id"],
            kind="mergesort",
        )
    )


def build_attribute_hierarchy_edges(
    hierarchy_edges: pd.DataFrame,
    attribute_nodes: pd.DataFrame,
) -> pd.DataFrame:
    parent_nodes = attribute_nodes.loc[:, ["attr_id", "attr_value"]].rename(
        columns={"attr_id": "parent_attr_id", "attr_value": "parent_attr_value"}
    )
    child_nodes = attribute_nodes.loc[:, ["attr_id", "attr_value"]].rename(
        columns={"attr_id": "child_attr_id", "attr_value": "child_attr_value"}
    )
    frame = hierarchy_edges.copy()
    frame["parent_attr_id"] = frame["parent_attr_id"].astype(str)
    frame["child_attr_id"] = frame["child_attr_id"].astype(str)
    frame = frame.merge(parent_nodes, on="parent_attr_id", how="left")
    frame = frame.merge(child_nodes, on="child_attr_id", how="left")
    if frame[["parent_attr_value", "child_attr_value"]].isna().any().any():
        raise ValueError("attribute hierarchy edge has missing attr values")
    return frame.loc[
        :,
        [
            "parent_attr_id",
            "child_attr_id",
            "parent_attr_type",
            "parent_attr_value",
            "child_attr_type",
            "child_attr_value",
            "relation_type",
        ],
    ]


def build_metrics_summary(
    trend_metric_payloads: (
        Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]
    ),
    recommendation_metric_payloads: (
        Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]
    ),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    order = 1
    for payload in _payload_values(trend_metric_payloads):
        for metric_row in flatten_trend_metrics(dict(payload)):
            for metric_name in (
                "mae",
                "rmse",
                "spearman",
                "ndcg_at_10",
                "precision_at_10",
                "recall_at_10",
            ):
                rows.append(
                    {
                        "metric_domain": "trend",
                        "model_or_method": metric_row["model_name"],
                        "split": metric_row["split"],
                        "metric_name": metric_name,
                        "metric_value": metric_row[metric_name],
                        "display_order": order,
                    }
                )
                order += 1
    for payload in _payload_values(recommendation_metric_payloads):
        for metric_row in flatten_recommendation_metrics(dict(payload)):
            for metric_name in (
                "map_at_12",
                "recall_at_12",
                "hit_rate_at_12",
                "ndcg_at_12",
                "coverage",
                "user_count",
            ):
                rows.append(
                    {
                        "metric_domain": "recommendation",
                        "model_or_method": metric_row["method"],
                        "split": metric_row["split"],
                        "metric_name": metric_name,
                        "metric_value": metric_row[metric_name],
                        "display_order": order,
                    }
                )
                order += 1
    return pd.DataFrame(rows)


def build_report_assets(manifest: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    output_artifacts = manifest.get("output_artifacts", {})
    figures = (
        output_artifacts.get("figures", [])
        if isinstance(output_artifacts, dict)
        else []
    )
    for path_value in figures:
        source_path = str(path_value)
        filename = source_path.rsplit("/", 1)[-1]
        rows.append(
            {
                "asset_name": filename,
                "asset_type": "figure",
                "title": filename.rsplit(".", 1)[0].replace("_", " "),
                "source_path": source_path,
                "static_url": f"/static/reports/{filename}",
                "description": "论文素材静态图表",
            }
        )
    return pd.DataFrame(rows)


def build_presentation_tables(sources: PresentationSources) -> dict[str, pd.DataFrame]:
    prediction_sample_view = build_lightgbm_prediction_sample_view(
        sources.predictions,
        sources.trend_model_samples,
    )
    trend_attributes = build_trend_attributes_for_source_weeks(
        prediction_sample_view,
        _display_source_weeks(prediction_sample_view, sources.report_cases),
    )
    recommendation_top12 = _top12_items_for_cases(
        sources.report_cases,
        sources.recommendation_items,
    )
    tables = {
        "app_metadata": _build_app_metadata(sources, trend_attributes),
        "demo_users": build_demo_users(sources.report_cases),
        "user_profile_attributes": build_user_profile_attributes(
            sources.report_cases,
            sources.user_profile,
        ),
        "recommendation_items": _build_recommendation_items_from_top12(
            recommendation_top12.copy(),
            sources.evaluation_labels,
        ),
        "recommendation_score_components": _build_recommendation_score_components_from_top12(
            recommendation_top12.copy()
        ),
        "articles": build_articles(sources.articles),
        "article_attributes": build_article_attributes(sources.article_attributes),
        "trend_attributes": trend_attributes,
        "attribute_heat_series": build_attribute_heat_series(
            trend_attributes,
            sources.attribute_week_heat,
            prediction_sample_view,
        ),
        "attribute_hierarchy_edges": build_attribute_hierarchy_edges(
            sources.attribute_hierarchy_edges,
            sources.attribute_nodes,
        ),
        "metrics_summary": build_metrics_summary(
            sources.trend_metrics,
            sources.recommendation_metrics,
        ),
        "report_assets": build_report_assets(sources.manifest),
    }
    return tables


def _display_source_weeks(
    prediction_sample_view: pd.DataFrame,
    report_cases: Sequence[Mapping[str, object]],
) -> list[int]:
    if prediction_sample_view.empty:
        return []
    available_weeks = set(
        pd.to_numeric(prediction_sample_view["week_id"], errors="raise").astype(int)
    )
    weeks = {_default_source_week(prediction_sample_view, None)}
    for payload in report_cases:
        cutoff_week = int(payload["cutoff_week"])
        if cutoff_week in available_weeks:
            weeks.add(cutoff_week)
    return sorted(weeks)


def _case_key(payload: Mapping[str, object]) -> tuple[str, str, int, int]:
    return (
        str(payload["customer_id"]),
        str(payload["split"]),
        int(payload["cutoff_week"]),
        int(payload["label_week"]),
    )


def _primary_tags(payload: Mapping[str, object], hit_count: int) -> list[str]:
    tags = ["命中用户" if hit_count > 0 else "未命中样本"]
    profile = payload.get("profile", [])
    if isinstance(profile, Sequence) and profile:
        first = profile[0]
        if isinstance(first, Mapping):
            tags.append(f"{first.get('attr_type')}: {first.get('attr_value')}")
    recommendations = payload.get("recommendations", [])
    if isinstance(recommendations, Sequence) and recommendations:
        tags.append("Top-12 推荐")
    return tags


def _profile_summary(payload: Mapping[str, object]) -> str:
    profile = payload.get("profile", [])
    if not isinstance(profile, Sequence) or not profile:
        return "该演示用户暂无可展示的偏好属性。"
    first = profile[0]
    if not isinstance(first, Mapping):
        return "该演示用户包含可展示的偏好属性。"
    attr_type = first.get("attr_type")
    attr_value = first.get("attr_value")
    return f"用户偏好集中在 {attr_type}={attr_value}。"


def _recommendation_summary(payload: Mapping[str, object], hit_count: int) -> str:
    recommendations = payload.get("recommendations", [])
    count = len(recommendations) if isinstance(recommendations, Sequence) else 0
    return f"展示 {count} 个推荐商品，命中评价标签 {hit_count} 个。"


def _filter_to_cases(
    dataframe: pd.DataFrame,
    report_cases: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    if dataframe.empty or not report_cases:
        return dataframe.iloc[0:0].copy()
    keys = pd.DataFrame(
        [
            {
                "customer_id": customer_id,
                "split": split,
                "cutoff_week": cutoff_week,
                "label_week": label_week,
            }
            for customer_id, split, cutoff_week, label_week in map(
                _case_key,
                report_cases,
            )
        ]
    )
    frame = dataframe.copy()
    frame["customer_id"] = frame["customer_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["cutoff_week"] = frame["cutoff_week"].astype(int)
    frame["label_week"] = frame["label_week"].astype(int)
    return frame.merge(keys, on=list(CASE_KEY_COLUMNS), how="inner")


def _add_case_id_column(frame: pd.DataFrame) -> None:
    frame["case_id"] = [
        build_case_id(
            str(row.customer_id),
            str(row.split),
            int(row.cutoff_week),
            int(row.label_week),
        )
        for row in frame.itertuples(index=False)
    ]


def _top12_items_for_cases(
    report_cases: Sequence[Mapping[str, object]],
    recommendation_items: pd.DataFrame,
) -> pd.DataFrame:
    items = _filter_to_cases(recommendation_items, report_cases).copy()
    if items.empty and report_cases:
        raise ValueError("Top-12 recommendation items are incomplete for demo cases")
    items["rank"] = items["rank"].astype(int)
    _add_case_id_column(items)
    invalid_rank_range = ~items["rank"].between(1, TOP_K)
    if invalid_rank_range.any():
        sample = items.loc[invalid_rank_range, ["case_id", "article_id", "rank"]]
        raise ValueError(
            f"Top-12 rank set invalid: {sample.head(3).to_dict('records')}"
        )
    items = items.loc[items["rank"] <= TOP_K].copy()
    for column in ("customer_id", "article_id", "candidate_sources"):
        items[column] = items[column].astype(str)
    duplicate_articles = items.duplicated(["case_id", "article_id"], keep=False)
    if duplicate_articles.any():
        raise ValueError("Top-12 recommendation items contain duplicate articles")
    counts = items.groupby("case_id")["rank"].nunique()
    incomplete = counts[counts != TOP_K]
    expected_case_ids = {build_case_id(*_case_key(payload)) for payload in report_cases}
    missing = expected_case_ids - set(counts.index)
    if not incomplete.empty or missing:
        raise ValueError(
            "Top-12 recommendation items incomplete: "
            f"incomplete={incomplete.to_dict()}, missing={sorted(missing)}"
        )
    invalid_ranks = _invalid_top12_rank_sets(items)
    if invalid_ranks:
        raise ValueError(f"Top-12 rank set invalid: {invalid_ranks}")
    return items


def _invalid_top12_rank_sets(items: pd.DataFrame) -> dict[str, list[int]]:
    expected = set(range(1, TOP_K + 1))
    invalid: dict[str, list[int]] = {}
    for case_id, ranks in items.groupby("case_id")["rank"]:
        actual = set(ranks.astype(int))
        if actual != expected:
            invalid[str(case_id)] = sorted(actual)
    return invalid


def _label_keys(evaluation_labels: pd.DataFrame) -> set[tuple[str, str, int, int, str]]:
    if evaluation_labels.empty:
        return set()
    labels = evaluation_labels.copy()
    labels["customer_id"] = labels["customer_id"].astype(str)
    labels["split"] = labels["split"].astype(str)
    labels["cutoff_week"] = labels["cutoff_week"].astype(int)
    labels["label_week"] = labels["label_week"].astype(int)
    labels["article_id"] = labels["article_id"].astype(str)
    return set(_iter_item_label_keys(labels))


def _iter_item_label_keys(
    dataframe: pd.DataFrame,
) -> list[tuple[str, str, int, int, str]]:
    return [
        (
            str(row.customer_id),
            str(row.split),
            int(row.cutoff_week),
            int(row.label_week),
            str(row.article_id),
        )
        for row in dataframe.itertuples(index=False)
    ]


def _finite_float_series(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column} must contain finite values")
    return values.astype(float)


def _select_with_defaults(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    frame = dataframe.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame.loc[:, list(columns)].copy()


def _require_columns(
    dataframe: pd.DataFrame,
    artifact_name: str,
    required_columns: Sequence[str],
) -> None:
    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{artifact_name} 缺少必需列: {missing}")


def _default_source_week(
    prediction_sample_view: pd.DataFrame,
    source_week: int | None,
) -> int:
    if source_week is not None:
        return int(source_week)
    frame = prediction_sample_view
    if "split" in frame.columns:
        test_frame = frame.loc[frame["split"].astype(str) == "test"]
        if not test_frame.empty:
            frame = test_frame
    return int(pd.to_numeric(frame["week_id"], errors="raise").max())


def _trend_attribute_columns() -> list[str]:
    return [
        "source_week",
        "target_week",
        "attr_id",
        "attr_type",
        "attr_value",
        "rank",
        "heat_t",
        "pred_share_t1",
        "pred_target_growth",
        "is_trend_eligible_t",
    ]


def _prediction_lookup(
    prediction_sample_view: pd.DataFrame,
) -> dict[tuple[str, int], dict[str, float]]:
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    if prediction_sample_view.empty:
        return lookup
    for row in prediction_sample_view.itertuples(index=False):
        lookup[(str(row.attr_id), int(row.week_id))] = {
            "target_growth": float(row.target_growth),
            "pred_target_growth": float(row.pred_target_growth),
            "pred_share_t1": float(row.pred_share_t1),
        }
    return lookup


def _payload_values(
    payloads: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(payloads, Mapping):
        return list(payloads.values())
    return list(payloads)


def _build_app_metadata(
    sources: PresentationSources,
    trend_attributes: pd.DataFrame,
) -> pd.DataFrame:
    source_artifacts = sources.source_artifacts or {}
    manifest_metadata = source_artifacts.get("reports_manifest", {})
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "default_source_week": (
            int(trend_attributes["source_week"].max())
            if not trend_attributes.empty
            else None
        ),
        "case_count": len(sources.report_cases),
        "artifact_warnings": sources.manifest.get("warnings", []),
        "core_trend_attr_types": list(CORE_TREND_ATTR_TYPES),
        "source_artifacts": source_artifacts,
        "source_manifest_path": manifest_metadata.get("path", ""),
    }
    return pd.DataFrame(
        [
            {"key": key, "value": _metadata_value(value)}
            for key, value in metadata.items()
        ]
    )


def _metadata_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
