from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def build_trend_model_metrics_figure(metrics: pd.DataFrame) -> Figure:
    _require_columns(
        metrics,
        required=("model_name", "split", "ndcg_at_10"),
        artifact_name="trend_model_metrics",
    )
    _require_non_empty(metrics, artifact_name="trend_model_metrics")
    _reject_duplicate_metric_key(
        metrics,
        key_columns=("model_name", "split"),
        artifact_name="trend_model_metrics",
    )

    pivot = metrics.pivot(index="model_name", columns="split", values="ndcg_at_10")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("趋势模型排序指标对比 NDCG@10")
    axis.set_xlabel("model_name")
    axis.set_ylabel("NDCG@10")
    axis.legend(title="split")
    figure.tight_layout()
    return figure


def build_recommendation_method_metrics_figure(metrics: pd.DataFrame) -> Figure:
    _require_columns(
        metrics,
        required=("method", "split", "ndcg_at_12"),
        artifact_name="recommendation_method_metrics",
    )
    _require_non_empty(metrics, artifact_name="recommendation_method_metrics")
    _reject_duplicate_metric_key(
        metrics,
        key_columns=("method", "split"),
        artifact_name="recommendation_method_metrics",
    )

    pivot = metrics.pivot(index="method", columns="split", values="ndcg_at_12")
    figure, axis = plt.subplots(figsize=(9, 4.8))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("推荐方法排序指标对比 NDCG@12")
    axis.set_xlabel("method")
    axis.set_ylabel("NDCG@12")
    axis.legend(title="split")
    figure.tight_layout()
    return figure


def build_feature_importance_figure(
    feature_importance: pd.DataFrame,
    *,
    top_n: int = 15,
) -> Figure:
    if top_n <= 0:
        raise ValueError(f"top_n 必须为正数: {top_n}")
    _require_columns(
        feature_importance,
        required=("feature", "normalized_gain_importance"),
        artifact_name="feature_importance",
    )
    _require_non_empty(feature_importance, artifact_name="feature_importance")

    top_features = (
        feature_importance.sort_values("normalized_gain_importance", ascending=False)
        .head(top_n)
        .sort_values("normalized_gain_importance")
    )
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(top_features["feature"], top_features["normalized_gain_importance"])
    axis.set_title("LightGBM 特征重要性 Top-N")
    axis.set_xlabel("normalized gain")
    axis.set_ylabel("feature")
    figure.tight_layout()
    return figure


def build_trend_curve_examples_figure(
    trend_view: pd.DataFrame,
    *,
    week_id: int,
    lookback_weeks: int = 8,
    top_n: int = 3,
) -> Figure:
    if lookback_weeks <= 0:
        raise ValueError(f"lookback_weeks 必须为正数: {lookback_weeks}")
    if top_n <= 0:
        raise ValueError(f"top_n 必须为正数: {top_n}")
    _require_columns(
        trend_view,
        required=(
            "week_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "heat_t",
            "pred_share_t1",
            "pred_target_growth",
            "is_trend_eligible_t",
        ),
        artifact_name="trend_view",
    )
    _require_non_empty(trend_view, artifact_name="trend_view")
    _reject_duplicate_trend_view_key(trend_view)
    plot_view = trend_view.copy()
    plot_view["_week_id_int"] = _normalize_trend_view_week_ids(plot_view["week_id"])

    current = plot_view.loc[
        (plot_view["_week_id_int"] == week_id)
        & (plot_view["is_trend_eligible_t"].astype(bool))
    ].copy()
    examples = (
        current.sort_values("pred_target_growth", ascending=False)
        .drop_duplicates("attr_type")
        .head(top_n)
    )
    if examples.empty:
        raise ValueError(f"week_id={week_id} 没有可绘制的趋势曲线案例。")

    axes_count = len(examples) * 3
    figure, axes = plt.subplots(
        axes_count,
        1,
        figsize=(9, 2.0 * axes_count),
        sharex=True,
    )
    if axes_count == 1:
        axes = [axes]

    min_week = week_id - lookback_weeks + 1
    for index, row in enumerate(examples.itertuples(index=False)):
        history = plot_view.loc[
            (plot_view["attr_id"].astype(str) == str(row.attr_id))
            & (plot_view["attr_type"].astype(str) == str(row.attr_type))
            & (plot_view["attr_value"].astype(str) == str(row.attr_value))
            & (plot_view["_week_id_int"].between(min_week, week_id))
        ].sort_values("_week_id_int")
        if history.empty:
            raise ValueError(
                f"week_id={week_id} 没有可绘制的趋势曲线历史: "
                f"{row.attr_type}={row.attr_value}"
            )

        heat_axis = axes[index * 3]
        share_axis = axes[index * 3 + 1]
        growth_axis = axes[index * 3 + 2]
        heat_axis.plot(history["_week_id_int"], history["heat_t"], marker="o")
        share_axis.plot(
            history["_week_id_int"],
            history["pred_share_t1"],
            marker="s",
            color="tab:orange",
        )
        growth_axis.bar(
            history["_week_id_int"],
            history["pred_target_growth"],
            alpha=0.25,
            color="tab:green",
        )
        heat_axis.set_title(f"{row.attr_type}: {row.attr_value}")
        heat_axis.set_ylabel("heat_t")
        share_axis.set_ylabel("pred_share_t1")
        growth_axis.set_ylabel("pred_target_growth")
    axes[-1].set_xlabel("week_id")
    figure.suptitle(f"典型趋势属性最近 {lookback_weeks} 周曲线")
    figure.subplots_adjust(hspace=0.45, top=0.92)
    return figure


def _require_columns(
    dataframe: pd.DataFrame,
    *,
    required: tuple[str, ...],
    artifact_name: str,
) -> None:
    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{artifact_name} 缺少列: {missing}")


def _require_non_empty(dataframe: pd.DataFrame, *, artifact_name: str) -> None:
    if dataframe.empty:
        raise ValueError(f"{artifact_name} 无可绘制数据")


def _reject_duplicate_metric_key(
    dataframe: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    duplicated = dataframe.duplicated(list(key_columns), keep=False)
    if not duplicated.any():
        return
    sample = (
        dataframe.loc[duplicated, list(key_columns)]
        .drop_duplicates()
        .head(3)
        .to_dict("records")
    )
    raise ValueError(f"{artifact_name} 存在重复 metric key: {sample}")


def _reject_duplicate_trend_view_key(trend_view: pd.DataFrame) -> None:
    key_columns = ("attr_id", "attr_type", "attr_value", "week_id")
    normalized_keys = pd.DataFrame(
        {
            "attr_id": trend_view["attr_id"].astype(str),
            "attr_type": trend_view["attr_type"].astype(str),
            "attr_value": trend_view["attr_value"].astype(str),
            "week_id": _normalize_trend_view_week_ids(trend_view["week_id"]),
        },
        index=trend_view.index,
    )
    duplicated = normalized_keys.duplicated(list(key_columns), keep=False)
    if not duplicated.any():
        return
    sample = (
        normalized_keys.loc[duplicated, list(key_columns)]
        .drop_duplicates()
        .head(3)
        .to_dict("records")
    )
    raise ValueError(f"trend_view 存在重复趋势曲线 key: {sample}")


def _normalize_trend_view_week_ids(week_ids: pd.Series) -> pd.Series:
    normalized: list[int] = []
    invalid_values: list[str] = []
    for value in week_ids:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            invalid_values.append(str(value))
    if invalid_values:
        sample = list(dict.fromkeys(invalid_values))[:3]
        raise ValueError(f"trend_view week_id 无法转换为 int: {sample}")
    return pd.Series(normalized, index=week_ids.index)
