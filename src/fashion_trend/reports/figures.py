from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def build_data_pipeline_figure() -> Figure:
    labels = [
        "H&M articles.csv",
        "属性图",
        "属性周热度",
        "LightGBM 趋势预测",
        "Top-N 推荐",
        "论文图表与案例",
    ]
    figure, axis = plt.subplots(figsize=(12, 2.8))
    axis.axis("off")
    x_positions = [0.02, 0.20, 0.38, 0.56, 0.74, 0.90]
    for index, (label, x_pos) in enumerate(zip(labels, x_positions)):
        box = FancyBboxPatch(
            (x_pos, 0.42),
            0.13,
            0.22,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            edgecolor="#334155",
            facecolor="#e0f2fe",
            transform=axis.transAxes,
        )
        axis.add_patch(box)
        axis.text(
            x_pos + 0.065,
            0.53,
            label,
            ha="center",
            va="center",
            fontsize=9,
            transform=axis.transAxes,
        )
        if index < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x_pos + 0.13, 0.53),
                (x_positions[index + 1], 0.53),
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.0,
                color="#334155",
                transform=axis.transAxes,
            )
            axis.add_patch(arrow)
    axis.set_title("数据处理与论文素材导出流程")
    return figure


def build_attribute_graph_schema_figure() -> Figure:
    nodes = {
        "商品节点": (0.15, 0.55),
        "属性节点": (0.48, 0.70),
        "父属性": (0.78, 0.78),
        "子属性": (0.78, 0.42),
    }
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    axis.axis("off")
    for label, (x_pos, y_pos) in nodes.items():
        box = FancyBboxPatch(
            (x_pos - 0.09, y_pos - 0.06),
            0.18,
            0.12,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            edgecolor="#475569",
            facecolor="#f8fafc",
            transform=axis.transAxes,
        )
        axis.add_patch(box)
        axis.text(
            x_pos, y_pos, label, ha="center", va="center", transform=axis.transAxes
        )
    for start, end, label in [
        ("商品节点", "属性节点", "article_has_attribute"),
        ("父属性", "子属性", "parent_contains_child"),
        ("属性节点", "父属性", "belongs_to_parent"),
    ]:
        start_xy = nodes[start]
        end_xy = nodes[end]
        axis.add_patch(
            FancyArrowPatch(
                start_xy,
                end_xy,
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.0,
                color="#475569",
                transform=axis.transAxes,
            )
        )
        axis.text(
            (start_xy[0] + end_xy[0]) / 2,
            (start_xy[1] + end_xy[1]) / 2 + 0.04,
            label,
            ha="center",
            fontsize=8,
            transform=axis.transAxes,
        )
    axis.set_title("商品属性层次图示意")
    return figure


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


def build_topk_trend_attributes_figure(
    trend_view: pd.DataFrame,
    *,
    week_id: int,
    top_k: int,
) -> Figure:
    if top_k <= 0:
        raise ValueError(f"top_k 必须为正数: {top_k}")
    _require_columns(
        trend_view,
        required=(
            "split",
            "week_id",
            "attr_type",
            "attr_value",
            "pred_target_growth",
            "heat_t",
            "history_total_heat_t",
            "history_active_weeks_t",
            "is_trend_eligible_t",
        ),
        artifact_name="trend_view",
    )
    target_types = (
        "colour_group_name",
        "product_type_name",
        "graphical_appearance_name",
    )
    filtered = trend_view.loc[
        (trend_view["split"].astype(str) == "test")
        & (trend_view["week_id"].astype(int) == int(week_id))
        & (trend_view["is_trend_eligible_t"].astype(bool))
        & (trend_view["heat_t"].astype(float) >= 20)
        & (trend_view["history_total_heat_t"].astype(float) >= 100)
        & (trend_view["history_active_weeks_t"].astype(float) >= 8)
        & (trend_view["attr_type"].astype(str).isin(target_types))
    ].copy()
    chart_data = (
        filtered.sort_values(
            ["attr_type", "pred_target_growth"], ascending=[True, False]
        )
        .groupby("attr_type", group_keys=False, sort=False)
        .head(top_k)
    )
    if chart_data.empty:
        raise ValueError("Top-K 趋势属性图没有可绘制数据。")
    figure, axes = plt.subplots(1, len(target_types), figsize=(13, 4.8), sharex=False)
    for axis, attr_type in zip(axes, target_types):
        subset = chart_data.loc[chart_data["attr_type"] == attr_type].sort_values(
            "pred_target_growth"
        )
        axis.barh(subset["attr_value"], subset["pred_target_growth"])
        axis.set_title(attr_type)
        axis.set_xlabel("pred_target_growth")
    figure.suptitle(f"test week {week_id} Top-K 趋势属性")
    figure.tight_layout()
    return figure


def build_recommendation_weight_analysis_figure(
    search_results: pd.DataFrame,
    *,
    best_weights: dict[str, float],
) -> Figure:
    _require_columns(
        search_results,
        required=("trend_score", "ndcg_at_12"),
        artifact_name="recommendation_weight_analysis",
    )
    _require_non_empty(search_results, artifact_name="recommendation_weight_analysis")
    weight_names = ("pop_score", "sim_score", "trend_score", "recent_score")
    missing_weights = sorted(set(weight_names) - set(best_weights))
    if missing_weights:
        raise ValueError(f"主实验权重缺少字段: {missing_weights}")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    scatter_axis, weight_axis = axes
    scatter_axis.scatter(search_results["trend_score"], search_results["ndcg_at_12"])
    scatter_axis.set_title("trend_score 权重与 valid NDCG@12")
    scatter_axis.set_xlabel("trend_score")
    scatter_axis.set_ylabel("valid NDCG@12")

    weight_values = [float(best_weights[name]) for name in weight_names]
    weight_axis.bar(weight_names, weight_values)
    weight_axis.set_title("主实验权重构成")
    weight_axis.set_ylabel("weight")
    weight_axis.tick_params(axis="x", labelrotation=30)
    figure.tight_layout()
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
