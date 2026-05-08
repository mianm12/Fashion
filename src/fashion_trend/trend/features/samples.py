from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.catalog.contracts import ATTRIBUTE_HIERARCHY_EDGE_COLUMNS
from fashion_trend.foundation.dataframe import (
    validate_no_missing_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
from fashion_trend.trend.heat.attribute_heat import (
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.labels.targets import (
    validate_attribute_week_target_matches_heat,
)
from fashion_trend.trend.schema import TREND_MODEL_SAMPLE_COLUMNS


def build_attribute_graph_features_frame(
    attribute_nodes: pd.DataFrame,
    attribute_hierarchy_edges: pd.DataFrame,
) -> pd.DataFrame:
    """构造属性图度数特征。

    参数:
        attribute_nodes: 属性节点表，提供属性全集、商品覆盖数和核心属性标志。
        attribute_hierarchy_edges: 属性层级边表，提供父子属性关系。

    返回:
        每个属性一行的图特征表，包含 `parent_count`、`child_count`
        和二者相加得到的 `degree`。

    异常:
        ValueError: 当节点或层级边契约不满足，或层级边引用未知属性时抛出。
    """
    validate_attribute_nodes_for_heat(attribute_nodes)
    validate_required_columns(
        attribute_hierarchy_edges,
        ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
        source_name="属性层级边表",
    )
    validate_no_missing_values(
        attribute_hierarchy_edges,
        ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
        source_name="属性层级边表",
    )
    validate_unique_key(
        attribute_hierarchy_edges,
        ["parent_attr_id", "child_attr_id", "relation_type"],
        source_name="属性层级边表",
    )
    validate_positive_values(
        attribute_hierarchy_edges,
        ["edge_weight"],
        source_name="属性层级边表",
    )

    known_attr_ids = set(attribute_nodes["attr_id"].astype("string"))
    referenced_attr_ids = set(
        pd.concat(
            [
                attribute_hierarchy_edges["parent_attr_id"],
                attribute_hierarchy_edges["child_attr_id"],
            ],
            ignore_index=True,
        ).astype("string")
    )
    missing_attr_ids = sorted(referenced_attr_ids - known_attr_ids)
    if missing_attr_ids:
        examples = ", ".join(missing_attr_ids[:5])
        raise ValueError(
            f"属性层级边表存在 {len(missing_attr_ids)} 个 attr_id "
            f"无法映射到属性节点，例如: {examples}"
        )

    features = attribute_nodes.loc[
        :, ["attr_id", "article_count", "is_core_attr"]
    ].copy()
    parent_counts = (
        attribute_hierarchy_edges.groupby("child_attr_id")
        .size()
        .rename("parent_count")
        .reset_index()
        .rename(columns={"child_attr_id": "attr_id"})
    )
    child_counts = (
        attribute_hierarchy_edges.groupby("parent_attr_id")
        .size()
        .rename("child_count")
        .reset_index()
        .rename(columns={"parent_attr_id": "attr_id"})
    )
    features = features.merge(parent_counts, on="attr_id", how="left")
    features = features.merge(child_counts, on="attr_id", how="left")
    features[["parent_count", "child_count"]] = (
        features[["parent_count", "child_count"]].fillna(0).astype("int64")
    )
    features["degree"] = features["parent_count"] + features["child_count"]
    return features


def build_trend_model_samples_frame(
    attribute_week_heat: pd.DataFrame,
    attribute_week_target: pd.DataFrame,
    attribute_nodes: pd.DataFrame,
    attribute_hierarchy_edges: pd.DataFrame,
    min_lag_weeks: int = 4,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """构造趋势模型训练样本表。

    参数:
        attribute_week_heat: 完整属性周热度面板，提供当周和历史热度特征。
        attribute_week_target: 由同一热度面板派生的趋势标签表。
        attribute_nodes: 属性节点表，提供商品覆盖数和核心属性标志。
        attribute_hierarchy_edges: 属性层级边表，用于生成图度数特征。
        min_lag_weeks: 样本最小历史周边界，必须覆盖固定 4 周特征窗口。
        epsilon: 与标签构造一致的占比增长率平滑参数。

    返回:
        满足 `TREND_MODEL_SAMPLE_COLUMNS` 的趋势训练样本表。样本使用固定
        4 周 lag、移动均值和波动特征，合并属性图度数特征，保留历史总热度、
        历史活跃周数和趋势资格标志，并只输出具有未来标签的非最后周样本。

    异常:
        ValueError: 当输入契约、`min_lag_weeks`、`epsilon` 或目标键完整性
            不满足要求时抛出。
    """
    validate_attribute_week_heat(attribute_week_heat)
    validate_attribute_nodes_for_heat(attribute_nodes)
    feature_lag_weeks = 4
    if min_lag_weeks < feature_lag_weeks:
        raise ValueError("min_lag_weeks 必须大于等于 4。")
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数。")
    validate_attribute_week_target_matches_heat(
        attribute_week_heat,
        attribute_week_target,
        epsilon=epsilon,
    )

    base = attribute_week_heat.sort_values(["attr_id", "week_id"]).copy()
    base = base.rename(
        columns={
            "heat_cnt": "heat_t",
            "heat_share": "share_t",
            "log_heat": "log_heat_t",
            "rank_in_type": "rank_in_type_t",
        }
    )
    grouped = base.groupby("attr_id", sort=False)
    # 固定 4 周特征窗口是训练样本契约的一部分，由 min_lag_weeks 保护边界。
    for lag in range(1, feature_lag_weeks + 1):
        base[f"heat_lag_{lag}"] = grouped["heat_t"].shift(lag)
        base[f"share_lag_{lag}"] = grouped["share_t"].shift(lag)

    base["growth_lag_1"] = np.log(
        (base["share_t"] + epsilon) / (base["share_lag_1"] + epsilon)
    )
    base["growth_lag_2"] = np.log(
        (base["share_lag_1"] + epsilon) / (base["share_lag_2"] + epsilon)
    )
    base["acc_lag_1"] = base["growth_lag_1"] - base["growth_lag_2"]

    rolling = grouped[["heat_t", "share_t"]].rolling(
        window=feature_lag_weeks,
        min_periods=feature_lag_weeks,
    )
    base["heat_ma_4"] = rolling["heat_t"].mean().reset_index(level=0, drop=True)
    base["share_ma_4"] = rolling["share_t"].mean().reset_index(level=0, drop=True)
    base["share_std_4"] = (
        grouped["share_t"]
        .rolling(window=feature_lag_weeks, min_periods=feature_lag_weeks)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    base["share_max_4"] = rolling["share_t"].max().reset_index(level=0, drop=True)
    base["share_min_4"] = rolling["share_t"].min().reset_index(level=0, drop=True)

    base["history_total_heat_t"] = grouped["heat_t"].cumsum()
    base["history_active_weeks_t"] = (
        base["heat_t"].gt(0).astype("int64").groupby(base["attr_id"]).cumsum()
    )
    base["is_trend_eligible_t"] = (base["history_total_heat_t"] >= 100) & (
        base["history_active_weeks_t"] >= 8
    )
    base["week_index"] = base["week_id"]
    base["week_mod_52"] = base["week_id"] % 52

    non_last_week_ids = set(attribute_week_heat["week_id"]) - {
        attribute_week_heat["week_id"].max()
    }
    # 先核对目标键，避免样本内连接静默丢弃本应存在的训练目标。
    expected_target_keys = base[
        (base["week_id"] >= min_lag_weeks) & (base["week_id"].isin(non_last_week_ids))
    ].loc[:, ["week_id", "attr_id"]]
    available_target_keys = attribute_week_target.loc[:, ["week_id", "attr_id"]]
    missing_target_keys = expected_target_keys.merge(
        available_target_keys,
        on=["week_id", "attr_id"],
        how="left",
        indicator=True,
    )
    missing_target_keys = missing_target_keys[
        missing_target_keys["_merge"] == "left_only"
    ]
    if not missing_target_keys.empty:
        example = missing_target_keys.iloc[0]
        raise ValueError(
            "趋势标签表缺失 "
            f"{len(missing_target_keys)} 个样本目标键，例如: "
            f"week_id={example.week_id}, attr_id={example.attr_id}"
        )

    graph_features = build_attribute_graph_features_frame(
        attribute_nodes,
        attribute_hierarchy_edges,
    )
    samples = base.merge(graph_features, on="attr_id", how="left")
    samples = samples.merge(
        attribute_week_target.loc[
            :,
            [
                "week_id",
                "attr_id",
                "target_growth",
                "target_log_heat_t1",
                "target_rank_in_type_t1",
            ],
        ],
        on=["week_id", "attr_id"],
        how="inner",
    )
    samples = samples[samples["week_id"] >= min_lag_weeks].copy()
    samples = samples.loc[:, list(TREND_MODEL_SAMPLE_COLUMNS)].sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    validate_trend_model_samples(samples)
    return samples


def validate_trend_model_samples(trend_model_samples: pd.DataFrame) -> None:
    """校验趋势训练样本表满足最终列契约、唯一键和数值有限性。

    参数:
        trend_model_samples: 待校验的趋势训练样本表。

    异常:
        ValueError: 当缺少契约列、存在缺失值、键重复或数值字段非有限时抛出。
    """
    validate_required_columns(
        trend_model_samples,
        TREND_MODEL_SAMPLE_COLUMNS,
        source_name="趋势训练样本表",
    )
    validate_no_missing_values(
        trend_model_samples,
        TREND_MODEL_SAMPLE_COLUMNS,
        source_name="趋势训练样本表",
    )
    validate_unique_key(
        trend_model_samples,
        ["week_id", "attr_id"],
        source_name="趋势训练样本表",
    )
    numeric_values = trend_model_samples.drop(
        columns=["attr_id", "attr_type", "attr_value"]
    )
    try:
        finite_numeric_values = numeric_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势训练样本表无法校验数值字段。") from exc
    if not np.isfinite(finite_numeric_values).all():
        raise ValueError("趋势训练样本表存在非有限数值。")
