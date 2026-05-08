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
from fashion_trend.trend.attribute_heat import (
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.schema import TREND_MODEL_SAMPLE_COLUMNS
from fashion_trend.trend.targets import validate_attribute_week_target_matches_heat


def build_attribute_graph_features_frame(
    attribute_nodes: pd.DataFrame,
    attribute_hierarchy_edges: pd.DataFrame,
) -> pd.DataFrame:
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
