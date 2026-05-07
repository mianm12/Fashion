from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    read_weekly_transactions,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    read_article_attribute_edges,
    read_attribute_nodes,
    read_attribute_week_heat,
    validate_all_sales_articles_have_attribute_edges,
    validate_article_attribute_edges_for_heat,
    validate_attribute_edge_node_metadata_consistency,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.io import (
    remove_file_if_exists,
    write_json,
    write_trend_csv,
    write_trend_parquet,
)
from fashion_trend.trend.schema import (
    ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS,
    ARTICLE_ATTRIBUTE_EDGE_HEAT_DTYPES,
    ARTICLE_WEEK_SALES_COLUMNS,
    ARTICLE_WEEK_SALES_DTYPES,
    ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
    ATTRIBUTE_HIERARCHY_EDGE_DTYPES,
    ATTRIBUTE_NODE_HEAT_COLUMNS,
    ATTRIBUTE_NODE_HEAT_DTYPES,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_DTYPES,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_DTYPES,
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_PRED_SHARE_GROUP_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
    TREND_MODEL_SHARE_TOLERANCE,
    TREND_MODEL_SPLIT_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
    WEEKLY_TRANSACTION_COLUMNS,
)
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    read_attribute_week_target,
    validate_attribute_week_target,
    validate_attribute_week_target_matches_heat,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)


def read_attribute_hierarchy_edges(
    attribute_hierarchy_edges_path: Path,
) -> pd.DataFrame:
    if not attribute_hierarchy_edges_path.exists():
        raise FileNotFoundError(f"属性层级边表不存在: {attribute_hierarchy_edges_path}")

    try:
        header = pd.read_csv(attribute_hierarchy_edges_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"无法读取属性层级边表: {attribute_hierarchy_edges_path}"
        ) from exc

    missing_columns = sorted(
        set(ATTRIBUTE_HIERARCHY_EDGE_COLUMNS) - set(header.columns)
    )
    if missing_columns:
        raise ValueError(
            "属性层级边表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_hierarchy_edges_path}"
        )

    try:
        return pd.read_csv(
            attribute_hierarchy_edges_path,
            usecols=list(ATTRIBUTE_HIERARCHY_EDGE_COLUMNS),
            dtype=ATTRIBUTE_HIERARCHY_EDGE_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"无法读取属性层级边表: {attribute_hierarchy_edges_path}"
        ) from exc


def build_attribute_graph_features_frame(
    attribute_nodes: pd.DataFrame,
    attribute_hierarchy_edges: pd.DataFrame,
) -> pd.DataFrame:
    validate_attribute_nodes_for_heat(attribute_nodes)
    validate_required_columns(
        attribute_hierarchy_edges.columns.tolist(),
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
        trend_model_samples.columns.tolist(),
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


def build_trend_model_split_frames(
    trend_model_samples: pd.DataFrame,
    valid_weeks: int,
    test_weeks: int,
) -> dict[str, pd.DataFrame]:
    validate_trend_model_samples(trend_model_samples)
    if valid_weeks <= 0:
        raise ValueError("valid_weeks 必须为正整数。")
    if test_weeks <= 0:
        raise ValueError("test_weeks 必须为正整数。")

    week_ids = sorted(trend_model_samples["week_id"].unique().tolist())
    required_week_count = valid_weeks + test_weeks + 1
    if len(week_ids) < required_week_count:
        raise ValueError(
            "样本周数不足，无法生成非空 train/valid/test: "
            f"当前 {len(week_ids)} 周，valid_weeks={valid_weeks}, "
            f"test_weeks={test_weeks}。"
        )

    max_sample_week = max(week_ids)
    test_start_week = max_sample_week - test_weeks + 1
    valid_start_week = test_start_week - valid_weeks

    split_masks = {
        "train": trend_model_samples["week_id"] < valid_start_week,
        "valid": (trend_model_samples["week_id"] >= valid_start_week)
        & (trend_model_samples["week_id"] < test_start_week),
        "test": trend_model_samples["week_id"] >= test_start_week,
    }
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = trend_model_samples.loc[split_masks[split_name]].copy()
        split_frame.insert(0, "split", split_name)
        split_frame = split_frame.loc[:, list(TREND_MODEL_SPLIT_COLUMNS)].sort_values(
            ["week_id", "attr_type", "attr_id"],
            ignore_index=True,
        )
        split_frames[split_name] = split_frame

    validate_trend_model_split_frames(split_frames, trend_model_samples)
    return split_frames


def validate_trend_model_split_frames(
    split_frames: dict[str, pd.DataFrame],
    original_samples: pd.DataFrame | None = None,
) -> None:
    missing_splits = set(TREND_MODEL_SPLIT_VALUES) - set(split_frames)
    if missing_splits:
        raise ValueError(f"趋势样本切分缺少 split: {sorted(missing_splits)}")

    combined_parts: list[pd.DataFrame] = []
    previous_max_week: int | None = None
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = split_frames[split_name]
        validate_trend_model_split_frame(split_frame, expected_split=split_name)
        min_week = int(split_frame["week_id"].min())
        max_week = int(split_frame["week_id"].max())
        if previous_max_week is not None and min_week <= previous_max_week:
            raise ValueError("趋势样本 split 周范围必须按时间递增且互不重叠。")
        previous_max_week = max_week
        combined_parts.append(split_frame.drop(columns=["split"]))

    if original_samples is not None:
        combined = pd.concat(combined_parts, ignore_index=True)
        combined_keys = combined.loc[:, ["week_id", "attr_id"]].sort_values(
            ["week_id", "attr_id"],
            ignore_index=True,
        )
        original_keys = original_samples.loc[:, ["week_id", "attr_id"]].sort_values(
            ["week_id", "attr_id"],
            ignore_index=True,
        )
        if not combined_keys.equals(original_keys):
            raise ValueError("趋势样本 split 合并后无法覆盖原始样本全集。")


def validate_trend_model_split_frame(
    split_frame: pd.DataFrame,
    expected_split: str | None = None,
) -> None:
    validate_required_columns(
        split_frame.columns.tolist(),
        TREND_MODEL_SPLIT_COLUMNS,
        source_name="趋势样本 split",
    )
    validate_no_missing_values(
        split_frame,
        TREND_MODEL_SPLIT_COLUMNS,
        source_name="趋势样本 split",
    )
    if split_frame.empty:
        raise ValueError("趋势样本 split 为空。")

    split_values = set(split_frame["split"])
    invalid_split_values = sorted(split_values - set(TREND_MODEL_SPLIT_VALUES))
    if invalid_split_values:
        raise ValueError(f"趋势样本 split 存在非法 split: {invalid_split_values}")
    if expected_split is not None and split_values != {expected_split}:
        raise ValueError(f"{expected_split} 趋势样本 split 字段不一致。")
    if expected_split is None and len(split_values) != 1:
        raise ValueError("趋势样本 split 字段必须固定为单一值。")

    validate_unique_key(
        split_frame,
        ["week_id", "attr_id"],
        source_name="趋势样本 split",
    )


def build_trend_model_split_metadata(
    split_frames: dict[str, pd.DataFrame],
    input_path: Path,
    output_paths: dict[str, Path],
    valid_weeks: int,
    test_weeks: int,
) -> dict[str, object]:
    validate_trend_model_split_frames(split_frames)
    split_metadata: dict[str, dict[str, object]] = {}
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = split_frames[split_name]
        split_metadata[split_name] = {
            "path": str(output_paths[split_name]),
            "rows": int(len(split_frame)),
            "weeks": int(split_frame["week_id"].nunique()),
            "attributes": int(split_frame["attr_id"].nunique()),
            "week_min": int(split_frame["week_id"].min()),
            "week_max": int(split_frame["week_id"].max()),
        }
    return {
        "split_strategy": "time",
        "valid_weeks": int(valid_weeks),
        "test_weeks": int(test_weeks),
        "input_path": str(input_path),
        "splits": split_metadata,
    }


def read_trend_model_split(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"趋势样本 split 不存在: {input_path}")
    dataframe = pd.read_parquet(input_path)
    validate_required_columns(
        dataframe.columns.tolist(),
        TREND_MODEL_SPLIT_COLUMNS,
        source_name=f"趋势样本 split: {input_path}",
    )
    split_frame = dataframe.loc[:, list(TREND_MODEL_SPLIT_COLUMNS)].copy()
    validate_trend_model_split_frame(split_frame)
    return split_frame


def validate_trend_model_predictions(
    predictions: pd.DataFrame,
    split_samples: pd.DataFrame,
) -> None:
    if predictions.columns.tolist() != list(TREND_MODEL_PREDICTION_COLUMNS):
        raise ValueError("趋势模型预测表列必须与契约完全一致。")
    validate_required_columns(
        predictions.columns.tolist(),
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name="趋势模型预测表",
    )
    validate_no_missing_values(
        predictions,
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name="趋势模型预测表",
    )
    validate_unique_key(
        predictions,
        ["week_id", "attr_id", "model_name"],
        source_name="趋势模型预测表",
    )
    if not set(predictions["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("趋势模型预测表存在非法 split。")
    copied_sample_columns = (
        "week_id",
        "attr_id",
        "attr_type",
        "attr_value",
        "split",
        "share_t",
        "target_growth",
        "target_rank_in_type_t1",
    )
    validate_required_columns(
        split_samples.columns.tolist(),
        copied_sample_columns,
        source_name="趋势模型输入样本",
    )
    sorted_predictions = predictions.sort_values(
        ["week_id", "attr_id"],
        ignore_index=True,
    )
    sorted_samples = split_samples.sort_values(
        ["week_id", "attr_id"],
        ignore_index=True,
    )
    prediction_split = sorted_predictions.loc[:, ["week_id", "attr_id", "split"]]
    sample_split = sorted_samples.loc[:, ["week_id", "attr_id", "split"]]
    if not prediction_split.equals(sample_split):
        raise ValueError("趋势模型预测 split 与输入不一致。")
    prediction_copied_values = sorted_predictions.loc[:, list(copied_sample_columns)]
    sample_copied_values = sorted_samples.loc[:, list(copied_sample_columns)]
    if not prediction_copied_values.equals(sample_copied_values):
        raise ValueError("趋势模型预测字段与输入不一致。")

    numeric_values = sorted_predictions.drop(
        columns=["attr_id", "attr_type", "attr_value", "model_name", "split"]
    )
    try:
        finite_numeric_values = numeric_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势模型预测表无法校验数值字段。") from exc
    if not np.isfinite(finite_numeric_values).all():
        raise ValueError("趋势模型预测表存在非有限数值。")
    validate_pred_share_t1_distribution(sorted_predictions, "趋势模型预测表")


def derive_normalized_pred_share_t1(
    predictions: pd.DataFrame,
    epsilon: float,
) -> pd.Series:
    validate_required_columns(
        predictions.columns.tolist(),
        (*TREND_MODEL_PRED_SHARE_GROUP_COLUMNS, "share_t", "pred_target_growth"),
        source_name="趋势模型预测原始表",
    )
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势模型 pred_share_t1 平滑参数必须为数值。") from exc
    if epsilon_value < 0 or not np.isfinite(epsilon_value):
        raise ValueError("趋势模型 pred_share_t1 平滑参数必须为非负有限数值。")
    try:
        share_t = pd.to_numeric(predictions["share_t"], errors="raise")
        pred_growth = pd.to_numeric(predictions["pred_target_growth"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势模型 pred_share_t1 原始字段必须为数值。") from exc

    raw_share = np.exp(pred_growth) * (share_t + epsilon_value) - epsilon_value
    if not np.isfinite(raw_share.to_numpy(dtype=float)).all():
        raise ValueError("趋势模型 pred_share_t1 原始预测存在非有限数值。")

    non_negative_share = raw_share.clip(lower=0.0)
    group_total = non_negative_share.groupby(
        [predictions[column] for column in TREND_MODEL_PRED_SHARE_GROUP_COLUMNS],
        dropna=False,
    ).transform("sum")
    if (group_total <= 0).any():
        raise ValueError("趋势模型 pred_share_t1 原始预测组内总和必须大于 0。")

    normalized_share = non_negative_share / group_total
    normalized_share.name = "pred_share_t1"
    return normalized_share


def validate_pred_share_t1_distribution(
    predictions: pd.DataFrame,
    source_name: str,
) -> None:
    validate_required_columns(
        predictions.columns.tolist(),
        (*TREND_MODEL_PRED_SHARE_GROUP_COLUMNS, "pred_share_t1"),
        source_name=source_name,
    )
    try:
        pred_share = pd.to_numeric(predictions["pred_share_t1"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} pred_share_t1 必须为数值。") from exc
    if not np.isfinite(pred_share.to_numpy(dtype=float)).all():
        raise ValueError(f"{source_name} pred_share_t1 存在非有限数值。")

    below_zero = pred_share < -TREND_MODEL_SHARE_TOLERANCE
    above_one = pred_share > 1 + TREND_MODEL_SHARE_TOLERANCE
    if below_zero.any() or above_one.any():
        raise ValueError(f"{source_name} pred_share_t1 必须在 [0, 1] 范围内。")

    share_totals = pred_share.groupby(
        [predictions[column] for column in TREND_MODEL_PRED_SHARE_GROUP_COLUMNS],
        dropna=False,
    ).sum()
    invalid_totals = share_totals[
        ~np.isclose(
            share_totals,
            1.0,
            rtol=0,
            atol=TREND_MODEL_SHARE_TOLERANCE,
        )
    ]
    if not invalid_totals.empty:
        raise ValueError(
            f"{source_name} pred_share_t1 必须在 split/week_id/attr_type 内归一化。"
        )
