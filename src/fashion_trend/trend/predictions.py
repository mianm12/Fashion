from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_PRED_SHARE_GROUP_COLUMNS,
    TREND_MODEL_SHARE_TOLERANCE,
    TREND_MODEL_SPLIT_VALUES,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_key,
)


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
