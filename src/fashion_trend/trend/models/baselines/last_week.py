from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.trend.predictions import validate_trend_model_predictions
from fashion_trend.trend.schema import (
    TREND_MODEL_PRED_SHARE_GROUP_COLUMNS,
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SHARE_TOLERANCE,
    TREND_MODEL_SPLIT_VALUES,
)

LAST_WEEK_MODEL_NAME = "last_week"
LAST_WEEK_PARAMS: dict[str, object] = {
    "model_name": LAST_WEEK_MODEL_NAME,
    "formula": "pred_share_t1 = group_normalize(share_t)",
    "derived_formula": (
        "pred_target_growth = log((pred_share_t1 + epsilon) / " "(share_t + epsilon))"
    ),
    "epsilon": 1e-6,
}

LAST_WEEK_REQUIRED_COLUMNS: tuple[str, ...] = (
    "split",
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "share_t",
    "target_growth",
    "target_rank_in_type_t1",
)


def predict_last_week(split_samples: pd.DataFrame) -> pd.DataFrame:
    """生成 Last Week Heat 基线预测表。

    预测语义是把当前 `share_t` 作为下一期份额分布预测，并在
    `split + week_id + attr_type` 内重归一化；`pred_target_growth` 再由
    预测份额和当前份额的平滑对数比值派生。
    """

    missing_columns = sorted(
        set(LAST_WEEK_REQUIRED_COLUMNS) - set(split_samples.columns)
    )
    if missing_columns:
        raise ValueError(
            "last_week 模型输入样本缺少必需列: " + ", ".join(missing_columns)
        )
    validate_required_columns(
        split_samples,
        LAST_WEEK_REQUIRED_COLUMNS,
        source_name="last_week 模型输入样本",
    )
    if not set(split_samples["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("last_week 模型输入样本存在非法 split。")

    predictions = split_samples.loc[
        :,
        [
            "week_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "split",
            "share_t",
            "target_growth",
            "target_rank_in_type_t1",
        ],
    ].copy()
    predictions.insert(4, "model_name", LAST_WEEK_MODEL_NAME)
    predictions["pred_share_t1"] = _derive_normalized_current_share(predictions)
    predictions["pred_target_growth"] = _derive_growth_from_pred_share(
        predictions,
        float(LAST_WEEK_PARAMS["epsilon"]),
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return predictions


class LastWeekTrainer:
    """last_week 基线训练器，为通用 runner 产出标准 TrendTrainResult。"""

    name = LAST_WEEK_MODEL_NAME
    model_type = MODEL_TYPE_BASELINE

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        split_samples = pd.concat(
            [context.split_frames[split_name] for split_name in context.split_order],
            ignore_index=True,
        )
        predictions = predict_last_week(split_samples)
        validate_trend_model_predictions(predictions, split_samples)
        return TrendTrainResult(
            model_name=self.name,
            model_type=self.model_type,
            predictions=predictions,
            params=dict(LAST_WEEK_PARAMS),
        )


def _derive_normalized_current_share(predictions: pd.DataFrame) -> pd.Series:
    """从当前 `share_t` 派生合法的下一期预测份额分布。"""
    validate_required_columns(
        predictions,
        (*TREND_MODEL_PRED_SHARE_GROUP_COLUMNS, "share_t"),
        source_name="last_week 模型预测原始表",
    )
    try:
        current_share = pd.to_numeric(predictions["share_t"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("last_week 模型 share_t 必须为数值。") from exc
    if not np.isfinite(current_share.to_numpy(dtype=float)).all():
        raise ValueError("last_week 模型 share_t 存在非有限数值。")

    below_zero = current_share < -TREND_MODEL_SHARE_TOLERANCE
    above_one = current_share > 1.0 + TREND_MODEL_SHARE_TOLERANCE
    if below_zero.any() or above_one.any():
        raise ValueError("last_week 模型 share_t 必须在 [0, 1] 容差范围内。")

    group_keys = [
        predictions[column] for column in TREND_MODEL_PRED_SHARE_GROUP_COLUMNS
    ]
    group_total = current_share.groupby(group_keys, dropna=False).transform("sum")
    if not np.isclose(
        group_total,
        1.0,
        rtol=0,
        atol=TREND_MODEL_SHARE_TOLERANCE,
    ).all():
        raise ValueError("last_week 模型 share_t 必须在组内归一化为 1。")

    non_negative_share = current_share.clip(lower=0.0)
    normalized_total = non_negative_share.groupby(
        group_keys,
        dropna=False,
    ).transform("sum")
    if (normalized_total <= 0).any():
        raise ValueError("last_week 模型 share_t 组内总和必须大于 0。")

    normalized_share = non_negative_share / normalized_total
    normalized_share.name = "pred_share_t1"
    return normalized_share


def _derive_growth_from_pred_share(
    predictions: pd.DataFrame,
    epsilon: float,
) -> pd.Series:
    """由预测份额和当前份额派生 Last Week Heat 的预测增长率。"""
    validate_required_columns(
        predictions,
        ("share_t", "pred_share_t1"),
        source_name="last_week 模型预测原始表",
    )
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("last_week 模型 epsilon 必须为数值。") from exc
    if epsilon_value < 0 or not np.isfinite(epsilon_value):
        raise ValueError("last_week 模型 epsilon 必须为非负有限数值。")
    try:
        share_t = pd.to_numeric(predictions["share_t"], errors="raise")
        pred_share_t1 = pd.to_numeric(predictions["pred_share_t1"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "last_week 模型 share_t 和 pred_share_t1 必须为数值。"
        ) from exc

    denominator = share_t + epsilon_value
    if (denominator <= 0).any():
        raise ValueError("last_week 模型 share_t 加 epsilon 后必须大于 0。")

    growth = np.log((pred_share_t1 + epsilon_value) / denominator)
    if not np.isfinite(growth.to_numpy(dtype=float)).all():
        raise ValueError("last_week 模型 pred_target_growth 存在非有限数值。")
    growth.name = "pred_target_growth"
    return growth
