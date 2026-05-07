from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.trend import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
    derive_normalized_pred_share_t1,
    validate_required_columns,
    validate_trend_model_predictions,
)

MOVING_AVERAGE_MODEL_NAME = "moving_average"
MOVING_AVERAGE_GROWTH_LAGS: tuple[str, ...] = ("growth_lag_1", "growth_lag_2")
MOVING_AVERAGE_PARAMS: dict[str, object] = {
    "model_name": MOVING_AVERAGE_MODEL_NAME,
    "formula": "pred_target_growth = mean(growth_lag_1, growth_lag_2)",
    "derived_formula": (
        "raw_pred_share_t1 = exp(pred_target_growth) * "
        "(share_t + epsilon) - epsilon; "
        "pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))"
    ),
    "epsilon": 1e-6,
    "growth_lags": list(MOVING_AVERAGE_GROWTH_LAGS),
}

MOVING_AVERAGE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "split",
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "share_t",
    *MOVING_AVERAGE_GROWTH_LAGS,
    "target_growth",
    "target_rank_in_type_t1",
)


def predict_moving_average(split_samples: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(
        set(MOVING_AVERAGE_REQUIRED_COLUMNS) - set(split_samples.columns)
    )
    if missing_columns:
        raise ValueError(
            "moving_average 模型输入样本缺少必需列: " + ", ".join(missing_columns)
        )
    validate_required_columns(
        split_samples.columns.tolist(),
        MOVING_AVERAGE_REQUIRED_COLUMNS,
        source_name="moving_average 模型输入样本",
    )
    if not set(split_samples["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("moving_average 模型输入样本存在非法 split。")

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
    predictions.insert(4, "model_name", MOVING_AVERAGE_MODEL_NAME)
    growth_lags = _read_growth_lags(split_samples)
    predictions["pred_target_growth"] = growth_lags.mean(axis=1)
    epsilon = float(MOVING_AVERAGE_PARAMS["epsilon"])
    predictions["pred_share_t1"] = derive_normalized_pred_share_t1(
        predictions,
        epsilon,
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    _validate_finite_predictions(predictions)
    predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return predictions


class MovingAverageTrainer:
    name = MOVING_AVERAGE_MODEL_NAME
    model_type = MODEL_TYPE_BASELINE

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        split_samples = pd.concat(
            [context.split_frames[split_name] for split_name in context.split_order],
            ignore_index=True,
        )
        predictions = predict_moving_average(split_samples)
        validate_trend_model_predictions(predictions, split_samples)
        return TrendTrainResult(
            model_name=self.name,
            model_type=self.model_type,
            predictions=predictions,
            params=_copy_moving_average_params(),
        )


def _read_growth_lags(split_samples: pd.DataFrame) -> pd.DataFrame:
    try:
        growth_lags = split_samples.loc[:, list(MOVING_AVERAGE_GROWTH_LAGS)].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("moving_average 模型输入增长 lag 必须为数值。") from exc
    if not np.isfinite(growth_lags.to_numpy(dtype=float)).all():
        raise ValueError("moving_average 模型输入增长 lag 存在非有限数值。")
    return growth_lags


def _copy_moving_average_params() -> dict[str, object]:
    params = dict(MOVING_AVERAGE_PARAMS)
    params["growth_lags"] = list(MOVING_AVERAGE_GROWTH_LAGS)
    return params


def _validate_finite_predictions(predictions: pd.DataFrame) -> None:
    numeric_columns = [
        "share_t",
        "pred_share_t1",
        "target_growth",
        "pred_target_growth",
    ]
    try:
        numeric_values = predictions.loc[:, numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("moving_average 模型预测存在无法解析的数值字段。") from exc
    if not np.isfinite(numeric_values).all():
        raise ValueError("moving_average 模型预测存在非有限数值。")
