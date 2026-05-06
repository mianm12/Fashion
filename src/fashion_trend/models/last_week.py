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
    validate_required_columns,
    validate_trend_model_predictions,
)

LAST_WEEK_MODEL_NAME = "last_week"
LAST_WEEK_PARAMS: dict[str, object] = {
    "model_name": LAST_WEEK_MODEL_NAME,
    "formula": "pred_target_growth = growth_lag_1",
    "derived_formula": (
        "pred_share_t1 = exp(pred_target_growth) * "
        "(share_t + epsilon) - epsilon"
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
    "growth_lag_1",
    "target_growth",
    "target_rank_in_type_t1",
)


def predict_last_week(split_samples: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(
        set(LAST_WEEK_REQUIRED_COLUMNS) - set(split_samples.columns)
    )
    if missing_columns:
        raise ValueError(
            "last_week 模型输入样本缺少必需列: " + ", ".join(missing_columns)
        )
    validate_required_columns(
        split_samples.columns.tolist(),
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
            "growth_lag_1",
            "target_growth",
            "target_rank_in_type_t1",
        ],
    ].copy()
    predictions.insert(4, "model_name", LAST_WEEK_MODEL_NAME)
    predictions["pred_target_growth"] = predictions["growth_lag_1"]
    epsilon = float(LAST_WEEK_PARAMS["epsilon"])
    predictions["pred_share_t1"] = (
        np.exp(predictions["pred_target_growth"]) * (predictions["share_t"] + epsilon)
        - epsilon
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return predictions


class LastWeekTrainer:
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
