from __future__ import annotations

import pandas as pd

from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.trend.predictions import (
    derive_normalized_pred_share_t1,
    validate_trend_model_predictions,
)
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)

PREVIOUS_GROWTH_MODEL_NAME = "previous_growth"
PREVIOUS_GROWTH_PARAMS: dict[str, object] = {
    "model_name": PREVIOUS_GROWTH_MODEL_NAME,
    "formula": "pred_target_growth = growth_lag_1",
    "derived_formula": (
        "raw_pred_share_t1 = exp(pred_target_growth) * "
        "(share_t + epsilon) - epsilon; "
        "pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))"
    ),
    "epsilon": 1e-6,
}

PREVIOUS_GROWTH_REQUIRED_COLUMNS: tuple[str, ...] = (
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


def predict_previous_growth(split_samples: pd.DataFrame) -> pd.DataFrame:
    """生成 previous_growth 基线预测表。

    公式：`pred_target_growth = growth_lag_1`，即直接沿用上一期增长率预测
    下一期增长率。输入样本必须包含 `growth_lag_1`、当前份额 `share_t`
    以及标准预测契约所需的 split、属性和目标列。

    返回值是按 `TREND_MODEL_PREDICTION_COLUMNS` 排列的预测表，其中
    `pred_share_t1` 由预测增长率和当前份额推导并在同一 split-week-attr_type
    分组内归一化。
    """

    missing_columns = sorted(
        set(PREVIOUS_GROWTH_REQUIRED_COLUMNS) - set(split_samples.columns)
    )
    if missing_columns:
        raise ValueError(
            "previous_growth 模型输入样本缺少必需列: "
            + ", ".join(missing_columns)
        )
    validate_required_columns(
        split_samples,
        PREVIOUS_GROWTH_REQUIRED_COLUMNS,
        source_name="previous_growth 模型输入样本",
    )
    if not set(split_samples["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("previous_growth 模型输入样本存在非法 split。")

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
    predictions.insert(4, "model_name", PREVIOUS_GROWTH_MODEL_NAME)
    predictions["pred_target_growth"] = predictions["growth_lag_1"]
    epsilon = float(PREVIOUS_GROWTH_PARAMS["epsilon"])
    predictions["pred_share_t1"] = derive_normalized_pred_share_t1(
        predictions,
        epsilon,
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return predictions


class PreviousGrowthTrainer:
    """previous_growth 基线训练器，为通用 runner 产出标准 TrendTrainResult。"""

    name = PREVIOUS_GROWTH_MODEL_NAME
    model_type = MODEL_TYPE_BASELINE

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        split_samples = pd.concat(
            [context.split_frames[split_name] for split_name in context.split_order],
            ignore_index=True,
        )
        predictions = predict_previous_growth(split_samples)
        validate_trend_model_predictions(predictions, split_samples)
        return TrendTrainResult(
            model_name=self.name,
            model_type=self.model_type,
            predictions=predictions,
            params=dict(PREVIOUS_GROWTH_PARAMS),
        )
