from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fashion_trend.config import OUTPUT_METRICS_DIR, OUTPUT_MODELS_DIR
from fashion_trend.trend import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_key,
)

TREND_EVALUATION_SPLITS: tuple[str, ...] = ("valid", "test")
TREND_EVALUATION_K_VALUES: tuple[int, ...] = (5, 10, 20)
TREND_EVALUATION_GROUP_COLUMNS: tuple[str, ...] = ("split", "week_id", "attr_type")
TREND_EVALUATION_TARGET_COLUMN = "target_growth"
TREND_EVALUATION_PREDICTION_COLUMN = "pred_target_growth"

_TEXT_PREDICTION_DTYPES: dict[str, str] = {
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "model_name": "string",
    "split": "string",
}


def derive_trend_metric_output_paths(
    model_name: str,
    model_output_root: Path = OUTPUT_MODELS_DIR,
    metrics_output_root: Path = OUTPUT_METRICS_DIR,
) -> dict[str, Path]:
    """根据模型名推导预测输入路径和趋势评价输出路径。"""
    output_dir = metrics_output_root / model_name
    return {
        "output_dir": output_dir,
        "predictions": model_output_root / model_name / "predictions.csv",
        "metrics": output_dir / "trend_metrics.json",
    }


def read_trend_model_predictions(prediction_path: Path) -> pd.DataFrame:
    """读取趋势模型预测 CSV，并保留标准列契约。"""
    if not prediction_path.exists():
        raise FileNotFoundError(f"趋势模型预测文件不存在: {prediction_path}")
    predictions = pd.read_csv(prediction_path, dtype=_TEXT_PREDICTION_DTYPES)
    if predictions.columns.tolist() != list(TREND_MODEL_PREDICTION_COLUMNS):
        raise ValueError(f"趋势模型预测文件列必须与契约完全一致: {prediction_path}")
    return predictions.copy()


def validate_trend_model_predictions_for_evaluation(
    predictions: pd.DataFrame,
    model_name: str,
) -> None:
    """在计算趋势评价指标前校验预测表。"""
    if predictions.columns.tolist() != list(TREND_MODEL_PREDICTION_COLUMNS):
        raise ValueError("趋势模型评价预测表列必须与契约完全一致。")
    validate_required_columns(
        predictions.columns.tolist(),
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name="趋势模型评价预测表",
    )
    numeric_columns = [
        "week_id",
        "share_t",
        "pred_share_t1",
        "target_growth",
        "pred_target_growth",
        "target_rank_in_type_t1",
    ]
    numeric_values = predictions.loc[:, numeric_columns]
    try:
        finite_numeric_values = numeric_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势模型评价预测表无法校验数值字段。") from exc
    if not np.isfinite(finite_numeric_values).all():
        raise ValueError("趋势模型评价预测表存在非有限数值。")

    validate_no_missing_values(
        predictions,
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name="趋势模型评价预测表",
    )
    validate_unique_key(
        predictions,
        ["week_id", "attr_id", "model_name"],
        source_name="趋势模型评价预测表",
    )
    split_values = set(predictions["split"].astype(str))
    if not split_values.issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("趋势模型评价预测表存在非法 split。")
    missing_eval_splits = set(TREND_EVALUATION_SPLITS) - split_values
    if missing_eval_splits:
        raise ValueError(f"趋势模型评价预测表缺少评价 split: {sorted(missing_eval_splits)}")

    model_values = set(predictions["model_name"].astype(str))
    if model_values != {model_name}:
        raise ValueError(
            "趋势模型评价预测表 model_name 与请求不一致: "
            f"expected={model_name}, actual={sorted(model_values)}"
        )

    _validate_integer_week_ids(predictions["week_id"], "趋势模型评价预测表")


def _validate_integer_week_ids(week_ids: pd.Series, source_name: str) -> pd.Series:
    try:
        numeric_week_ids = pd.to_numeric(week_ids, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} week_id 必须为整数。") from exc
    if numeric_week_ids.isna().any() or not (numeric_week_ids % 1 == 0).all():
        raise ValueError(f"{source_name} week_id 必须为整数。")
    return numeric_week_ids.astype("int64")
