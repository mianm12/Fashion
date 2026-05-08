from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fashion_trend.foundation.artifacts import (
    validate_output_parent_dirs,
    validate_safe_path_segment,
)
from fashion_trend.foundation.dataframe import (
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_key,
)
from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.trend.evaluation.metrics import (
    TREND_EVALUATION_GROUP_COLUMNS,
    TREND_EVALUATION_K_VALUES,
    TREND_EVALUATION_PREDICTION_COLUMN,
    TREND_EVALUATION_SPLITS,
    TREND_EVALUATION_TARGET_COLUMN,
    _validate_k_values,
    compute_trend_metrics,
)
from fashion_trend.trend.paths import OUTPUT_METRICS_DIR, OUTPUT_MODELS_DIR
from fashion_trend.trend.predictions import validate_pred_share_t1_distribution
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)

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
    validate_safe_path_segment(model_name, "model_name")
    prediction_dir = model_output_root / model_name
    output_dir = metrics_output_root / model_name
    validate_output_parent_dirs(prediction_dir, model_output_root)
    validate_output_parent_dirs(output_dir, metrics_output_root)
    return {
        "output_dir": output_dir,
        "predictions": prediction_dir / "predictions.csv",
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
        predictions,
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
    validate_pred_share_t1_distribution(predictions, "趋势模型评价预测表")

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
        raise ValueError(
            f"趋势模型评价预测表缺少评价 split: {sorted(missing_eval_splits)}"
        )

    model_values = set(predictions["model_name"].astype(str))
    if model_values != {model_name}:
        raise ValueError(
            "趋势模型评价预测表 model_name 与请求不一致: "
            f"expected={model_name}, actual={sorted(model_values)}"
        )

    _validate_integer_week_ids(predictions["week_id"], "趋势模型评价预测表")


def build_trend_metrics_payload(
    predictions: pd.DataFrame,
    model_name: str,
    prediction_path: Path,
    output_path: Path,
    k_values: Sequence[int] = TREND_EVALUATION_K_VALUES,
) -> dict[str, object]:
    """构建 trend_metrics.json 的内存结构，不写入文件。"""
    _validate_k_values(k_values)
    validate_trend_model_predictions_for_evaluation(predictions, model_name)
    metrics = compute_trend_metrics(predictions, k_values)
    payload: dict[str, object] = {
        "model_name": model_name,
        "prediction_path": str(prediction_path),
        "output_path": str(output_path),
        "evaluated_splits": list(TREND_EVALUATION_SPLITS),
        "ranking": {
            "target_column": TREND_EVALUATION_TARGET_COLUMN,
            "prediction_column": TREND_EVALUATION_PREDICTION_COLUMN,
            "group_by": list(TREND_EVALUATION_GROUP_COLUMNS),
            "k_values": [int(k) for k in k_values],
        },
        "overall": metrics["overall"],
        "by_attr_type": metrics["by_attr_type"],
        "groups": metrics["groups"],
    }
    _validate_json_payload(payload)
    return payload


def write_trend_metrics(payload: dict[str, object], output_path: Path) -> None:
    """确认 payload 是严格 JSON 后写出趋势评价指标。"""
    _validate_json_payload(payload)
    write_json_atomic(payload, output_path)


def _validate_integer_week_ids(week_ids: pd.Series, source_name: str) -> pd.Series:
    try:
        numeric_week_ids = pd.to_numeric(week_ids, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} week_id 必须为整数。") from exc
    if numeric_week_ids.isna().any() or not (numeric_week_ids % 1 == 0).all():
        raise ValueError(f"{source_name} week_id 必须为整数。")
    return numeric_week_ids.astype("int64")


def _validate_json_payload(payload: dict[str, object]) -> None:
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
