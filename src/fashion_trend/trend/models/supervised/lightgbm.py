from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.trend.models.base import MODEL_TYPE_SUPERVISED

LIGHTGBM_MODEL_NAME = "lightgbm"
LIGHTGBM_TARGET_COLUMN = "target_growth"
LIGHTGBM_EPSILON = 1e-6
LIGHTGBM_NUMERIC_FEATURES: tuple[str, ...] = (
    "heat_t",
    "share_t",
    "log_heat_t",
    "rank_in_type_t",
    "heat_lag_1",
    "heat_lag_2",
    "heat_lag_3",
    "heat_lag_4",
    "share_lag_1",
    "share_lag_2",
    "share_lag_3",
    "share_lag_4",
    "growth_lag_1",
    "growth_lag_2",
    "acc_lag_1",
    "heat_ma_4",
    "share_ma_4",
    "share_std_4",
    "share_max_4",
    "share_min_4",
    "article_count",
    "is_core_attr",
    "parent_count",
    "child_count",
    "degree",
    "history_total_heat_t",
    "history_active_weeks_t",
    "is_trend_eligible_t",
    "week_index",
    "week_mod_52",
)
LIGHTGBM_CATEGORICAL_FEATURES: tuple[str, ...] = ("attr_type",)
LIGHTGBM_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "attr_id",
    "attr_value",
    "target_growth",
    "target_log_heat_t1",
    "target_rank_in_type_t1",
    "split",
)
LIGHTGBM_ALLOWED_OBJECTIVES: tuple[str, ...] = ("regression", "regression_l1")
LIGHTGBM_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}
LIGHTGBM_EARLY_STOPPING: dict[str, int] = {"stopping_rounds": 30}
_LIGHTGBM_REQUIRED_COLUMNS: tuple[str, ...] = (
    "split",
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "share_t",
    "target_growth",
    "target_rank_in_type_t1",
    *LIGHTGBM_NUMERIC_FEATURES,
)


@dataclass(frozen=True)
class LightGBMFeatureFrame:
    features: pd.DataFrame
    attr_type_categories: tuple[str, ...]


class LightGBMTrendTrainer:
    name = LIGHTGBM_MODEL_NAME
    model_type = MODEL_TYPE_SUPERVISED

    def train(self, context):
        raise ValueError("lightgbm 模型训练流程缺少拟合实现。")


def prepare_lightgbm_feature_frame(
    samples: pd.DataFrame,
    attr_type_categories: tuple[str, ...] | None = None,
) -> LightGBMFeatureFrame:
    if samples.empty:
        raise ValueError("lightgbm 模型输入 split 不能为空。")
    validate_required_columns(
        samples,
        _LIGHTGBM_REQUIRED_COLUMNS,
        source_name="lightgbm 模型输入样本",
    )
    numeric_features = _read_numeric_features(samples)
    attr_type = _read_attr_type(samples, attr_type_categories)
    features = pd.concat([numeric_features, attr_type], axis=1)
    return LightGBMFeatureFrame(
        features=features.loc[
            :, [*LIGHTGBM_NUMERIC_FEATURES, *LIGHTGBM_CATEGORICAL_FEATURES]
        ],
        attr_type_categories=tuple(attr_type.cat.categories.astype(str)),
    )


def _read_numeric_features(samples: pd.DataFrame) -> pd.DataFrame:
    numeric_features = pd.DataFrame(index=samples.index)
    for column in LIGHTGBM_NUMERIC_FEATURES:
        try:
            numeric_features[column] = pd.to_numeric(samples[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"lightgbm 模型数值特征无法解析: {column}") from exc
        if not np.isfinite(numeric_features[column].to_numpy(dtype=float)).all():
            raise ValueError(f"lightgbm 模型数值特征存在非有限值: {column}")
    return numeric_features


def _read_attr_type(
    samples: pd.DataFrame,
    attr_type_categories: tuple[str, ...] | None,
) -> pd.Series:
    if samples["attr_type"].isna().any():
        raise ValueError("lightgbm 模型 attr_type 不能为空。")
    attr_values = samples["attr_type"].astype(str)
    if attr_type_categories is None:
        categories = tuple(sorted(attr_values.unique()))
    else:
        categories = tuple(attr_type_categories)
        unknown_values = sorted(set(attr_values.unique()) - set(categories))
        if unknown_values:
            examples = ", ".join(unknown_values[:5])
            raise ValueError(f"lightgbm 模型存在未知 attr_type: {examples}")
    dtype = CategoricalDtype(categories=list(categories))
    attr_type = attr_values.astype(dtype)
    attr_type.name = "attr_type"
    return attr_type


def describe_target_distribution(
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float | int]]:
    return {
        split_name: _describe_target_series(split_frame[LIGHTGBM_TARGET_COLUMN])
        for split_name, split_frame in split_frames.items()
    }


def describe_zero_share_rows(
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, int]:
    return {
        split_name: int(
            pd.to_numeric(split_frame["share_t"], errors="raise").eq(0).sum()
        )
        for split_name, split_frame in split_frames.items()
    }


def describe_residual_distribution(
    prediction_frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float | int]]:
    distribution: dict[str, dict[str, float | int]] = {}
    for split_name, predictions in prediction_frames.items():
        residual = (
            pd.to_numeric(predictions["target_growth"], errors="raise")
            - pd.to_numeric(predictions["pred_target_growth"], errors="raise")
        )
        summary = _describe_residual_series(residual)
        summary["mae"] = float(residual.abs().mean())
        summary["rmse"] = float(np.sqrt(np.square(residual).mean()))
        distribution[split_name] = summary
    return distribution


def build_feature_importance_frame(booster) -> pd.DataFrame:
    feature_names = list(booster.feature_name())
    split_importance = np.asarray(
        booster.feature_importance(importance_type="split"),
        dtype=np.int64,
    )
    gain_importance = np.asarray(
        booster.feature_importance(importance_type="gain"),
        dtype=float,
    )
    total_gain = float(gain_importance.sum())
    if total_gain > 0:
        normalized_gain = gain_importance / total_gain
    else:
        normalized_gain = np.zeros_like(gain_importance, dtype=float)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "split_importance": split_importance,
            "gain_importance": gain_importance,
            "normalized_gain_importance": normalized_gain,
        }
    )
    if not np.isfinite(
        importance.loc[:, ["gain_importance", "normalized_gain_importance"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("lightgbm 特征重要性存在非有限数值。")
    return importance


def _describe_numeric_series(series: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("lightgbm 诊断数值存在非有限值。")
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "p01": float(numeric.quantile(0.01)),
        "p05": float(numeric.quantile(0.05)),
        "p50": float(numeric.quantile(0.50)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
    }


def _describe_target_series(series: pd.Series) -> dict[str, float | int]:
    summary = _describe_numeric_series(series)
    numeric = pd.to_numeric(series, errors="raise")
    summary["abs_gt_2"] = int(numeric.abs().gt(2).sum())
    return summary


def _describe_residual_series(series: pd.Series) -> dict[str, float | int]:
    return _describe_numeric_series(series)
