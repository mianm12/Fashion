from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPORT_TREND_JOIN_KEY = ("week_id", "attr_id", "attr_type", "attr_value")
REPORT_TREND_SAMPLE_COLUMNS = (
    *REPORT_TREND_JOIN_KEY,
    "heat_t",
    "share_t",
    "target_growth",
    "history_total_heat_t",
    "history_active_weeks_t",
    "is_trend_eligible_t",
)
REPORT_TREND_VIEW_COLUMNS = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "model_name",
    "split",
    "share_t",
    "pred_share_t1",
    "target_growth",
    "pred_target_growth",
    "target_rank_in_type_t1",
    "heat_t",
    "history_total_heat_t",
    "history_active_weeks_t",
    "is_trend_eligible_t",
)


def read_json_object(path: Path, *, artifact_name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{artifact_name} 文件不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 {artifact_name}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} 必须是 JSON object: {path}")
    return payload


def read_feature_importance(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = {
        "feature",
        "split_importance",
        "gain_importance",
        "normalized_gain_importance",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"LightGBM feature_importance 缺少列: {missing}")
    for column in (
        "split_importance",
        "gain_importance",
        "normalized_gain_importance",
    ):
        dataframe[column] = _non_negative_finite_series(
            dataframe[column],
            artifact_name="LightGBM feature_importance",
            column=column,
        )
    return dataframe


def read_trend_samples(path: Path) -> pd.DataFrame:
    try:
        dataframe = pd.read_parquet(path, columns=list(REPORT_TREND_SAMPLE_COLUMNS))
    except Exception as exc:
        try:
            dataframe = pd.read_parquet(path)
        except Exception:
            raise exc
        _raise_missing_trend_sample_columns(dataframe)
        raise exc
    _raise_missing_trend_sample_columns(dataframe)
    return dataframe


def _raise_missing_trend_sample_columns(dataframe: pd.DataFrame) -> None:
    missing = sorted(set(REPORT_TREND_SAMPLE_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(f"trend_model_samples 缺少列: {missing}")


def build_lightgbm_prediction_sample_view(
    predictions: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    _reject_duplicate_join_key(predictions, artifact_name="LightGBM predictions")
    _reject_duplicate_join_key(samples, artifact_name="trend_model_samples")
    _validate_join_key_sets_match(predictions, samples)

    joined = predictions.merge(
        samples.loc[:, list(REPORT_TREND_SAMPLE_COLUMNS)],
        on=list(REPORT_TREND_JOIN_KEY),
        how="left",
        suffixes=("", "_sample"),
        validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all() or len(joined) != len(predictions):
        sample = joined.loc[joined["_merge"] != "both", list(REPORT_TREND_JOIN_KEY)]
        raise ValueError(
            "LightGBM predictions 与 trend_model_samples 无法 1:1 join: "
            f"{sample.head(3).to_dict('records')}"
        )

    _validate_joined_numeric_consistency(joined, column="share_t")
    _validate_joined_numeric_consistency(joined, column="target_growth")
    return joined.loc[:, list(REPORT_TREND_VIEW_COLUMNS)].copy()


def flatten_trend_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    model_name = _required_text(payload, "model_name")
    run_id = payload.get("run_id")
    rows: list[dict[str, Any]] = []
    for split, metrics in sorted(_required_dict(payload, "overall").items()):
        metric_payload = _as_dict(metrics, f"overall.{split}")
        rows.append(
            {
                "model_name": model_name,
                "split": split,
                "mae": _finite_number(metric_payload, "mae"),
                "rmse": _finite_number(metric_payload, "rmse"),
                "spearman": _finite_number(metric_payload, "spearman"),
                "ndcg_at_10": _metric_at_k(metric_payload, "ndcg_at_k", "10"),
                "precision_at_10": _metric_at_k(metric_payload, "precision_at_k", "10"),
                "recall_at_10": _metric_at_k(metric_payload, "recall_at_k", "10"),
                "run_id": "" if run_id is None else str(run_id),
            }
        )
    return rows


def flatten_trend_metrics_by_attr_type(payload: dict[str, Any]) -> list[dict[str, Any]]:
    model_name = _required_text(payload, "model_name")
    rows: list[dict[str, Any]] = []
    for split, attr_type_metrics in sorted(
        _required_dict(payload, "by_attr_type").items()
    ):
        attr_payload = _as_dict(attr_type_metrics, f"by_attr_type.{split}")
        for attr_type, metrics in sorted(attr_payload.items()):
            metric_payload = _as_dict(metrics, f"by_attr_type.{split}.{attr_type}")
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "attr_type": attr_type,
                    "mae": _finite_number(metric_payload, "mae"),
                    "rmse": _finite_number(metric_payload, "rmse"),
                    "spearman": _optional_finite_number(metric_payload, "spearman"),
                    "ndcg_at_10": _metric_at_k(metric_payload, "ndcg_at_k", "10"),
                    "precision_at_10": _metric_at_k(
                        metric_payload,
                        "precision_at_k",
                        "10",
                    ),
                    "recall_at_10": _metric_at_k(metric_payload, "recall_at_k", "10"),
                }
            )
    return rows


def flatten_recommendation_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    method = _required_text(payload, "method")
    rows: list[dict[str, Any]] = []
    for split, metrics in sorted(_required_dict(payload, "metrics").items()):
        metric_payload = _as_dict(metrics, f"metrics.{split}")
        rows.append(
            {
                "method": method,
                "split": split,
                "map_at_12": _finite_number(metric_payload, "map_at_12"),
                "recall_at_12": _finite_number(metric_payload, "recall_at_12"),
                "hit_rate_at_12": _finite_number(metric_payload, "hit_rate_at_12"),
                "ndcg_at_12": _finite_number(metric_payload, "ndcg_at_12"),
                "coverage": _finite_number(metric_payload, "coverage"),
                "user_count": _non_negative_integer(metric_payload, "user_count"),
                "missing_recommendation_user_count": _non_negative_integer(
                    metric_payload,
                    "missing_recommendation_user_count",
                ),
            }
        )
    return rows


def _reject_duplicate_join_key(dataframe: pd.DataFrame, *, artifact_name: str) -> None:
    duplicated = dataframe.duplicated(list(REPORT_TREND_JOIN_KEY), keep=False)
    if duplicated.any():
        sample = dataframe.loc[duplicated, list(REPORT_TREND_JOIN_KEY)].head(3)
        raise ValueError(
            f"{artifact_name} 存在重复 join key: {sample.to_dict('records')}"
        )


def _validate_join_key_sets_match(
    predictions: pd.DataFrame,
    samples: pd.DataFrame,
) -> None:
    prediction_keys = predictions.loc[:, list(REPORT_TREND_JOIN_KEY)].drop_duplicates()
    sample_keys = samples.loc[:, list(REPORT_TREND_JOIN_KEY)].drop_duplicates()
    joined_keys = prediction_keys.merge(
        sample_keys,
        on=list(REPORT_TREND_JOIN_KEY),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = joined_keys.loc[joined_keys["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            "LightGBM predictions 与 trend_model_samples 无法 1:1 join: "
            f"{unmatched.head(3).to_dict('records')}"
        )


def _validate_joined_numeric_consistency(
    dataframe: pd.DataFrame, *, column: str
) -> None:
    left = pd.to_numeric(dataframe[column], errors="raise")
    right = pd.to_numeric(dataframe[f"{column}_sample"], errors="raise")
    if not np.allclose(
        left.to_numpy(dtype=float),
        right.to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(f"LightGBM predictions 与 samples 的 {column} 不一致。")


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"metrics payload 缺少字段: {key}")
    return _as_dict(payload[key], key)


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 JSON object。")
    return value


def _required_text(payload: dict[str, Any], key: str) -> str:
    if key not in payload or payload[key] is None:
        raise ValueError(f"metrics payload 缺少字段: {key}")
    value = str(payload[key])
    if not value:
        raise ValueError(f"metrics payload 字段不能为空: {key}")
    return value


def _finite_number(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"metrics payload 缺少字段: {key}")
    if isinstance(payload[key], bool):
        raise ValueError(f"metrics payload 字段不是有限数值: {key}")
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metrics payload 字段不是有限数值: {key}") from exc
    if not np.isfinite(value):
        raise ValueError(f"metrics payload 字段不是有限数值: {key}")
    return value


def _optional_finite_number(payload: dict[str, Any], key: str) -> float | str:
    if key not in payload:
        raise ValueError(f"metrics payload 缺少字段: {key}")
    if payload[key] is None:
        return ""
    return _finite_number(payload, key)


def _non_negative_integer(payload: dict[str, Any], key: str) -> int:
    if key in payload and isinstance(payload[key], bool):
        raise ValueError(f"metrics payload 字段必须是非负整数: {key}")
    value = _finite_number(payload, key)
    if value < 0 or not value.is_integer():
        raise ValueError(f"metrics payload 字段必须是非负整数: {key}")
    return int(value)


def _metric_at_k(payload: dict[str, Any], metric_name: str, k: str) -> float:
    metrics = _required_dict(payload, metric_name)
    return _finite_number(metrics, k)


def _non_negative_finite_series(
    series: pd.Series,
    *,
    artifact_name: str,
    column: str,
) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{artifact_name} 字段不可解析为数值: {column}") from exc
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{artifact_name} 字段不是有限数值: {column}")
    if (values < 0).any():
        raise ValueError(f"{artifact_name} 字段不能为负数: {column}")
    return values
