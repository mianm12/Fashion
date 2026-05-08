from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from fashion_trend.foundation.dataframe import validate_required_columns

TREND_EVALUATION_SPLITS: tuple[str, ...] = ("valid", "test")
TREND_EVALUATION_K_VALUES: tuple[int, ...] = (5, 10, 20)
TREND_EVALUATION_GROUP_COLUMNS: tuple[str, ...] = ("split", "week_id", "attr_type")
TREND_EVALUATION_TARGET_COLUMN = "target_growth"
TREND_EVALUATION_PREDICTION_COLUMN = "pred_target_growth"


def compute_trend_group_metrics(
    group_predictions: pd.DataFrame,
    k_values: Sequence[int] = TREND_EVALUATION_K_VALUES,
) -> dict[str, object]:
    """计算单个 split-week-attr_type 分组的趋势评价指标。

    单组指标同时覆盖增长率回归误差和 Top-K 排名质量。MAE、RMSE 和 Spearman
    使用 `target_growth` 与 `pred_target_growth`，Precision/Recall/NDCG 使用同一
    分组内的预测增长率排序，并在分组规模小于 K 时使用有效 K。
    """

    if group_predictions.empty:
        raise ValueError("趋势评价分组不能为空。")
    _validate_k_values(k_values)

    target = pd.to_numeric(
        group_predictions[TREND_EVALUATION_TARGET_COLUMN],
        errors="raise",
    ).astype(float)
    prediction = pd.to_numeric(
        group_predictions[TREND_EVALUATION_PREDICTION_COLUMN],
        errors="raise",
    ).astype(float)
    errors = target - prediction
    precision_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float | None] = {}

    metrics: dict[str, object] = {
        "mae": _json_float(np.abs(errors).mean()),
        "rmse": _json_float(math.sqrt(float(np.square(errors).mean()))),
        "spearman": _spearman_or_none(target, prediction),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "ndcg_at_k": ndcg_at_k,
    }
    for k in k_values:
        key = str(k)
        effective_k = min(k, len(group_predictions))
        predicted_top = _top_attr_ids(
            group_predictions,
            TREND_EVALUATION_PREDICTION_COLUMN,
            effective_k,
        )
        actual_top = _top_attr_ids(
            group_predictions,
            TREND_EVALUATION_TARGET_COLUMN,
            effective_k,
        )
        hits = len(set(predicted_top) & set(actual_top))
        precision_at_k[key] = _json_float(hits / effective_k)
        recall_at_k[key] = _json_float(hits / effective_k)
        ndcg_at_k[key] = _ndcg_or_none(group_predictions, effective_k)
    return metrics


def compute_trend_metrics(
    predictions: pd.DataFrame,
    k_values: Sequence[int] = TREND_EVALUATION_K_VALUES,
) -> dict[str, object]:
    """聚合趋势模型预测表的整体、属性类型和分组指标。

    评价只聚合 `valid` 和 `test` 两个切分；`train` 预测可以保留在模型输出
    `predictions.csv` 中，但不会进入趋势评价结果。返回载荷包含按 split 汇总的
    overall、by_attr_type 和 groups，供上层组装 `trend_metrics.json`。
    """

    _validate_k_values(k_values)
    validate_required_columns(
        predictions,
        [
            *TREND_EVALUATION_GROUP_COLUMNS,
            "attr_id",
            TREND_EVALUATION_TARGET_COLUMN,
            TREND_EVALUATION_PREDICTION_COLUMN,
        ],
        source_name="趋势评价预测表",
    )

    split_values = predictions["split"].astype(str)
    evaluated = predictions.loc[split_values.isin(TREND_EVALUATION_SPLITS)].copy()
    group_metric_records = []
    grouped = evaluated.groupby(list(TREND_EVALUATION_GROUP_COLUMNS), sort=True)
    for (split, week_id, attr_type), group_predictions in grouped:
        group_metric_records.append(
            {
                "split": str(split),
                "week_id": int(week_id),
                "attr_type": str(attr_type),
                "metrics": compute_trend_group_metrics(group_predictions, k_values),
            }
        )

    overall: dict[str, object] = {}
    by_attr_type: dict[str, object] = {}
    groups: dict[str, object] = {}
    for split in TREND_EVALUATION_SPLITS:
        split_frame = evaluated.loc[evaluated["split"].astype(str) == split]
        if split_frame.empty:
            continue

        split_records = [
            record for record in group_metric_records if record["split"] == split
        ]
        overall[split] = _summarize_metric_records(
            [record["metrics"] for record in split_records]
        )
        groups[split] = {
            "rows": int(len(split_frame)),
            "weeks": int(split_frame["week_id"].nunique()),
            "attr_types": int(split_frame["attr_type"].nunique()),
            "ranking_groups": int(len(split_records)),
        }

        by_attr_type[split] = {}
        attr_types = sorted(split_frame["attr_type"].astype(str).unique())
        for attr_type in attr_types:
            attr_records = [
                record["metrics"]
                for record in split_records
                if record["attr_type"] == attr_type
            ]
            by_attr_type[split][attr_type] = _summarize_metric_records(attr_records)

    return {
        "overall": overall,
        "by_attr_type": by_attr_type,
        "groups": groups,
    }


def _validate_k_values(k_values: Sequence[int]) -> None:
    """校验评价 K 值是非空正整数序列。"""

    if not k_values:
        raise ValueError("趋势评价 K 值不能为空。")
    for k in k_values:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("趋势评价 K 值必须为正整数。")


def _top_attr_ids(
    group_predictions: pd.DataFrame,
    score_column: str,
    k: int,
) -> list[str]:
    """按分数降序和 attr_id 升序提取热门 Top-K 属性。"""

    ranking_frame = pd.DataFrame(
        {
            "attr_id": group_predictions["attr_id"].astype(str),
            "_score": pd.to_numeric(
                group_predictions[score_column],
                errors="raise",
            ).astype(float),
        }
    )
    ranking_frame = ranking_frame.sort_values(
        ["_score", "attr_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranking_frame.head(k)["attr_id"].tolist()


def _spearman_or_none(target: pd.Series, prediction: pd.Series) -> float | None:
    """计算 Spearman 相关系数，遇到常量排序时回退为 None。"""

    if target.nunique(dropna=False) <= 1 or prediction.nunique(dropna=False) <= 1:
        return None

    target_ranks = target.rank(method="average").to_numpy(dtype=float)
    prediction_ranks = prediction.rank(method="average").to_numpy(dtype=float)
    target_centered = target_ranks - target_ranks.mean()
    prediction_centered = prediction_ranks - prediction_ranks.mean()
    denominator = math.sqrt(float(np.square(target_centered).sum())) * math.sqrt(
        float(np.square(prediction_centered).sum())
    )
    if denominator == 0:
        return None
    correlation = float(np.dot(target_centered, prediction_centered)) / denominator
    return _json_float(correlation)


def _ndcg_or_none(group_predictions: pd.DataFrame, k: int) -> float | None:
    """计算 NDCG，遇到全零相关性或零理想增益时回退为 None。"""

    target = pd.to_numeric(
        group_predictions[TREND_EVALUATION_TARGET_COLUMN],
        errors="raise",
    ).astype(float)
    relevance_frame = pd.DataFrame(
        {
            "attr_id": group_predictions["attr_id"].astype(str),
            "relevance": target - target.min(),
        }
    )
    if (relevance_frame["relevance"] == 0).all():
        return None

    relevance_by_attr_id = dict(
        zip(
            relevance_frame["attr_id"],
            relevance_frame["relevance"].astype(float),
            strict=True,
        )
    )
    predicted_relevance = [
        relevance_by_attr_id[attr_id]
        for attr_id in _top_attr_ids(
            group_predictions,
            TREND_EVALUATION_PREDICTION_COLUMN,
            k,
        )
    ]
    ideal_relevance = (
        relevance_frame.sort_values(
            ["relevance", "attr_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(k)["relevance"]
        .tolist()
    )
    ideal_gain = _discounted_gain(ideal_relevance)
    if ideal_gain == 0:
        return None
    return _json_float(_discounted_gain(predicted_relevance) / ideal_gain)


def _discounted_gain(relevance_values: Iterable[float]) -> float:
    """按排名折扣累加相关性增益。"""

    gain = 0.0
    for index, relevance in enumerate(relevance_values):
        gain += float(relevance) / math.log2(index + 2)
    return _json_float(gain)


def _summarize_metric_records(metric_records: Sequence[object]) -> dict[str, object]:
    """汇总多个分组指标记录，生成可写入 JSON 的平均指标。"""

    records = list(metric_records)
    summary: dict[str, object] = {
        "mae": _mean_or_none(record["mae"] for record in records),
        "rmse": _mean_or_none(record["rmse"] for record in records),
        "spearman": _mean_or_none(record["spearman"] for record in records),
        "precision_at_k": {},
        "recall_at_k": {},
        "ndcg_at_k": {},
    }
    for metric_name in ("precision_at_k", "recall_at_k", "ndcg_at_k"):
        keys = sorted(
            {key for record in records for key in record[metric_name]},
            key=int,
        )
        summary[metric_name] = {
            key: _mean_or_none(record[metric_name][key] for record in records)
            for key in keys
        }
    return summary


def _mean_or_none(values: Iterable[object]) -> float | None:
    """计算忽略 None 的均值，没有有效数值时返回 None。"""

    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return _json_float(sum(numeric_values) / len(numeric_values))


def _json_float(value: object) -> float:
    """将指标值转换为 JSON 可写的有限浮点数。"""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("趋势评价指标存在非有限数值。")
    if number == 0:
        return 0.0
    return number
