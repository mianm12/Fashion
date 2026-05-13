from __future__ import annotations

import math
from typing import Any

SCORE_FEATURES = ("pop_score", "sim_score", "trend_score", "recent_score")
STRICT_VARIANTS = (
    ("without_trend_in_rec", "w/o Trend in Rec", "trend_score"),
    ("without_similarity", "w/o Similarity", "sim_score"),
    ("without_recent", "w/o Recent", "recent_score"),
)


def build_ablation_summary(
    metrics_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in metrics_payloads:
        method = str(payload["method"])
        metrics_by_split = dict(payload["metrics"])
        for split, metrics in metrics_by_split.items():
            rows.append({"method": method, "split": split, **dict(metrics)})
    return sorted(rows, key=lambda row: (row["split"], row["method"]))


def derive_strict_ablation_weights(
    best_weights: dict[str, float],
    dropped_feature: str,
) -> dict[str, float]:
    if dropped_feature not in SCORE_FEATURES:
        raise ValueError(f"未知 strict ablation feature: {dropped_feature}")

    weights = _read_weights(best_weights, context="best_weights")
    remaining_total = sum(
        value for feature, value in weights.items() if feature != dropped_feature
    )
    if remaining_total <= 0.0 or not math.isfinite(remaining_total):
        raise ValueError("best_weights 删除目标组件后无法归一化。")

    derived = {
        feature: (
            0.0 if feature == dropped_feature else weights[feature] / remaining_total
        )
        for feature in SCORE_FEATURES
    }
    _validate_weight_sum(derived, context=f"strict ablation {dropped_feature}")
    return derived


def build_named_ablation_rows(
    *,
    best_weights: dict[str, float],
    strict_metrics: dict[str, dict[str, dict[str, float]]],
    full_model_metrics: dict[str, dict[str, float]],
    stable_baseline_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    weights = _read_weights(best_weights, context="best_weights")
    rows: list[dict[str, Any]] = [
        {
            "variant_id": "full_model",
            "display_name": "Full Model",
            "method": "pop_similarity_trend",
            "base_method": "pop_similarity_trend",
            "candidate_strategy": "default",
            "weight_policy": "stable_full_model",
            "selection_split": "valid",
            "metrics_source": "stable_method_output",
            "weights": weights,
            "metrics": _read_metrics_by_split(full_model_metrics, "full_model"),
        }
    ]

    for variant_id, display_name, dropped_feature in STRICT_VARIANTS:
        if variant_id not in strict_metrics:
            raise ValueError(f"strict_metrics 缺少 {variant_id}")
        rows.append(
            {
                "variant_id": variant_id,
                "display_name": display_name,
                "method": "pop_similarity_trend",
                "base_method": "pop_similarity_trend",
                "candidate_strategy": "default",
                "weight_policy": "strict_drop_and_renormalize_from_full",
                "selection_split": "valid",
                "metrics_source": "in_memory_evaluation",
                "weights": derive_strict_ablation_weights(weights, dropped_feature),
                "metrics": _read_metrics_by_split(
                    strict_metrics[variant_id],
                    variant_id,
                ),
            }
        )

    for variant_id in ("recent_only_baseline", "pop_similarity_baseline"):
        if variant_id not in stable_baseline_metrics:
            raise ValueError(f"stable_baseline_metrics 缺少 {variant_id}")
        baseline = stable_baseline_metrics[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "display_name": str(baseline["display_name"]),
                "method": str(baseline["method"]),
                "base_method": str(baseline["method"]),
                "candidate_strategy": "not_applicable",
                "weight_policy": "stable_method_baseline",
                "selection_split": "not_applicable",
                "metrics_source": "stable_method_output",
                "weights": {},
                "metrics": _read_metrics_by_split(
                    dict(baseline["metrics"]),
                    variant_id,
                ),
            }
        )

    return rows


def select_trend_bucket_representatives(
    search_results: list[dict[str, Any]],
    *,
    required_trend_scores: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4),
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for trend_score in required_trend_scores:
        bucket = [
            result
            for result in search_results
            if math.isclose(
                float(dict(result["weights"])["trend_score"]),
                trend_score,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        if not bucket:
            raise ValueError(f"trend bucket 缺少代表组合: {trend_score}")

        best = min(
            bucket,
            key=lambda result: (
                -float(dict(result["valid_metrics"])["ndcg_at_12"]),
                int(result["grid_index"]),
            ),
        )
        weights = _read_weights(
            dict(best["weights"]),
            context="trend bucket weights",
        )
        selected.append(
            {
                "variant_id": _trend_bucket_variant_id(trend_score),
                "display_name": f"trend_score={trend_score:g} valid-best",
                "trend_score": trend_score,
                "grid_index": int(best["grid_index"]),
                "base_method": "pop_similarity_trend",
                "candidate_strategy": "default",
                "weight_policy": "trend_bucket_best_by_valid_ndcg_at_12",
                "selection_split": "valid",
                "metrics_source": "in_memory_evaluation",
                "weights": weights,
                "metrics": {
                    "valid": {
                        str(metric_name): _finite_number(
                            metric_value,
                            f"trend bucket {trend_score}.valid",
                        )
                        for metric_name, metric_value in dict(
                            best["valid_metrics"]
                        ).items()
                    }
                },
            }
        )
    return selected


def _read_weights(weights: dict[str, float], *, context: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for feature in SCORE_FEATURES:
        raw_value = weights.get(feature, 0.0)
        value = float(raw_value)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{context} 包含非法权重: {feature}={raw_value!r}")
        result[feature] = value
    _validate_weight_sum(result, context=context)
    return result


def _validate_weight_sum(weights: dict[str, float], *, context: str) -> None:
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{context} 权重和必须为 1.0，当前为 {total}")


def _read_metrics_by_split(
    metrics_by_split: dict[str, dict[str, float]],
    context: str,
) -> dict[str, dict[str, float]]:
    if not metrics_by_split:
        raise ValueError(f"{context} metrics 为空")

    result: dict[str, dict[str, float]] = {}
    for split, metrics in metrics_by_split.items():
        if not metrics:
            raise ValueError(f"{context} split={split} metrics 为空")
        result[str(split)] = {
            str(metric_name): _finite_number(metric_value, f"{context}.{split}")
            for metric_name, metric_value in dict(metrics).items()
        }
    return result


def _finite_number(value: Any, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} 包含非有限指标值: {value!r}")
    return number


def _trend_bucket_variant_id(trend_score: float) -> str:
    return f"trend_bucket_{str(trend_score).replace('.', '_')}"
