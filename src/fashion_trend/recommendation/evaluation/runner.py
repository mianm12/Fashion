from __future__ import annotations

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.contracts import RECOMMENDATION_TOP_K
from fashion_trend.recommendation.evaluation.metrics import evaluate_recommendations
from fashion_trend.recommendation.evaluation.payloads import (
    build_recommendation_metrics_payload,
)
from fashion_trend.recommendation.features.cache import build_recommendable_pool
from fashion_trend.recommendation.paths import method_output_paths


def run_recommendation_evaluation(
    method: str,
    recommendations: pd.DataFrame,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
    recommendable_pool: pd.DataFrame,
    input_paths: dict[str, str],
    strict_missing_users: bool = False,
) -> dict[str, object]:
    output_paths = method_output_paths(method)
    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        RECOMMENDATION_TOP_K,
        strict_missing_users=strict_missing_users,
    )
    payload = build_recommendation_metrics_payload(method, metrics, input_paths)
    write_json_atomic(payload, output_paths.metrics)
    return payload


def build_recommendable_pool_for_windows(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    return build_recommendable_pool(transactions, windows)


def input_paths_for_method(method: str) -> dict[str, str]:
    output_paths = method_output_paths(method)
    return {"recommendations": str(output_paths.recommendations)}
