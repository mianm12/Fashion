from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.contracts import RECOMMENDATION_TOP_K
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.outputs import write_recommendation_result
from fashion_trend.recommendation.registry import get_recommendation_method


def run_recommendation_method(
    method_name: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    user_profile: pd.DataFrame | None = None,
    trend_predictions: pd.DataFrame | None = None,
    exclude_seen: bool = True,
    weights: dict[str, float] | None = None,
) -> RecommendationResult:
    method = get_recommendation_method(method_name)
    context = RecommendationContext(
        method=method_name,
        top_k=RECOMMENDATION_TOP_K,
        exclude_seen=exclude_seen,
        transactions=transactions,
        article_attributes=article_attributes,
        windows=windows,
        target_users=target_users,
        candidates=candidates,
        user_profile=user_profile,
        trend_predictions=trend_predictions,
        weights=weights,
    )
    result = method.build_recommendations(context)
    write_recommendation_result(result)
    return result
