from __future__ import annotations

from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.recommendation.methods.base import RecommendationMethod
from fashion_trend.recommendation.methods.baselines import (
    AttributeSimilarityMethod,
    GlobalPopularityMethod,
    PopSimilarityMethod,
    RecentPopularityMethod,
)

RECOMMENDATION_METHOD_REGISTRY: dict[str, RecommendationMethod] = {
    "global_popularity": GlobalPopularityMethod(),
    "recent_popularity": RecentPopularityMethod(),
    "attribute_similarity": AttributeSimilarityMethod(),
    "pop_similarity": PopSimilarityMethod(),
}


def get_recommendation_method(name: str) -> RecommendationMethod:
    validate_safe_path_segment(name, "recommendation method")
    try:
        return RECOMMENDATION_METHOD_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(RECOMMENDATION_METHOD_REGISTRY))
        raise ValueError(f"未知推荐 method: {name}. 可用 method: {choices}") from exc
