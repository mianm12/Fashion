from __future__ import annotations

from dataclasses import dataclass, field

from fashion_trend.recommendation.contracts import RECOMMENDATION_CORE_ATTR_TYPES
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.methods.baselines.global_popularity import (
    build_baseline_recommendation_result,
)


@dataclass(frozen=True)
class PopSimilarityTrendMethod:
    name: str = "pop_similarity_trend"
    method_type: str = "trend_aware"
    default_candidate_strategy: str = "default"
    default_weights: dict[str, float] = field(
        default_factory=lambda: {
            "pop_score": 0.35,
            "sim_score": 0.35,
            "trend_score": 0.25,
            "recent_score": 0.05,
        }
    )
    required_features: tuple[str, ...] = (
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
    )

    def build_recommendations(
        self,
        context: RecommendationContext,
    ) -> RecommendationResult:
        if context.trend_predictions is None or context.trend_predictions.empty:
            raise FileNotFoundError("pop_similarity_trend requires trend predictions")
        if context.candidates is None:
            raise FileNotFoundError("pop_similarity_trend requires default candidates")
        return build_baseline_recommendation_result(
            self,
            context,
            context.candidates,
            metadata={
                "trend_score_source": "pred_target_growth",
                "trend_score_attribute_types": tuple(RECOMMENDATION_CORE_ATTR_TYPES),
            },
        )
