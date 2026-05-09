from __future__ import annotations

from dataclasses import dataclass, field

from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.methods.baselines.global_popularity import (
    build_baseline_recommendation_result,
)


@dataclass(frozen=True)
class PopSimilarityMethod:
    name: str = "pop_similarity"
    method_type: str = "baseline"
    default_candidate_strategy: str = "default"
    default_weights: dict[str, float] = field(
        default_factory=lambda: {
            "pop_score": 0.45,
            "sim_score": 0.45,
            "recent_score": 0.10,
        }
    )
    required_features: tuple[str, ...] = ("pop_score", "sim_score", "recent_score")

    def build_recommendations(
        self,
        context: RecommendationContext,
    ) -> RecommendationResult:
        if context.candidates is None:
            raise FileNotFoundError("pop_similarity requires default candidates")
        return build_baseline_recommendation_result(
            self,
            context,
            context.candidates,
        )
