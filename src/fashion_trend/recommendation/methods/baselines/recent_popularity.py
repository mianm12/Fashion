from __future__ import annotations

from dataclasses import dataclass, field

from fashion_trend.recommendation.contracts import RECOMMENDATION_CANDIDATES_PER_SOURCE
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.methods.baselines.global_popularity import (
    build_baseline_recommendation_result,
)
from fashion_trend.recommendation.retrieval.candidates import build_candidate_items
from fashion_trend.recommendation.retrieval.popularity import (
    build_recent_popularity_candidates,
)


@dataclass(frozen=True)
class RecentPopularityMethod:
    name: str = "recent_popularity"
    method_type: str = "baseline"
    default_candidate_strategy: str | None = None
    default_weights: dict[str, float] = field(
        default_factory=lambda: {"recent_score": 1.0}
    )
    required_features: tuple[str, ...] = ("recent_score",)

    def build_recommendations(
        self,
        context: RecommendationContext,
    ) -> RecommendationResult:
        source = build_recent_popularity_candidates(
            context.transactions,
            context.windows,
            context.target_users,
            top_n=max(context.top_k, RECOMMENDATION_CANDIDATES_PER_SOURCE),
        )
        candidates = build_candidate_items("popularity", [source])
        return build_baseline_recommendation_result(self, context, candidates)
