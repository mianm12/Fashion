from __future__ import annotations

from dataclasses import dataclass, field

from fashion_trend.recommendation.contracts import (
    RECOMMENDATION_CANDIDATES_PER_SOURCE,
)
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
class AttributeSimilarityMethod:
    name: str = "attribute_similarity"
    method_type: str = "baseline"
    default_candidate_strategy: str = "similarity"
    default_weights: dict[str, float] = field(
        default_factory=lambda: {"sim_score": 1.0}
    )
    required_features: tuple[str, ...] = ("sim_score",)

    def build_recommendations(
        self,
        context: RecommendationContext,
    ) -> RecommendationResult:
        if _needs_recent_fallback(context):
            source = build_recent_popularity_candidates(
                context.transactions,
                context.windows,
                context.target_users,
                top_n=max(context.top_k, RECOMMENDATION_CANDIDATES_PER_SOURCE),
            )
            candidates = build_candidate_items("popularity", [source])
            return build_baseline_recommendation_result(
                _AttributeSimilarityRecentFallbackMethod(),
                context,
                candidates,
                metadata={"fallback_user_count": _target_user_window_count(context)},
                backfill_mode="recent",
            )
        return build_baseline_recommendation_result(
            self,
            context,
            context.candidates,
            metadata={"fallback_user_count": 0},
            backfill_mode="recent",
        )


def _needs_recent_fallback(context: RecommendationContext) -> bool:
    return (
        context.user_profile is None
        or context.user_profile.empty
        or context.candidates is None
        or context.candidates.empty
    )


@dataclass(frozen=True)
class _AttributeSimilarityRecentFallbackMethod:
    name: str = "attribute_similarity"
    method_type: str = "baseline"
    default_candidate_strategy: str = "popularity"
    default_weights: dict[str, float] = field(
        default_factory=lambda: {"recent_score": 1.0}
    )
    required_features: tuple[str, ...] = ("recent_score",)


def _target_user_window_count(context: RecommendationContext) -> int:
    columns = ["split", "cutoff_week", "label_week", "customer_id"]
    if context.target_users.empty:
        return 0
    return int(context.target_users.loc[:, columns].drop_duplicates().shape[0])
