from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fashion_trend.recommendation.contracts import (
    RECOMMENDATION_CANDIDATES_PER_SOURCE,
)
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationMethod,
    RecommendationResult,
)
from fashion_trend.recommendation.outputs import (
    build_recommendations_csv,
    format_recommendation_items,
)
from fashion_trend.recommendation.ranking.backfill import append_backfill_items
from fashion_trend.recommendation.ranking.features import build_ranking_features
from fashion_trend.recommendation.ranking.filters import filter_seen_items
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items
from fashion_trend.recommendation.retrieval.candidates import build_candidate_items
from fashion_trend.recommendation.retrieval.popularity import (
    build_popularity_candidates,
)


@dataclass(frozen=True)
class GlobalPopularityMethod:
    name: str = "global_popularity"
    method_type: str = "baseline"
    default_candidate_strategy: str | None = None
    default_weights: dict[str, float] = field(
        default_factory=lambda: {"pop_score": 1.0}
    )
    required_features: tuple[str, ...] = ("pop_score",)

    def build_recommendations(
        self,
        context: RecommendationContext,
    ) -> RecommendationResult:
        source = build_popularity_candidates(
            context.transactions,
            context.windows,
            context.target_users,
            top_n=max(context.top_k, RECOMMENDATION_CANDIDATES_PER_SOURCE),
        )
        candidates = build_candidate_items("popularity", [source])
        return build_baseline_recommendation_result(
            self,
            context,
            candidates,
            backfill_mode="popularity",
        )


def build_baseline_recommendation_result(
    method: RecommendationMethod,
    context: RecommendationContext,
    candidates: pd.DataFrame,
    metadata: dict[str, object] | None = None,
    backfill_mode: str | None = None,
) -> RecommendationResult:
    weights = context.weights if context.weights is not None else method.default_weights
    feature_frame = build_ranking_features(
        candidates,
        context.transactions,
        context.article_attributes,
        context.user_profile,
        context.trend_predictions,
    )
    feature_frame["method"] = context.method
    if context.exclude_seen:
        feature_frame = filter_seen_items(feature_frame, context.transactions)
    ranked = rank_candidate_items(
        feature_frame,
        weights,
        context.top_k,
        method.required_features,
    )
    ranked = append_backfill_items(
        context,
        candidates,
        ranked,
        weights,
        backfill_mode,
    )
    recommendation_items = format_recommendation_items(ranked)
    recommendations = build_recommendations_csv(recommendation_items, context.top_k)
    return RecommendationResult(
        recommendations=recommendations,
        recommendation_items=recommendation_items,
        params={
            "method": context.method,
            "method_type": method.method_type,
            "top_k": context.top_k,
            "exclude_seen": context.exclude_seen,
            "weights": dict(weights),
            "candidate_strategy": method.default_candidate_strategy,
            "score_features": list(method.required_features),
        },
        metadata={
            "method": context.method,
            "candidate_rows": int(len(candidates)),
            "recommendation_item_rows": int(len(recommendation_items)),
            **(metadata or {}),
        },
    )
