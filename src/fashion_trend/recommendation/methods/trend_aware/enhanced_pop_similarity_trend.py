from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fashion_trend.recommendation.contracts import (
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
)
from fashion_trend.recommendation.features.cache import build_candidate_seen_flags
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.outputs import (
    build_recommendations_csv,
    format_recommendation_items,
)
from fashion_trend.recommendation.ranking.features import build_ranking_features
from fashion_trend.recommendation.ranking.filters import (
    filter_seen_items_by_source_policy,
)
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items


@dataclass(frozen=True)
class EnhancedPopSimilarityTrendMethod:
    name: str = "enhanced_pop_similarity_trend"
    method_type: str = "trend_aware"
    default_candidate_strategy: str = "enhanced_default"
    default_weights: dict[str, float] = field(
        default_factory=lambda: {
            "pop_score": 0.14,
            "recent_score": 0.30,
            "sim_score": 0.10,
            "trend_score": 0.08,
            "reorder_score": 0.16,
            "variant_score": 0.08,
            "age_pop_score": 0.04,
            "preference_pop_score": 0.04,
            "source_rank_score": 0.03,
            "source_count_score": 0.03,
        }
    )
    required_features: tuple[str, ...] = ENHANCED_RECOMMENDATION_SCORE_COLUMNS

    def build_recommendations(
        self,
        context: RecommendationContext,
    ) -> RecommendationResult:
        if context.trend_predictions is None or context.trend_predictions.empty:
            raise FileNotFoundError(
                "enhanced_pop_similarity_trend requires trend predictions"
            )
        if context.candidates is None:
            raise FileNotFoundError(
                "enhanced_pop_similarity_trend requires enhanced_default candidates"
            )

        weights = (
            context.weights if context.weights is not None else self.default_weights
        )
        feature_frame = build_ranking_features(
            context.candidates,
            context.transactions,
            context.article_attributes,
            context.user_profile,
            context.trend_predictions,
        )
        feature_frame["method"] = context.method
        if context.exclude_seen:
            feature_frame = _filter_seen_by_source_policy(
                feature_frame,
                context.candidates,
                context.transactions,
            )
        ranked = rank_candidate_items(
            feature_frame,
            weights=weights,
            top_k=context.top_k,
            required_features=self.required_features,
        )
        underfilled_user_count = _underfilled_user_count(
            context.target_users,
            ranked,
            context.top_k,
        )
        recommendation_items = format_recommendation_items(ranked)
        recommendations = build_recommendations_csv(recommendation_items, context.top_k)
        return RecommendationResult(
            recommendations=recommendations,
            recommendation_items=recommendation_items,
            params={
                "method": context.method,
                "method_type": self.method_type,
                "top_k": context.top_k,
                "exclude_seen": context.exclude_seen,
                "weights": dict(weights),
                "candidate_strategy": self.default_candidate_strategy,
                "score_features": list(self.required_features),
            },
            metadata={
                "method": context.method,
                "candidate_rows": int(len(context.candidates)),
                "recommendation_item_rows": int(len(recommendation_items)),
                "backfill_mode": None,
                "underfilled_user_count": underfilled_user_count,
                "backfilled_user_count": 0,
                "still_underfilled_user_count": underfilled_user_count,
                "candidate_strategy": self.default_candidate_strategy,
                "source_level_seen_policy": "reorder_only",
            },
        )


def _filter_seen_by_source_policy(
    feature_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    seen_flags = build_candidate_seen_flags(candidates, transactions)
    join_columns = [
        "split",
        "cutoff_week",
        "label_week",
        "strategy",
        "customer_id",
        "article_id",
    ]
    if seen_flags.empty:
        result = feature_frame.copy()
        result["is_seen"] = False
        return filter_seen_items_by_source_policy(result).loc[:, feature_frame.columns]

    marker = seen_flags.loc[:, [*join_columns, "is_seen"]].drop_duplicates()
    result = feature_frame.merge(marker, on=join_columns, how="left")
    result["is_seen"] = result["is_seen"].fillna(False).astype(bool)
    return filter_seen_items_by_source_policy(result).loc[:, feature_frame.columns]


def _underfilled_user_count(
    target_users: pd.DataFrame,
    items: pd.DataFrame,
    top_k: int,
) -> int:
    key_columns = ["split", "cutoff_week", "label_week", "customer_id"]
    targets = target_users.loc[:, key_columns].drop_duplicates().copy()
    if items.empty:
        return int(len(targets))
    counts = (
        items.groupby(key_columns, as_index=False)
        .size()
        .rename(columns={"size": "_count"})
    )
    merged = targets.merge(counts, on=key_columns, how="left")
    merged["_count"] = merged["_count"].fillna(0).astype(int)
    return int((merged["_count"] < top_k).sum())
