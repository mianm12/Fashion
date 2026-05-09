from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.recommendation.contracts import (
    CANDIDATE_ITEM_COLUMNS,
    RECOMMENDATION_ITEMS_COLUMNS,
    RECOMMENDATIONS_COLUMNS,
    USER_PROFILE_COLUMNS,
)
from fashion_trend.recommendation.methods.base import RecommendationContext
from fashion_trend.recommendation.registry import get_recommendation_method


def test_registry_lists_unknown_method_choices() -> None:
    with pytest.raises(ValueError, match="global_popularity.*pop_similarity"):
        get_recommendation_method("missing")


def test_global_popularity_method_contract() -> None:
    method = get_recommendation_method("global_popularity")

    assert method.name == "global_popularity"
    assert method.default_candidate_strategy is None
    assert method.required_features == ("pop_score",)
    assert method.default_weights == {"pop_score": 1.0}


def test_recent_popularity_method_contract() -> None:
    method = get_recommendation_method("recent_popularity")

    assert method.name == "recent_popularity"
    assert method.default_candidate_strategy is None
    assert method.required_features == ("recent_score",)


def test_pop_similarity_method_contract() -> None:
    method = get_recommendation_method("pop_similarity")

    assert method.default_candidate_strategy == "default"
    assert method.required_features == ("pop_score", "sim_score", "recent_score")
    assert method.default_weights == {
        "pop_score": 0.45,
        "sim_score": 0.45,
        "recent_score": 0.10,
    }


def sample_method_context(
    *,
    method_name: str = "global_popularity",
    exclude_seen: bool = True,
    user_profile: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
) -> RecommendationContext:
    windows = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            }
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4"],
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "week_id": [9, 9, 10, 10],
        }
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "attr_id": [101, 102, 101, 103],
            "attr_type": ["product_type_name"] * 4,
            "attr_value": ["Dress", "Shirt", "Dress", "Shoes"],
        }
    )
    if user_profile is None:
        user_profile = pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "customer_id": "u1",
                    "attr_id": 101,
                    "attr_type": "product_type_name",
                    "attr_value": "Dress",
                    "preference_score": 1.0,
                    "purchase_count": 1,
                    "last_purchase_week": 9,
                }
            ],
            columns=list(USER_PROFILE_COLUMNS),
        )
    if candidates is None:
        candidates = pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "strategy": "default",
                    "customer_id": "u1",
                    "article_id": "0000000003",
                    "candidate_sources": "popularity|similarity",
                    "primary_source": "similarity",
                    "best_source_rank": 1,
                },
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "strategy": "default",
                    "customer_id": "u1",
                    "article_id": "0000000001",
                    "candidate_sources": "popularity",
                    "primary_source": "popularity",
                    "best_source_rank": 2,
                },
            ],
            columns=list(CANDIDATE_ITEM_COLUMNS),
        )
    return RecommendationContext(
        method=method_name,
        top_k=12,
        exclude_seen=exclude_seen,
        transactions=transactions,
        article_attributes=article_attributes,
        windows=windows,
        target_users=target_users,
        candidates=candidates,
        user_profile=user_profile,
        trend_predictions=None,
    )


def assert_method_result_shape(result, method_name: str) -> None:
    assert tuple(result.recommendations.columns) == RECOMMENDATIONS_COLUMNS
    assert tuple(result.recommendation_items.columns) == RECOMMENDATION_ITEMS_COLUMNS
    assert set(result.recommendations["method"]) == {method_name}
    assert set(result.recommendation_items["method"]) == {method_name}
    assert result.recommendation_items["rank"].tolist() == list(
        range(1, len(result.recommendation_items) + 1)
    )
    assert result.recommendation_items["rank"].between(1, 12).all()
    assert result.params["method"] == method_name


@pytest.mark.parametrize("method_name", ["global_popularity", "recent_popularity"])
def test_popularity_baselines_build_without_profile_or_candidates(
    method_name: str,
) -> None:
    method = get_recommendation_method(method_name)
    context = sample_method_context(
        method_name=method_name,
        user_profile=None,
        candidates=None,
    )

    result = method.build_recommendations(context)

    assert_method_result_shape(result, method_name)
    assert "0000000001" not in set(result.recommendation_items["article_id"])
    assert result.params["exclude_seen"] is True


def test_attribute_similarity_falls_back_when_profile_is_empty() -> None:
    method = get_recommendation_method("attribute_similarity")
    empty_profile = pd.DataFrame(columns=list(USER_PROFILE_COLUMNS))
    context = sample_method_context(
        method_name="attribute_similarity",
        user_profile=empty_profile,
    )

    result = method.build_recommendations(context)

    assert_method_result_shape(result, "attribute_similarity")
    assert result.metadata["fallback_user_count"] == 1
    assert result.params["weights"] == {"recent_score": 1.0}


def test_pop_similarity_builds_without_trend_predictions() -> None:
    method = get_recommendation_method("pop_similarity")
    context = sample_method_context(method_name="pop_similarity")

    result = method.build_recommendations(context)

    assert_method_result_shape(result, "pop_similarity")
    assert "trend_score" in result.recommendation_items.columns
    assert result.recommendation_items["trend_score"].eq(0.0).all()
