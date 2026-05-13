from __future__ import annotations

import math

import pandas as pd
import pytest

from fashion_trend.recommendation import contracts
from fashion_trend.recommendation.ranking.features import (
    build_article_trend_scores,
    build_ranking_features,
    minmax_normalize_by_group,
)
from fashion_trend.recommendation.ranking.filters import filter_seen_items
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items
from fashion_trend.recommendation.ranking.weights import validate_score_weights

REQUIRED_WEIGHTS = ("pop_score", "recent_score", "sim_score", "trend_score")
EXPECTED_ENHANCED_SCORE_COLUMNS = (
    "pop_score",
    "recent_score",
    "sim_score",
    "trend_score",
    "reorder_score",
    "variant_score",
    "age_pop_score",
    "preference_pop_score",
    "source_rank_score",
    "source_count_score",
)


def test_minmax_constant_group_fills_zero() -> None:
    frame = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "value": [5.0, 5.0],
        }
    )

    result = minmax_normalize_by_group(
        frame,
        value_column="value",
        output_column="score",
        group_columns=("split", "cutoff_week", "label_week"),
    )

    assert result["score"].tolist() == [0.0, 0.0]
    assert not result["score"].isna().any()


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"pop_score": 0.5, "recent_score": 0.5, "sim_score": 0.5}, "keys"),
        (
            {
                "pop_score": 0.4,
                "recent_score": 0.3,
                "sim_score": 0.2,
                "trend_score": 0.1,
                "extra_score": 0.0,
            },
            "keys",
        ),
        (
            {
                "pop_score": 0.4,
                "recent_score": 0.3,
                "sim_score": 0.4,
                "trend_score": -0.1,
            },
            "invalid",
        ),
        (
            {
                "pop_score": 0.4,
                "recent_score": 0.3,
                "sim_score": 0.3,
                "trend_score": math.inf,
            },
            "invalid",
        ),
        (
            {
                "pop_score": 0.5,
                "recent_score": 0.5,
                "sim_score": 0.5,
                "trend_score": 0.0,
            },
            "sum",
        ),
    ],
)
def test_validate_score_weights_rejects_invalid_weights(
    weights: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_score_weights(weights, required_features=REQUIRED_WEIGHTS)


def test_validate_score_weights_returns_copy_without_normalizing() -> None:
    weights = {
        "pop_score": 0.3,
        "recent_score": 0.2,
        "sim_score": 0.4,
        "trend_score": 0.1,
    }

    result = validate_score_weights(weights, required_features=REQUIRED_WEIGHTS)

    assert result == weights
    assert result is not weights


def test_rank_candidate_items_uses_stable_tie_break_and_top_k() -> None:
    candidates = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "split": ["valid", "valid", "valid"],
            "cutoff_week": [10, 10, 10],
            "label_week": [11, 11, 11],
            "method": ["pop_similarity", "pop_similarity", "pop_similarity"],
            "article_id": ["0000000020", "0000000010", "0000000030"],
            "pop_score": [0.5, 0.5, 0.9],
            "recent_score": [0.0, 0.0, 0.0],
            "sim_score": [0.5, 0.5, 0.0],
            "trend_score": [0.0, 0.0, 0.0],
            "candidate_sources": ["popularity", "popularity", "popularity"],
        }
    )

    ranked = rank_candidate_items(
        candidates,
        weights={
            "pop_score": 0.5,
            "recent_score": 0.0,
            "sim_score": 0.5,
            "trend_score": 0.0,
        },
        top_k=2,
        required_features=REQUIRED_WEIGHTS,
    )

    assert ranked["article_id"].tolist() == ["0000000010", "0000000020"]
    assert ranked["rank"].tolist() == [1, 2]


def test_rank_candidate_items_rejects_weights_outside_required_features() -> None:
    candidates = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "article_id": ["0000000010"],
            "pop_score": [0.5],
            "recent_score": [0.5],
        }
    )

    with pytest.raises(ValueError, match="keys"):
        rank_candidate_items(
            candidates,
            weights={"pop_score": 0.5, "recent_score": 0.5},
            top_k=1,
            required_features=("pop_score",),
        )


def test_filter_seen_items_uses_history_at_or_before_cutoff() -> None:
    candidates = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "split": ["valid", "valid", "valid"],
            "cutoff_week": [10, 10, 10],
            "label_week": [11, 11, 11],
            "article_id": ["0000000001", "0000000002", "0000000003"],
        }
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1"],
            "article_id": ["0000000001", "0000000003"],
            "week_id": [10, 11],
        }
    )

    filtered = filter_seen_items(candidates, transactions)

    assert filtered["article_id"].tolist() == ["0000000002", "0000000003"]
    assert filtered.columns.tolist() == candidates.columns.tolist()


def test_build_ranking_features_uses_cutoff_history_and_bounded_scores() -> None:
    candidates = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "split": ["valid", "valid", "valid"],
            "cutoff_week": [10, 10, 10],
            "label_week": [11, 11, 11],
            "article_id": ["0000000001", "0000000002", "0000000003"],
        }
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4", "u5"],
            "article_id": [
                "0000000001",
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000003",
            ],
            "week_id": [7, 10, 9, 11, 6],
        }
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002", "0000000003"],
            "attr_type": ["colour_group_name"] * 3,
            "attr_value": ["red", "blue", "green"],
        }
    )
    user_profile = pd.DataFrame(
        {
            "customer_id": ["u1", "u1"],
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "attr_type": ["colour_group_name", "colour_group_name"],
            "attr_value": ["red", "blue"],
            "preference_score": [2.0, 1.0],
        }
    )

    features = build_ranking_features(
        candidates,
        transactions,
        article_attributes,
        user_profile,
        trend_predictions=None,
    )

    assert features["article_id"].tolist() == candidates["article_id"].tolist()
    assert features["pop_score"].tolist() == [1.0, 0.0, 0.0]
    assert features["recent_score"].tolist() == [1.0, 0.5, 0.0]
    assert features["sim_score"].tolist() == [1.0, 0.5, 0.0]
    assert features["trend_score"].tolist() == [0.0, 0.0, 0.0]
    score_columns = ["pop_score", "recent_score", "sim_score", "trend_score"]
    assert features[score_columns].map(lambda value: 0.0 <= value <= 1.0).all().all()


def test_enhanced_rank_norm_scores_are_group_normalized_to_unit_interval() -> None:
    candidates = _enhanced_score_candidates()

    assert (
        contracts.ENHANCED_RECOMMENDATION_SCORE_COLUMNS
        == EXPECTED_ENHANCED_SCORE_COLUMNS
    )
    features = build_ranking_features(
        candidates,
        _enhanced_transactions(),
        _enhanced_article_attributes(),
        _enhanced_user_profile(),
        trend_predictions=None,
        customer_profile=_enhanced_customer_profile(),
        article_product_map=_enhanced_article_product_map(),
    )

    enhanced_columns = list(EXPECTED_ENHANCED_SCORE_COLUMNS[4:])
    assert features[enhanced_columns].map(lambda value: 0.0 <= value <= 1.0).all().all()
    assert features[enhanced_columns].max().tolist() == [1.0] * len(enhanced_columns)
    assert features[enhanced_columns].min().tolist() == [0.0] * len(enhanced_columns)


def test_enhanced_scores_fill_missing_sources_with_zero() -> None:
    candidates = _enhanced_score_candidates().iloc[[3]].reset_index(drop=True)

    features = build_ranking_features(
        candidates,
        _enhanced_transactions(),
        _enhanced_article_attributes(),
        _enhanced_user_profile(),
        trend_predictions=None,
        customer_profile=_enhanced_customer_profile(),
        article_product_map=_enhanced_article_product_map(),
    )

    assert features.loc[
        0,
        [
            "reorder_score",
            "variant_score",
            "age_pop_score",
            "preference_pop_score",
            "source_rank_score",
            "source_count_score",
        ],
    ].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_source_rank_score_uses_materialized_best_source_rank() -> None:
    candidates = pd.DataFrame(
        [
            _enhanced_candidate("0000000004", "age_popularity", 1),
            _enhanced_candidate("0000000003", "age_popularity", 2),
        ]
    )

    scores = build_ranking_features(
        candidates,
        _enhanced_transactions(),
        _enhanced_article_attributes(),
        _enhanced_user_profile(),
        trend_predictions=None,
        customer_profile=_enhanced_customer_profile(),
    )

    assert scores.loc[0, "source_rank_score"] > scores.loc[1, "source_rank_score"]
    assert scores.loc[0, "best_source_rank"] < scores.loc[1, "best_source_rank"]


def test_source_count_score_uses_filtered_source_count_cap() -> None:
    candidates = _source_count_candidates()
    filtered = candidates.iloc[:2].copy()
    filtered.loc[0, "candidate_sources"] = "popularity|similarity"

    full_scores = build_ranking_features(
        candidates,
        _enhanced_transactions(),
        _enhanced_article_attributes(),
        _enhanced_user_profile(),
        trend_predictions=None,
    )
    filtered_scores = build_ranking_features(
        filtered,
        _enhanced_transactions(),
        _enhanced_article_attributes(),
        _enhanced_user_profile(),
        trend_predictions=None,
    )

    assert full_scores.loc[0, "source_count_score"] == pytest.approx(1.0)
    assert filtered_scores.loc[0, "source_count_score"] == pytest.approx(0.0)


def test_trend_score_uses_prediction_week_equal_cutoff_week() -> None:
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002"],
            "attr_id": [101, 102],
            "attr_type": ["product_type_name", "product_type_name"],
            "attr_value": ["Dress", "Shirt"],
        }
    )
    predictions = pd.DataFrame(
        {
            "split": ["valid", "valid", "valid", "valid"],
            "week_id": [10, 10, 11, 11],
            "attr_id": [101, 102, 101, 102],
            "attr_type": ["product_type_name"] * 4,
            "attr_value": ["Dress", "Shirt", "Dress", "Shirt"],
            "pred_target_growth": [2.0, 1.0, 0.0, 10.0],
            "pred_share_t1": [0.0, 10.0, 0.0, 10.0],
        }
    )

    scores = build_article_trend_scores(predictions, article_attributes, windows)

    assert scores.loc[
        scores["article_id"] == "0000000001", "trend_score"
    ].item() == pytest.approx(1.0)


def test_trend_score_renormalizes_weights_for_matched_attribute_types() -> None:
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001"],
            "attr_id": [101],
            "attr_type": ["product_type_name"],
            "attr_value": ["Dress"],
        }
    )
    predictions = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "week_id": [10, 10],
            "attr_id": [101, 102],
            "attr_type": ["product_type_name", "product_type_name"],
            "attr_value": ["Dress", "Shirt"],
            "pred_target_growth": [2.0, 1.0],
            "pred_share_t1": [0.0, 10.0],
        }
    )

    scores = build_article_trend_scores(predictions, article_attributes, windows)

    assert scores["trend_score"].tolist() == [pytest.approx(1.0)]


def _enhanced_score_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _enhanced_candidate(
                "0000000001",
                "reorder|age_popularity|preference_popularity",
                1,
                has_reorder_source=True,
            ),
            _enhanced_candidate(
                "0000000002",
                "reorder|age_popularity",
                2,
                has_reorder_source=True,
            ),
            _enhanced_candidate(
                "0000000003",
                "product_variant|preference_popularity",
                3,
            ),
            _enhanced_candidate("0000000004", "popularity", 4),
        ]
    )


def _source_count_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _enhanced_candidate("0000000001", "popularity|similarity|trend", 1),
            _enhanced_candidate("0000000002", "popularity|similarity", 2),
            _enhanced_candidate("0000000003", "trend", 3),
        ]
    )


def _enhanced_candidate(
    article_id: str,
    candidate_sources: str,
    best_source_rank: int,
    *,
    has_reorder_source: bool = False,
) -> dict[str, object]:
    return {
        "customer_id": "u1",
        "split": "valid",
        "cutoff_week": 10,
        "label_week": 11,
        "strategy": "enhanced_default",
        "article_id": article_id,
        "candidate_sources": candidate_sources,
        "primary_source": candidate_sources.split("|")[0],
        "best_source_rank": best_source_rank,
        "has_reorder_source": has_reorder_source,
        "allow_seen": has_reorder_source,
    }


def _enhanced_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "week_id": 10},
            {"customer_id": "u1", "article_id": "0000000001", "week_id": 8},
            {"customer_id": "u1", "article_id": "0000000002", "week_id": 9},
            {"customer_id": "u2", "article_id": "0000000001", "week_id": 10},
            {"customer_id": "u2", "article_id": "0000000002", "week_id": 10},
            {"customer_id": "u2", "article_id": "0000000003", "week_id": 9},
            {"customer_id": "u2", "article_id": "0000000003", "week_id": 8},
            {"customer_id": "u2", "article_id": "0000000004", "week_id": 7},
        ]
    )


def _enhanced_article_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
            ],
            "attr_type": ["colour_group_name"] * 4,
            "attr_value": ["red", "blue", "red", "red"],
        }
    )


def _enhanced_user_profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "u1",
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "attr_type": "colour_group_name",
                "attr_value": "red",
                "preference_score": 1.0,
            },
            {
                "customer_id": "u1",
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "attr_type": "colour_group_name",
                "attr_value": "blue",
                "preference_score": 0.5,
            },
        ]
    )


def _enhanced_customer_profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "age_bucket": ["20-29", "20-29"],
        }
    )


def _enhanced_article_product_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
            ],
            "product_code": ["p1", "p2", "p1", "p2"],
        }
    )
