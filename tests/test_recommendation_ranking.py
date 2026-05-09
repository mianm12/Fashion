from __future__ import annotations

import math

import pandas as pd
import pytest

from fashion_trend.recommendation.ranking.features import (
    build_ranking_features,
    minmax_normalize_by_group,
)
from fashion_trend.recommendation.ranking.filters import filter_seen_items
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items
from fashion_trend.recommendation.ranking.weights import validate_score_weights


REQUIRED_WEIGHTS = ("pop_score", "recent_score", "sim_score", "trend_score")


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
    )

    assert ranked["article_id"].tolist() == ["0000000010", "0000000020"]
    assert ranked["rank"].tolist() == [1, 2]


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
            "attr_type": ["color", "color", "color"],
            "attr_value": ["red", "blue", "green"],
        }
    )
    user_profile = pd.DataFrame(
        {
            "customer_id": ["u1", "u1"],
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "attr_type": ["color", "color"],
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
