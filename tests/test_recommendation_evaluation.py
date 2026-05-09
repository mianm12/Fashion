from __future__ import annotations

import math

import pandas as pd
import pytest

from fashion_trend.recommendation.evaluation.metrics import (
    evaluate_recommendations,
    parse_prediction_items,
)


def test_missing_recommendation_user_scores_zero_by_default() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u2",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "customer_id": ["u1", "u2"],
            "article_id": ["0000000001", "0000000002"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "method": ["recent_popularity"],
            "prediction": ["0000000001 0000000003 0000000004"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "article_id": ["0000000001"],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["user_count"] == 2
    assert metrics["valid"]["missing_recommendation_user_count"] == 1
    assert metrics["valid"]["hit_rate_at_12"] == 0.5
    assert metrics["valid"]["recall_at_12"] == 0.5


def test_missing_recommendation_user_fails_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="missing"):
        evaluate_recommendations(
            pd.DataFrame(
                columns=[
                    "customer_id",
                    "split",
                    "cutoff_week",
                    "label_week",
                    "method",
                    "prediction",
                ]
            ),
            pd.DataFrame(
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
            ),
            pd.DataFrame(
                {
                    "split": ["valid"],
                    "cutoff_week": [10],
                    "label_week": [11],
                    "customer_id": ["u1"],
                    "article_id": ["0000000001"],
                }
            ),
            pd.DataFrame(
                {
                    "split": ["valid"],
                    "cutoff_week": [10],
                    "label_week": [11],
                    "article_id": ["0000000001"],
                }
            ),
            top_k=12,
            strict_missing_users=True,
        )


def test_ranking_metrics_use_exact_relevant_sets() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 2,
            }
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid", "valid"],
            "cutoff_week": [10, 10, 10],
            "label_week": [11, 11, 11],
            "customer_id": ["u1", "u1", "u1"],
            "article_id": ["0000000001", "0000000003", "0000000003"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "method": ["pop_similarity_trend"],
            "prediction": ["0000000001 0000000002 0000000003"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"] * 4,
            "cutoff_week": [10] * 4,
            "label_week": [11] * 4,
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
            ],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["map_at_12"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert metrics["valid"]["recall_at_12"] == 1.0
    assert metrics["valid"]["hit_rate_at_12"] == 1.0
    assert metrics["valid"]["ndcg_at_12"] == pytest.approx(
        (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
    )
    assert metrics["valid"]["coverage"] == 0.75


def test_coverage_is_computed_per_window_before_split_average() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
            {
                "split": "valid",
                "cutoff_week": 20,
                "label_week": 21,
                "customer_id": "u2",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 20],
            "label_week": [11, 21],
            "customer_id": ["u1", "u2"],
            "article_id": ["0000000001", "0000000005"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "split": ["valid", "valid"],
            "cutoff_week": [10, 20],
            "label_week": [11, 21],
            "method": ["recent_popularity", "recent_popularity"],
            "prediction": ["0000000001 0000000002", "0000000005"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"] * 6,
            "cutoff_week": [10, 10, 10, 10, 20, 20],
            "label_week": [11, 11, 11, 11, 21, 21],
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
                "0000000005",
                "0000000006",
            ],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["coverage_by_window"] == [
        {"cutoff_week": 10, "label_week": 11, "coverage": 0.5},
        {"cutoff_week": 20, "label_week": 21, "coverage": 0.5},
    ]
    assert metrics["valid"]["coverage"] == 0.5


def test_coverage_ignores_recommendations_outside_recommendable_pool() -> None:
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
    labels = pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "customer_id": ["u1"],
            "article_id": ["0000000001"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "method": ["recent_popularity"],
            "prediction": ["0000000001 0000000999"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "article_id": ["0000000001"],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
    )

    assert metrics["valid"]["coverage"] == 1.0


def test_parse_prediction_items_rejects_duplicate_or_too_many_articles() -> None:
    assert parse_prediction_items("0000000001 0000000002", top_k=2) == [
        "0000000001",
        "0000000002",
    ]

    with pytest.raises(ValueError, match="duplicate"):
        parse_prediction_items("0000000001 0000000001", top_k=12)

    with pytest.raises(ValueError, match="more than 1"):
        parse_prediction_items("0000000001 0000000002", top_k=1)


def test_empty_target_users_fails() -> None:
    with pytest.raises(ValueError, match="target_users"):
        evaluate_recommendations(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            top_k=12,
        )
