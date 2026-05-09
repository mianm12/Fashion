from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.recommendation.retrieval.candidates import (
    build_candidate_items,
    build_source_frames_for_frames,
    validate_candidate_strategy,
)
from fashion_trend.recommendation.retrieval.popularity import (
    build_popularity_candidates,
)
from fashion_trend.recommendation.retrieval.trend import build_trend_candidates


def sample_window() -> pd.DataFrame:
    return pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )


def sample_targets() -> pd.DataFrame:
    return pd.DataFrame(
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


def test_popularity_candidates_ignore_label_week_transactions() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3"],
            "article_id": ["0000000001", "0000000002", "0000000999"],
            "week_id": [8, 10, 11],
        }
    )

    candidates = build_popularity_candidates(
        transactions,
        sample_window(),
        sample_targets(),
        top_n=10,
    )

    assert set(candidates["article_id"]) == {"0000000001", "0000000002"}
    assert "0000000999" not in set(candidates["article_id"])
    assert candidates["article_id"].map(type).eq(str).all()
    assert candidates["customer_id"].map(type).eq(str).all()


def test_default_candidates_merge_sources_with_best_rank() -> None:
    popularity = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "article_id": "0000000001",
                "source": "popularity",
                "source_rank": 2,
            }
        ]
    )
    similarity = popularity.assign(source="similarity", source_rank=1)
    trend = popularity.assign(source="trend", source_rank=3)

    candidates = build_candidate_items(
        strategy="default",
        source_frames=[popularity, similarity, trend],
    )

    assert candidates.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "strategy": "default",
            "customer_id": "u1",
            "article_id": "0000000001",
            "candidate_sources": "popularity|similarity|trend",
            "primary_source": "similarity",
            "best_source_rank": 1,
        }
    ]


def test_trend_union_requires_predictions() -> None:
    with pytest.raises(FileNotFoundError):
        build_candidate_items(strategy="trend_union", source_frames=[])


def test_unknown_strategy_fails_in_domain_layer() -> None:
    with pytest.raises(ValueError, match="未知候选 strategy"):
        validate_candidate_strategy("missing")
    with pytest.raises(ValueError, match="未知候选 strategy"):
        build_candidate_items(strategy="missing", source_frames=[])


def test_popularity_strategy_does_not_require_profile_or_trend_predictions() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "article_id": ["0000000001", "0000000002"],
            "week_id": [8, 10],
        }
    )

    frames = build_source_frames_for_frames(
        strategy="popularity",
        transactions=transactions,
        article_attributes=None,
        trend_predictions=None,
        windows=sample_window(),
        target_users=sample_targets(),
        user_profile=None,
    )

    assert len(frames) == 1
    assert set(frames[0]["source"]) == {"popularity"}


def test_trend_strategy_requires_trend_predictions() -> None:
    with pytest.raises(FileNotFoundError, match="trend predictions"):
        build_source_frames_for_frames(
            strategy="trend_union",
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=None,
            windows=sample_window(),
            target_users=sample_targets(),
            user_profile=None,
        )


def test_trend_candidates_use_cutoff_week_and_core_attributes_only() -> None:
    predictions = pd.DataFrame(
        [
            {
                "split": "valid",
                "week_id": 10,
                "attr_type": "product_type_name",
                "attr_value": "dress",
                "pred_target_growth": 0.4,
            },
            {
                "split": "valid",
                "week_id": 11,
                "attr_type": "product_type_name",
                "attr_value": "shirt",
                "pred_target_growth": 9.0,
            },
            {
                "split": "valid",
                "week_id": 10,
                "attr_type": "detail_desc",
                "attr_value": "ignored",
                "pred_target_growth": 10.0,
            },
        ]
    )
    article_attributes = pd.DataFrame(
        [
            {
                "article_id": "0000000002",
                "attr_type": "product_type_name",
                "attr_value": "dress",
            },
            {
                "article_id": "0000000001",
                "attr_type": "product_type_name",
                "attr_value": "dress",
            },
            {
                "article_id": "0000000999",
                "attr_type": "product_type_name",
                "attr_value": "shirt",
            },
            {
                "article_id": "0000000888",
                "attr_type": "detail_desc",
                "attr_value": "ignored",
            },
        ]
    )

    candidates = build_trend_candidates(
        predictions,
        article_attributes,
        sample_window(),
        sample_targets(),
        top_n=10,
    )

    assert candidates["article_id"].tolist() == ["0000000001", "0000000002"]
    assert candidates["source_rank"].tolist() == [1, 2]


def test_popularity_source_rank_is_one_based_with_article_tie_break() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3"],
            "article_id": ["0000000002", "0000000001", "0000000003"],
            "week_id": [10, 10, 10],
        }
    )

    candidates = build_popularity_candidates(
        transactions,
        sample_window(),
        sample_targets(),
        top_n=3,
    )

    assert candidates["article_id"].tolist() == [
        "0000000001",
        "0000000002",
        "0000000003",
    ]
    assert candidates["source_rank"].tolist() == [1, 2, 3]
