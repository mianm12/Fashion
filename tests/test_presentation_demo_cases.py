from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.presentation.demo_cases import build_demo_case_payloads


def test_build_demo_case_payloads_selects_complete_cases_by_hit_count() -> None:
    recommendation_items = _recommendation_items(["bad", "hit2", "hit1", "miss"])
    recommendation_items.loc[
        (recommendation_items["customer_id"] == "bad")
        & (recommendation_items["rank"] == 1),
        "candidate_sources",
    ] = None

    payloads = build_demo_case_payloads(
        recommendation_items=recommendation_items,
        evaluation_labels=_evaluation_labels(
            {"bad": [1, 2, 3, 4, 5], "hit2": [1, 2], "hit1": [1]}
        ),
        user_profile=_user_profile(["bad", "hit2", "hit1", "miss"]),
        min_case_count=2,
        max_case_count=2,
    )

    assert [payload["customer_id"] for payload in payloads] == ["hit2", "hit1"]
    assert payloads[0]["hit_count"] == 2
    assert len(payloads[0]["recommendations"]) == 12
    assert payloads[0]["profile"][0]["attr_value"] == "Black"


def test_build_demo_case_payloads_rejects_too_small_quality_pool() -> None:
    with pytest.raises(ValueError, match="不足 2 个高质量演示用户案例"):
        build_demo_case_payloads(
            recommendation_items=_recommendation_items(["only-one"]),
            evaluation_labels=_evaluation_labels({"only-one": [1]}),
            user_profile=_user_profile(["only-one"]),
            min_case_count=2,
            max_case_count=50,
        )


def _recommendation_items(customer_ids: list[str]):
    return pd.DataFrame(
        [
            {
                "customer_id": customer_id,
                "split": "test",
                "cutoff_week": 10,
                "label_week": 11,
                "method": "pop_similarity_trend",
                "article_id": f"{rank:010d}",
                "rank": rank,
                "score": 1.0 / rank,
                "pop_score": 0.1,
                "sim_score": 0.2,
                "trend_score": 0.3,
                "recent_score": 0.4,
                "candidate_sources": "popularity,trend_union",
            }
            for customer_id in customer_ids
            for rank in range(1, 13)
        ]
    )


def _evaluation_labels(hits_by_customer: dict[str, list[int]]):
    return pd.DataFrame(
        [
            {
                "customer_id": customer_id,
                "split": "test",
                "cutoff_week": 10,
                "label_week": 11,
                "article_id": f"{rank:010d}",
            }
            for customer_id, ranks in hits_by_customer.items()
            for rank in ranks
        ]
    )


def _user_profile(customer_ids: list[str]):
    return pd.DataFrame(
        [
            {
                "customer_id": customer_id,
                "split": "test",
                "cutoff_week": 10,
                "label_week": 11,
                "attr_id": f"colour_group_name::{value}",
                "attr_type": "colour_group_name",
                "attr_value": value,
                "preference_score": score,
                "purchase_count": count,
                "last_purchase_week": 10 - index,
            }
            for customer_id in customer_ids
            for index, (value, score, count) in enumerate(
                [("Black", 0.9, 3), ("Blue", 0.7, 2), ("White", 0.5, 1)]
            )
        ]
    )
