from __future__ import annotations

import json

import pandas as pd
import pytest

from fashion_trend.reports.cases import (
    build_case_payload,
    render_case_markdown,
    select_recommendation_cases,
)


def _item(
    customer_id: str,
    split: str,
    article_id: str,
    rank: int,
    score: float,
    *,
    cutoff_week: int = 103,
    label_week: int = 104,
    candidate_sources: str | None = "popularity,trend_union",
) -> dict[str, object]:
    return {
        "customer_id": customer_id,
        "split": split,
        "cutoff_week": cutoff_week,
        "label_week": label_week,
        "method": "pop_similarity_trend",
        "article_id": article_id,
        "rank": rank,
        "score": score,
        "pop_score": score - 0.10,
        "sim_score": score - 0.20,
        "trend_score": score - 0.30,
        "recent_score": score - 0.40,
        "candidate_sources": candidate_sources,
    }


def _recommendation_items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _item("000customer-high", "test", "0000000001", 1, 0.99),
            _item("000customer-high", "test", "0000000002", 2, 0.95),
            _item("000customer-high", "test", "0000000003", 3, 0.90),
            _item("000customer-low", "test", "0000000004", 1, 0.80),
            _item("000customer-low", "test", "0000000005", 2, 0.70),
            _item("000customer-valid", "valid", "0000000006", 1, 0.99),
            _item("000customer-valid", "valid", "0000000007", 2, 0.98),
            _item("000customer-valid", "valid", "0000000008", 3, 0.97),
        ]
    )


def _evaluation_labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "000customer-high",
                "split": "test",
                "cutoff_week": 103,
                "label_week": 104,
                "article_id": "0000000001",
            },
            {
                "customer_id": "000customer-high",
                "split": "test",
                "cutoff_week": 103,
                "label_week": 104,
                "article_id": "0000000002",
            },
            {
                "customer_id": "000customer-low",
                "split": "test",
                "cutoff_week": 103,
                "label_week": 104,
                "article_id": "0000000004",
            },
            {
                "customer_id": "000customer-valid",
                "split": "valid",
                "cutoff_week": 103,
                "label_week": 104,
                "article_id": "0000000006",
            },
            {
                "customer_id": "000customer-valid",
                "split": "valid",
                "cutoff_week": 103,
                "label_week": 104,
                "article_id": "0000000007",
            },
            {
                "customer_id": "000customer-valid",
                "split": "valid",
                "cutoff_week": 103,
                "label_week": 104,
                "article_id": "0000000008",
            },
        ]
    )


def _profile_row(
    customer_id: str,
    attr_type: str,
    attr_value: str | None,
    score: float,
    *,
    cutoff_week: int = 103,
    label_week: int = 104,
) -> dict[str, object]:
    return {
        "customer_id": customer_id,
        "split": "test",
        "cutoff_week": cutoff_week,
        "label_week": label_week,
        "attr_id": f"{attr_type}::{attr_value}",
        "attr_type": attr_type,
        "attr_value": attr_value,
        "preference_score": score,
        "purchase_count": int(score * 10),
        "last_purchase_week": cutoff_week - 1,
    }


def _user_profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _profile_row(
                "000customer-high",
                "colour_group_name",
                "Black",
                0.90,
            ),
            _profile_row(
                "000customer-high",
                "product_group_name",
                "Garment Upper body",
                0.80,
            ),
            _profile_row(
                "000customer-low",
                "graphical_appearance_name",
                "Solid",
                0.70,
            ),
            _profile_row(
                "000customer-valid",
                "colour_group_name",
                "Blue",
                0.95,
            )
            | {"split": "valid"},
        ]
    )


def _article_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": "0000000001",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
            },
            {
                "article_id": "0000000001",
                "attr_type": "product_group_name",
                "attr_value": "Garment Upper body",
            },
            {
                "article_id": "0000000002",
                "attr_type": "graphical_appearance_name",
                "attr_value": "Solid",
            },
            {
                "article_id": "0000000004",
                "attr_type": "colour_group_name",
                "attr_value": "Red",
            },
        ]
    )


def _representative_trends() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "week_id": 102,
                "attr_type": "colour_group_name",
                "attr_value": "Red",
                "pred_target_growth": 0.50,
                "heat_t": 900.0,
            },
            {
                "week_id": 103,
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "pred_target_growth": 0.40,
                "heat_t": 1200.0,
            },
            {
                "week_id": 103,
                "attr_type": "product_group_name",
                "attr_value": "Garment Upper body",
                "pred_target_growth": 0.30,
                "heat_t": 1100.0,
            },
        ]
    )


def test_select_recommendation_cases_uses_test_split_and_prioritizes_hits() -> None:
    selected = select_recommendation_cases(
        recommendation_items=_recommendation_items(),
        evaluation_labels=_evaluation_labels(),
        user_profile=_user_profile(),
        case_count=1,
    )

    assert selected == [("000customer-high", "test", 103, 104)]


def test_select_recommendation_cases_requires_complete_explanation_values() -> None:
    items = _recommendation_items()
    items.loc[items["customer_id"] == "000customer-high", "candidate_sources"] = None

    selected = select_recommendation_cases(
        recommendation_items=items,
        evaluation_labels=_evaluation_labels(),
        user_profile=_user_profile(),
        case_count=1,
    )

    assert selected == [("000customer-low", "test", 103, 104)]


def test_select_recommendation_cases_requires_clear_preference_attr_type() -> None:
    profile = _user_profile()
    high_mask = profile["customer_id"] == "000customer-high"
    profile.loc[high_mask, "attr_type"] = "department_name"
    profile.loc[high_mask, "attr_value"] = "Divided"

    selected = select_recommendation_cases(
        recommendation_items=_recommendation_items(),
        evaluation_labels=_evaluation_labels(),
        user_profile=profile,
        case_count=1,
    )

    assert selected == [("000customer-low", "test", 103, 104)]


def test_select_recommendation_cases_fails_when_not_enough_cases() -> None:
    with pytest.raises(ValueError, match="不足 2 个推荐案例"):
        select_recommendation_cases(
            recommendation_items=_recommendation_items(),
            evaluation_labels=_evaluation_labels(),
            user_profile=_user_profile().iloc[0:0],
            case_count=2,
        )


def test_select_recommendation_cases_rejects_duplicate_top12_articles() -> None:
    items = _recommendation_items()
    duplicate_mask = (items["customer_id"] == "000customer-high") & (items["rank"] == 2)
    items.loc[duplicate_mask, "article_id"] = "0000000001"

    with pytest.raises(ValueError, match="Top-12.*重复"):
        select_recommendation_cases(
            recommendation_items=items,
            evaluation_labels=_evaluation_labels(),
            user_profile=_user_profile(),
            case_count=1,
        )


def test_build_case_payload_is_json_serializable_and_preserves_string_ids() -> None:
    payload = build_case_payload(
        case_key=("000customer-high", "test", 103, 104),
        recommendation_items=_recommendation_items(),
        evaluation_labels=_evaluation_labels(),
        user_profile=_user_profile(),
        article_attributes=_article_attributes(),
        representative_trends=_representative_trends(),
    )

    json.dumps(payload, ensure_ascii=False)
    assert payload["customer_id"] == "000customer-high"
    assert payload["window_id"] == "test:103:104"
    assert payload["hit_count"] == 2
    assert payload["profile"][0]["attr_value"] == "Black"
    assert payload["representative_trends"][0]["attr_value"] == "Black"
    assert payload["recommendations"][0]["article_id"] == "0000000001"
    assert payload["recommendations"][0]["is_hit"] is True
    assert payload["recommendations"][0]["score_decomposition"] == {
        "score": 0.99,
        "pop_score": 0.89,
        "sim_score": 0.79,
        "trend_score": 0.69,
        "recent_score": 0.59,
    }
    assert payload["recommendations"][0]["attributes"]["colour_group_name"] == "Black"


def test_build_case_payload_uses_case_cutoff_week_trends() -> None:
    items = _recommendation_items().assign(cutoff_week=102, label_week=103)
    labels = _evaluation_labels().assign(cutoff_week=102, label_week=103)
    profile = _user_profile().assign(cutoff_week=102, label_week=103)

    payload = build_case_payload(
        case_key=("000customer-high", "test", 102, 103),
        recommendation_items=items,
        evaluation_labels=labels,
        user_profile=profile,
        article_attributes=_article_attributes(),
        representative_trends=_representative_trends(),
    )

    assert payload["representative_trends"][0]["attr_value"] == "Red"


def test_render_case_markdown_includes_case_sections_and_article_attributes() -> None:
    payload = build_case_payload(
        case_key=("000customer-high", "test", 103, 104),
        recommendation_items=_recommendation_items(),
        evaluation_labels=_evaluation_labels(),
        user_profile=_user_profile(),
        article_attributes=_article_attributes(),
        representative_trends=_representative_trends(),
    )

    text = render_case_markdown(payload)

    assert "# 推荐案例" in text
    assert "## 用户偏好属性" in text
    assert "## 代表性趋势属性" in text
    assert "## 推荐商品与解释" in text
    assert "商品属性" in text
    assert "colour_group_name=Black" in text
