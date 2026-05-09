from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.inputs import (
    build_evaluation_labels,
    build_target_users,
    build_user_profile,
)


def test_target_users_require_history_and_label_purchase() -> None:
    windows = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2", "u3"],
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "week_id": [9, 11, 11, 9],
        }
    )

    target_users = build_target_users(transactions, windows)

    assert target_users.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "history_purchase_count": 1,
            "label_purchase_count": 1,
        }
    ]


def test_evaluation_labels_deduplicate_articles_per_user_window() -> None:
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
                "label_purchase_count": 2,
            }
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "article_id": ["0000000002", "0000000002", "0000000003"],
            "week_id": [11, 11, 10],
        }
    )

    labels = build_evaluation_labels(transactions, windows, target_users)

    assert labels.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "article_id": "0000000002",
        }
    ]


def test_user_profile_uses_history_before_or_at_cutoff_only() -> None:
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
                "history_purchase_count": 2,
                "label_purchase_count": 1,
            }
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "article_id": ["0000000001", "0000000002", "0000000003"],
            "week_id": [8, 10, 11],
        }
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002", "0000000003"],
            "attr_id": [101, 102, 103],
            "attr_type": ["product_type_name"] * 3,
            "attr_value": ["Dress", "Shirt", "Shoes"],
        }
    )

    profile = build_user_profile(transactions, article_attributes, windows, target_users)

    assert "article_id" not in profile.columns
    assert set(profile["attr_value"]) == {"Dress", "Shirt"}
    assert "Shoes" not in set(profile["attr_value"])


def test_user_profile_keeps_top_core_attributes_only() -> None:
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
                "history_purchase_count": 4,
                "label_purchase_count": 1,
            }
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1", "u1"],
            "article_id": [
                "0000000001",
                "0000000001",
                "0000000002",
                "0000000003",
            ],
            "week_id": [8, 9, 10, 10],
        }
    )
    article_attributes = pd.DataFrame(
        [
            {
                "article_id": "0000000001",
                "attr_id": 101,
                "attr_type": "product_type_name",
                "attr_value": "Dress",
            },
            {
                "article_id": "0000000002",
                "attr_id": 102,
                "attr_type": "colour_group_name",
                "attr_value": "Black",
            },
            {
                "article_id": "0000000003",
                "attr_id": 103,
                "attr_type": "garment_group_name",
                "attr_value": "Tops",
            },
            {
                "article_id": "0000000003",
                "attr_id": 999,
                "attr_type": "detail_desc",
                "attr_value": "Ignored",
            },
        ]
    )

    profile = build_user_profile(transactions, article_attributes, windows, target_users)

    assert profile["attr_value"].tolist() == ["Dress", "Black", "Tops"]
    assert "detail_desc" not in set(profile["attr_type"])
