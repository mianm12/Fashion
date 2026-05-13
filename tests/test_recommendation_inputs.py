from __future__ import annotations

import json

import pandas as pd
import pytest

from fashion_trend.recommendation import inputs as recommendation_inputs
from fashion_trend.recommendation.inputs import (
    build_and_write_recommendation_inputs,
    build_evaluation_labels,
    build_target_users,
    build_user_profile,
)


def test_target_users_require_history_and_label_purchase() -> None:
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
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
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
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
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
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

    profile = build_user_profile(
        transactions, article_attributes, windows, target_users
    )

    assert "article_id" not in profile.columns
    assert set(profile["attr_value"]) == {"Dress", "Shirt"}
    assert "Shoes" not in set(profile["attr_value"])


def test_user_profile_keeps_top_core_attributes_only() -> None:
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
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

    profile = build_user_profile(
        transactions, article_attributes, windows, target_users
    )

    assert profile["attr_value"].tolist() == ["Dress", "Black", "Tops"]
    assert "detail_desc" not in set(profile["attr_type"])


def test_customer_profile_buckets_age_and_preserves_unknowns() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": [
                "0000001",
                "0000002",
                "0000003",
                "0000004",
                "0000005",
                "0000006",
                "0000007",
            ],
            "age": ["not-a-number", 19, 20, 30, 40, 50, 60],
            "club_member_status": [
                None,
                "ACTIVE",
                None,
                "PRE-CREATE",
                None,
                None,
                None,
            ],
            "fashion_news_frequency": [
                None,
                "Regularly",
                None,
                None,
                "Monthly",
                None,
                None,
            ],
        }
    )

    profile = recommendation_inputs.build_customer_profile(customers)

    assert profile.columns.tolist() == [
        "customer_id",
        "age",
        "age_bucket",
        "club_member_status",
        "fashion_news_frequency",
    ]
    assert profile["customer_id"].astype(str).tolist()[0] == "0000001"
    assert pd.isna(profile.loc[0, "age"])
    assert profile["age_bucket"].tolist() == [
        "unknown",
        "0-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60+",
    ]
    assert profile["club_member_status"].tolist()[0] == "unknown"
    assert profile["fashion_news_frequency"].tolist()[0] == "unknown"


def test_customer_profile_rejects_duplicate_customer_id() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["0000001", "0000001"],
            "age": [20, 21],
            "club_member_status": ["ACTIVE", "ACTIVE"],
            "fashion_news_frequency": ["Regularly", "Regularly"],
        }
    )

    with pytest.raises(ValueError, match="重复"):
        recommendation_inputs.build_customer_profile(customers)


def test_article_product_map_preserves_string_ids_and_rejects_missing_product_code() -> (
    None
):
    clean_articles = pd.DataFrame(
        {
            "article_id": ["0000000001"],
            "product_code": ["001234"],
        }
    )

    product_map = recommendation_inputs.build_article_product_map(clean_articles)

    assert product_map.columns.tolist() == ["article_id", "product_code"]
    assert product_map["article_id"].astype(str).tolist() == ["0000000001"]
    assert product_map["product_code"].astype(str).tolist() == ["001234"]

    with pytest.raises(ValueError, match="product_code"):
        recommendation_inputs.build_article_product_map(
            pd.DataFrame(
                {
                    "article_id": ["0000000002"],
                    "product_code": [None],
                }
            )
        )


def test_build_and_write_inputs_records_upstream_metadata(
    tmp_path, monkeypatch
) -> None:
    time_windows_path = tmp_path / "time_windows.parquet"
    target_users_path = tmp_path / "target_users.parquet"
    evaluation_labels_path = tmp_path / "evaluation_labels.parquet"
    user_profile_path = tmp_path / "user_profile.parquet"
    metadata_path = tmp_path / "metadata.json"
    monkeypatch.setattr(recommendation_inputs, "TIME_WINDOWS_PATH", time_windows_path)
    monkeypatch.setattr(recommendation_inputs, "TARGET_USERS_PATH", target_users_path)
    monkeypatch.setattr(
        recommendation_inputs,
        "EVALUATION_LABELS_PATH",
        evaluation_labels_path,
    )
    monkeypatch.setattr(recommendation_inputs, "USER_PROFILE_PATH", user_profile_path)
    monkeypatch.setattr(recommendation_inputs, "RECOMMEND_METADATA_PATH", metadata_path)
    upstream_paths = {
        "weekly_transactions": str(tmp_path / "weekly_transactions.parquet"),
        "article_attributes": str(tmp_path / "article_attributes.csv"),
        "trend_predictions": str(tmp_path / "predictions.csv"),
    }
    for name, path in upstream_paths.items():
        pd.DataFrame({"source": [name]}).to_csv(path, index=False)

    build_and_write_recommendation_inputs(
        transactions=pd.DataFrame(
            {
                "customer_id": ["u1", "u1", "u2", "u2"],
                "article_id": [
                    "0000000001",
                    "0000000002",
                    "0000000001",
                    "0000000003",
                ],
                "week_id": [10, 11, 11, 12],
            }
        ),
        article_attributes=pd.DataFrame(
            {
                "article_id": ["0000000001", "0000000002", "0000000003"],
                "attr_id": [101, 102, 103],
                "attr_type": ["product_type_name"] * 3,
                "attr_value": ["Dress", "Shirt", "Coat"],
            }
        ),
        trend_predictions=pd.DataFrame(
            {
                "split": ["valid", "test"],
                "week_id": [10, 11],
            }
        ),
        input_paths=upstream_paths,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["input_artifacts"] == upstream_paths
    assert set(metadata["input_fingerprints"]) == {
        "weekly_transactions",
        "article_attributes",
        "trend_predictions",
    }
    assert metadata["output_artifacts"] == {
        "time_windows": str(time_windows_path),
        "target_users": str(target_users_path),
        "evaluation_labels": str(evaluation_labels_path),
        "user_profile": str(user_profile_path),
    }


def test_build_and_write_inputs_records_customer_and_product_artifacts(
    tmp_path, monkeypatch
) -> None:
    time_windows_path = tmp_path / "time_windows.parquet"
    target_users_path = tmp_path / "target_users.parquet"
    evaluation_labels_path = tmp_path / "evaluation_labels.parquet"
    user_profile_path = tmp_path / "user_profile.parquet"
    customer_profile_path = tmp_path / "customer_profile.parquet"
    article_product_map_path = tmp_path / "article_product_map.parquet"
    metadata_path = tmp_path / "metadata.json"
    monkeypatch.setattr(recommendation_inputs, "TIME_WINDOWS_PATH", time_windows_path)
    monkeypatch.setattr(recommendation_inputs, "TARGET_USERS_PATH", target_users_path)
    monkeypatch.setattr(
        recommendation_inputs,
        "EVALUATION_LABELS_PATH",
        evaluation_labels_path,
    )
    monkeypatch.setattr(recommendation_inputs, "USER_PROFILE_PATH", user_profile_path)
    monkeypatch.setattr(
        recommendation_inputs,
        "CUSTOMER_PROFILE_PATH",
        customer_profile_path,
    )
    monkeypatch.setattr(
        recommendation_inputs,
        "ARTICLE_PRODUCT_MAP_PATH",
        article_product_map_path,
    )
    monkeypatch.setattr(recommendation_inputs, "RECOMMEND_METADATA_PATH", metadata_path)
    upstream_paths = {
        "weekly_transactions": str(tmp_path / "weekly_transactions.parquet"),
        "article_attributes": str(tmp_path / "article_attributes.csv"),
        "trend_predictions": str(tmp_path / "predictions.csv"),
        "raw_customers": str(tmp_path / "customers.csv"),
        "clean_articles": str(tmp_path / "articles_clean.csv"),
    }
    for name, path in upstream_paths.items():
        pd.DataFrame({"source": [name]}).to_csv(path, index=False)

    artifacts = build_and_write_recommendation_inputs(
        transactions=pd.DataFrame(
            {
                "customer_id": ["u1", "u1", "u2", "u2"],
                "article_id": [
                    "0000000001",
                    "0000000002",
                    "0000000001",
                    "0000000003",
                ],
                "week_id": [10, 11, 11, 12],
            }
        ),
        article_attributes=pd.DataFrame(
            {
                "article_id": ["0000000001", "0000000002", "0000000003"],
                "attr_id": [101, 102, 103],
                "attr_type": ["product_type_name"] * 3,
                "attr_value": ["Dress", "Shirt", "Coat"],
            }
        ),
        trend_predictions=pd.DataFrame(
            {
                "split": ["valid", "test"],
                "week_id": [10, 11],
            }
        ),
        input_paths=upstream_paths,
        customers=pd.DataFrame(
            {
                "customer_id": ["u1", "u2"],
                "age": [20, None],
                "club_member_status": ["ACTIVE", None],
                "fashion_news_frequency": ["Regularly", None],
            }
        ),
        clean_articles=pd.DataFrame(
            {
                "article_id": ["0000000001", "0000000002", "0000000003"],
                "product_code": ["001", "002", "003"],
            }
        ),
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert customer_profile_path.exists()
    assert article_product_map_path.exists()
    assert artifacts.customer_profile.columns.tolist() == [
        "customer_id",
        "age",
        "age_bucket",
        "club_member_status",
        "fashion_news_frequency",
    ]
    assert artifacts.article_product_map.columns.tolist() == [
        "article_id",
        "product_code",
    ]
    assert metadata["output_artifacts"]["customer_profile"] == str(
        customer_profile_path
    )
    assert metadata["output_artifacts"]["article_product_map"] == str(
        article_product_map_path
    )
    assert metadata["row_counts"]["customer_profile"] == 2
    assert metadata["row_counts"]["article_product_map"] == 3
    assert metadata["config"]["customer_profile_schema_version"] == 1
    assert metadata["config"]["article_product_map_schema_version"] == 1
    assert metadata["config"]["customer_age_bucket_algorithm_version"] == (
        "customer-age-buckets-v1"
    )
    assert metadata["input_artifacts"] == upstream_paths
    assert set(metadata["input_fingerprints"]) == set(upstream_paths)
