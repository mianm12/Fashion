from __future__ import annotations

import json

import pandas as pd
import pytest

from fashion_trend.recommendation.retrieval import candidates as candidate_module
from fashion_trend.recommendation.retrieval.attributes import (
    build_attribute_similarity_candidates,
)
from fashion_trend.recommendation.retrieval.candidates import (
    build_and_write_candidate_items,
    build_candidate_items,
    build_source_frames_for_frames,
    validate_candidate_strategy,
)
from fashion_trend.recommendation.retrieval.popularity import (
    build_popularity_candidates,
)
from fashion_trend.recommendation.retrieval.trend import build_trend_candidates


def sample_window() -> pd.DataFrame:
    return pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])


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


def test_candidate_writer_records_input_fingerprints(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "transactions.parquet"
    input_path.write_text("transactions", encoding="utf-8")
    candidate_path = tmp_path / "candidate_items.parquet"
    monkeypatch.setattr(
        candidate_module,
        "candidate_items_path",
        lambda strategy: candidate_path,
    )

    output_path = build_and_write_candidate_items(
        strategy="popularity",
        transactions=pd.DataFrame(
            {
                "customer_id": ["u1"],
                "article_id": ["0000000001"],
                "week_id": [10],
            }
        ),
        article_attributes=None,
        trend_predictions=None,
        windows=sample_window(),
        target_users=sample_targets(),
        user_profile=None,
        input_paths={"weekly_transactions": str(input_path)},
    )
    metadata = json.loads(
        output_path.with_name("metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["strategy"] == "popularity"
    assert metadata["candidate_rows"] == 1
    assert metadata["input_artifacts"] == {"weekly_transactions": str(input_path)}
    assert metadata["input_fingerprints"]["weekly_transactions"]["size_bytes"] == (
        input_path.stat().st_size
    )


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


def test_similarity_candidates_use_core_profile_and_limited_article_matches() -> None:
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
                "purchase_count": 2,
                "last_purchase_week": 10,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "attr_id": 999,
                "attr_type": "detail_desc",
                "attr_value": "Ignored",
                "preference_score": 9.0,
                "purchase_count": 9,
                "last_purchase_week": 10,
            },
        ]
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": [
                "0000000003",
                "0000000001",
                "0000000002",
                "0000000999",
            ],
            "attr_type": [
                "product_type_name",
                "product_type_name",
                "product_type_name",
                "detail_desc",
            ],
            "attr_value": ["Dress", "Dress", "Dress", "Ignored"],
        }
    )

    candidates = build_attribute_similarity_candidates(
        user_profile,
        article_attributes,
        sample_window(),
        sample_targets(),
        top_n=2,
    )

    assert candidates["article_id"].tolist() == ["0000000001", "0000000002"]
    assert set(candidates["source"]) == {"similarity"}


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


def test_reorder_candidates_use_cutoff_history_and_stable_ranking() -> None:
    from fashion_trend.recommendation.retrieval.reorder import (
        build_reorder_candidates,
    )

    transactions = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "week_id": 9},
            {"customer_id": "u1", "article_id": "0000000002", "week_id": 9},
            {"customer_id": "u1", "article_id": "0000000004", "week_id": 9},
            {"customer_id": "u1", "article_id": "0000000003", "week_id": 10},
            {"customer_id": "u1", "article_id": "0000000001", "week_id": 10},
            {"customer_id": "u1", "article_id": "0000000999", "week_id": 11},
            {"customer_id": "other", "article_id": "0000000005", "week_id": 10},
        ]
    )

    candidates = build_reorder_candidates(
        transactions,
        sample_window(),
        sample_targets(),
        top_n=10,
    )

    assert candidates.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "article_id": "0000000001",
            "source": "reorder",
            "source_rank": 1,
        },
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "article_id": "0000000003",
            "source": "reorder",
            "source_rank": 2,
        },
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "article_id": "0000000002",
            "source": "reorder",
            "source_rank": 3,
        },
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "article_id": "0000000004",
            "source": "reorder",
            "source_rank": 4,
        },
    ]


def test_reorder_candidates_cap_each_user_window_at_top_12() -> None:
    from fashion_trend.recommendation.retrieval.reorder import (
        build_reorder_candidates,
    )

    transactions = pd.DataFrame(
        {
            "customer_id": ["u1"] * 15,
            "article_id": [f"{article_id:010d}" for article_id in range(1, 16)],
            "week_id": [10] * 15,
        }
    )

    candidates = build_reorder_candidates(
        transactions,
        sample_window(),
        sample_targets(),
    )

    assert len(candidates) == 12
    assert candidates["article_id"].tolist() == [
        f"{article_id:010d}" for article_id in range(1, 13)
    ]
    assert candidates["source_rank"].tolist() == list(range(1, 13))


def test_product_variant_candidates_use_reorder_top_6_seeds() -> None:
    from fashion_trend.recommendation.retrieval.product_variants import (
        build_product_variant_candidates,
    )

    reorder_candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "article_id": f"{article_id:010d}",
                "source": "reorder",
                "source_rank": article_id,
            }
            for article_id in range(1, 8)
        ]
    )
    article_product_map = pd.DataFrame(
        [
            {"article_id": "0000000001", "product_code": "p1"},
            {"article_id": "0000000101", "product_code": "p1"},
            {"article_id": "0000000002", "product_code": "p2"},
            {"article_id": "0000000201", "product_code": "p2"},
            {"article_id": "0000000006", "product_code": "p6"},
            {"article_id": "0000000601", "product_code": "p6"},
            {"article_id": "0000000007", "product_code": "p7"},
            {"article_id": "0000000701", "product_code": "p7"},
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["x"] * 16,
            "article_id": (
                ["0000000201"] * 3
                + ["0000000101"] * 2
                + ["0000000601"]
                + ["0000000701"] * 10
            ),
            "week_id": [10] * 16,
        }
    )

    candidates = build_product_variant_candidates(
        reorder_candidates,
        transactions,
        article_product_map,
        sample_window(),
    )

    assert candidates["article_id"].tolist() == [
        "0000000201",
        "0000000101",
        "0000000601",
    ]
    assert "0000000701" not in set(candidates["article_id"])
    assert set(candidates["source"]) == {"product_variant"}
    assert candidates["source_rank"].tolist() == [1, 2, 3]


def test_product_variant_candidates_skip_self_and_missing_product_code() -> None:
    from fashion_trend.recommendation.retrieval.product_variants import (
        build_product_variant_candidates,
    )

    reorder_candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "article_id": "0000000001",
                "source": "reorder",
                "source_rank": 1,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "article_id": "0000000002",
                "source": "reorder",
                "source_rank": 2,
            },
        ]
    )
    article_product_map = pd.DataFrame(
        [
            {"article_id": "0000000001", "product_code": "p1"},
            {"article_id": "0000000009", "product_code": "p1"},
            {"article_id": "0000000002", "product_code": None},
            {"article_id": "0000000008", "product_code": None},
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["x", "x", "x"],
            "article_id": ["0000000001", "0000000008", "0000000009"],
            "week_id": [10, 10, 10],
        }
    )

    candidates = build_product_variant_candidates(
        reorder_candidates,
        transactions,
        article_product_map,
        sample_window(),
    )

    assert candidates["article_id"].tolist() == ["0000000009"]
    assert candidates["source_rank"].tolist() == [1]


def test_product_variant_candidates_rank_each_user_window_independently() -> None:
    from fashion_trend.recommendation.retrieval.product_variants import (
        build_product_variant_candidates,
    )

    windows = pd.DataFrame(
        [
            {"split": "valid", "cutoff_week": 10, "label_week": 11},
            {"split": "test", "cutoff_week": 12, "label_week": 13},
        ]
    )
    reorder_candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "article_id": "0000000001",
                "source": "reorder",
                "source_rank": 1,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u2",
                "article_id": "0000000002",
                "source": "reorder",
                "source_rank": 1,
            },
            {
                "split": "test",
                "cutoff_week": 12,
                "label_week": 13,
                "customer_id": "u1",
                "article_id": "0000000003",
                "source": "reorder",
                "source_rank": 1,
            },
        ]
    )
    article_product_map = pd.DataFrame(
        [
            {"article_id": "0000000001", "product_code": "p1"},
            {"article_id": "0000000101", "product_code": "p1"},
            {"article_id": "0000000002", "product_code": "p2"},
            {"article_id": "0000000201", "product_code": "p2"},
            {"article_id": "0000000003", "product_code": "p3"},
            {"article_id": "0000000301", "product_code": "p3"},
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["x", "x", "x"],
            "article_id": ["0000000101", "0000000201", "0000000301"],
            "week_id": [10, 10, 12],
        }
    )

    candidates = build_product_variant_candidates(
        reorder_candidates,
        transactions,
        article_product_map,
        windows,
        top_n=1,
    )

    assert candidates.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "article_id": "0000000101",
            "source": "product_variant",
            "source_rank": 1,
        },
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u2",
            "article_id": "0000000201",
            "source": "product_variant",
            "source_rank": 1,
        },
        {
            "split": "test",
            "cutoff_week": 12,
            "label_week": 13,
            "customer_id": "u1",
            "article_id": "0000000301",
            "source": "product_variant",
            "source_rank": 1,
        },
    ]


def test_enhanced_source_order_rejects_unknown_source() -> None:
    assert candidate_module.SOURCE_ORDER == {
        "popularity": 0,
        "similarity": 1,
        "trend": 2,
        "reorder": 3,
        "product_variant": 4,
        "age_popularity": 5,
        "preference_popularity": 6,
    }

    source = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "article_id": "0000000001",
                "source": "product_variant",
                "source_rank": 1,
            }
        ]
    )
    candidates = build_candidate_items(strategy="default", source_frames=[source])
    assert candidates["candidate_sources"].tolist() == ["product_variant"]

    with pytest.raises(ValueError, match="未知候选 source"):
        build_candidate_items(
            strategy="default",
            source_frames=[source.assign(source="missing_source")],
        )


def test_age_popularity_uses_age_bucket_and_recent_four_weeks() -> None:
    from fashion_trend.recommendation.retrieval.customer_segments import (
        build_age_popularity_candidates,
    )

    customer_profile = pd.DataFrame(
        [
            {"customer_id": "target-20", "age_bucket": "20-29"},
            {"customer_id": "target-30", "age_bucket": "30-39"},
            {"customer_id": "hist-20-a", "age_bucket": "20-29"},
            {"customer_id": "hist-20-b", "age_bucket": "20-29"},
            {"customer_id": "hist-30", "age_bucket": "30-39"},
        ]
    )
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "target-20",
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "target-30",
            },
        ]
    )
    transactions = pd.DataFrame(
        [
            {"customer_id": "hist-20-a", "article_id": "0000000001", "week_id": 7},
            {"customer_id": "hist-20-b", "article_id": "0000000001", "week_id": 10},
            {"customer_id": "hist-20-a", "article_id": "0000000002", "week_id": 6},
            {"customer_id": "hist-20-b", "article_id": "0000000003", "week_id": 8},
            {"customer_id": "hist-30", "article_id": "0000000099", "week_id": 10},
            {"customer_id": "hist-30", "article_id": "0000000098", "week_id": 11},
        ]
    )

    candidates = build_age_popularity_candidates(
        transactions,
        customer_profile,
        sample_window(),
        target_users,
    )

    assert set(candidates["source"]) == {"age_popularity"}
    assert candidates.loc[
        candidates["customer_id"] == "target-20", "article_id"
    ].tolist() == ["0000000001", "0000000003"]
    assert candidates.loc[
        candidates["customer_id"] == "target-20", "source_rank"
    ].tolist() == [1, 2]
    assert candidates.loc[
        candidates["customer_id"] == "target-30", "article_id"
    ].tolist() == ["0000000099"]


def test_age_popularity_does_not_backfill_global_popularity() -> None:
    from fashion_trend.recommendation.retrieval.customer_segments import (
        build_age_popularity_candidates,
    )

    customer_profile = pd.DataFrame(
        [{"customer_id": "target", "age_bucket": "60+"}]
        + [{"customer_id": "global", "age_bucket": "20-29"}]
        + [{"customer_id": "senior", "age_bucket": "60+"}]
    )
    transactions = pd.DataFrame(
        [
            {"customer_id": "senior", "article_id": "0000000001", "week_id": 10},
            *[
                {
                    "customer_id": "global",
                    "article_id": f"{article_id:010d}",
                    "week_id": 10,
                }
                for article_id in range(2, 15)
            ],
        ]
    )
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "target",
            }
        ]
    )

    candidates = build_age_popularity_candidates(
        transactions,
        customer_profile,
        sample_window(),
        target_users,
    )

    assert set(candidates["source"]) == {"age_popularity"}
    assert candidates["article_id"].tolist() == ["0000000001"]
    assert candidates["source_rank"].tolist() == [1]


def test_preference_popularity_uses_top_3_core_attributes() -> None:
    from fashion_trend.recommendation.retrieval.preference_popularity import (
        build_preference_popularity_candidates,
    )

    user_profile = pd.DataFrame(
        [
            _profile_row("u1", "product_type_name", "dress", 0.9),
            _profile_row("u1", "colour_group_name", "black", 0.8),
            _profile_row("u1", "garment_group_name", "jersey", 0.7),
            _profile_row("u1", "product_group_name", "tops", 0.6),
        ]
    )
    article_attributes = pd.DataFrame(
        [
            {
                "article_id": "0000000001",
                "attr_type": "product_type_name",
                "attr_value": "dress",
            },
            {
                "article_id": "0000000002",
                "attr_type": "colour_group_name",
                "attr_value": "black",
            },
            {
                "article_id": "0000000003",
                "attr_type": "garment_group_name",
                "attr_value": "jersey",
            },
            {
                "article_id": "0000000004",
                "attr_type": "product_group_name",
                "attr_value": "tops",
            },
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["buyer"] * 4,
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
            ],
            "week_id": [10, 10, 10, 10],
        }
    )

    candidates = build_preference_popularity_candidates(
        transactions,
        article_attributes,
        user_profile,
        sample_window(),
        sample_targets(),
    )

    assert set(candidates["source"]) == {"preference_popularity"}
    assert candidates["article_id"].tolist() == [
        "0000000001",
        "0000000002",
        "0000000003",
    ]
    assert "0000000004" not in set(candidates["article_id"])
    assert candidates["source_rank"].tolist() == [1, 2, 3]


def test_preference_popularity_caps_each_attribute_at_top_4_and_window_at_top_12() -> (
    None
):
    from fashion_trend.recommendation.retrieval.preference_popularity import (
        build_preference_popularity_candidates,
    )

    attrs = [
        ("product_type_name", "dress", 0.9, range(1, 6)),
        ("colour_group_name", "black", 0.8, range(11, 16)),
        ("garment_group_name", "jersey", 0.7, range(21, 26)),
    ]
    user_profile = pd.DataFrame(
        [
            _profile_row("u1", attr_type, attr_value, score)
            for attr_type, attr_value, score, _ in attrs
        ]
    )
    article_attributes = pd.DataFrame(
        [
            {
                "article_id": f"{article_id:010d}",
                "attr_type": attr_type,
                "attr_value": attr_value,
            }
            for attr_type, attr_value, _, article_ids in attrs
            for article_id in article_ids
        ]
    )
    transactions = pd.DataFrame(
        [
            {
                "customer_id": f"buyer-{article_id}",
                "article_id": f"{article_id:010d}",
                "week_id": 10,
            }
            for _, _, _, article_ids in attrs
            for article_id in article_ids
        ]
    )

    candidates = build_preference_popularity_candidates(
        transactions,
        article_attributes,
        user_profile,
        sample_window(),
        sample_targets(),
    )

    assert set(candidates["source"]) == {"preference_popularity"}
    assert len(candidates) == 12
    assert candidates["article_id"].tolist() == [
        "0000000001",
        "0000000002",
        "0000000003",
        "0000000004",
        "0000000011",
        "0000000012",
        "0000000013",
        "0000000014",
        "0000000021",
        "0000000022",
        "0000000023",
        "0000000024",
    ]
    assert candidates["source_rank"].tolist() == list(range(1, 13))


def test_preference_popularity_ignores_non_core_attributes() -> None:
    from fashion_trend.recommendation.retrieval.preference_popularity import (
        build_preference_popularity_candidates,
    )

    user_profile = pd.DataFrame(
        [
            _profile_row("u1", "detail_desc", "ignored", 9.0),
            _profile_row("u1", "product_type_name", "dress", 0.5),
        ]
    )
    article_attributes = pd.DataFrame(
        [
            {
                "article_id": "0000000099",
                "attr_type": "detail_desc",
                "attr_value": "ignored",
            },
            {
                "article_id": "0000000001",
                "attr_type": "product_type_name",
                "attr_value": "dress",
            },
        ]
    )
    transactions = pd.DataFrame(
        [
            {"customer_id": "buyer", "article_id": "0000000099", "week_id": 10},
            {"customer_id": "buyer", "article_id": "0000000001", "week_id": 10},
        ]
    )

    candidates = build_preference_popularity_candidates(
        transactions,
        article_attributes,
        user_profile,
        sample_window(),
        sample_targets(),
    )

    assert set(candidates["source"]) == {"preference_popularity"}
    assert candidates["article_id"].tolist() == ["0000000001"]


def _profile_row(
    customer_id: str,
    attr_type: str,
    attr_value: str,
    preference_score: float,
) -> dict[str, object]:
    return {
        "split": "valid",
        "cutoff_week": 10,
        "label_week": 11,
        "customer_id": customer_id,
        "attr_id": f"{attr_type}:{attr_value}",
        "attr_type": attr_type,
        "attr_value": attr_value,
        "preference_score": preference_score,
        "purchase_count": 1,
        "last_purchase_week": 10,
    }
