from __future__ import annotations

import json

import pandas as pd
import pytest

from fashion_trend.recommendation.features.cache import (
    FEATURE_NAMES,
    build_and_write_feature_cache_for_strategy,
    build_candidate_seen_flags,
    build_recommendable_pool,
    read_recommendable_pool_cache,
    recommendable_pool_cache_fresh,
    update_feature_cache_manifest,
    write_recommendable_pool_cache,
)
from fashion_trend.recommendation.paths import (
    FEATURE_CACHE_METADATA_PATH,
    feature_cache_partition_path,
)
from fashion_trend.recommendation.ranking import filters as filter_module


def test_feature_cache_partition_path_is_strategy_window_scoped() -> None:
    path = feature_cache_partition_path(
        "candidate_seen_flags",
        strategy="default",
        split="valid",
        cutoff_week=104,
    )

    assert "candidate_seen_flags" in FEATURE_NAMES
    assert path == (
        FEATURE_CACHE_METADATA_PATH.parent
        / "candidate_seen_flags"
        / "strategy=default"
        / "split=valid"
        / "cutoff_week=104"
        / "part.parquet"
    )


def test_candidate_seen_flags_only_contains_seen_candidate_pairs() -> None:
    candidates = pd.DataFrame(
        {
            "split": ["valid", "valid", "valid", "test"],
            "cutoff_week": [10, 10, 10, 12],
            "label_week": [11, 11, 11, 13],
            "strategy": ["default", "default", "default", "popularity"],
            "customer_id": ["1", "1", "2", "3"],
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
        }
    )
    transactions = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3, 9],
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
                "0000000005",
            ],
            "week_id": [10, 11, 9, 13, 10],
        }
    )

    result = build_candidate_seen_flags(candidates, transactions)

    assert result.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "strategy": "default",
            "customer_id": "1",
            "article_id": "0000000001",
            "seen": True,
        },
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "strategy": "default",
            "customer_id": "2",
            "article_id": "0000000003",
            "seen": True,
        },
    ]
    assert len(result) < len(candidates)


def test_candidate_seen_flags_deduplicates_candidate_keys() -> None:
    candidates = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "strategy": ["default", "default"],
            "customer_id": ["1", "1"],
            "article_id": ["0000000001", "0000000001"],
        }
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["1"],
            "article_id": ["0000000001"],
            "week_id": [10],
        }
    )

    result = build_candidate_seen_flags(candidates, transactions)

    assert result.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "strategy": "default",
            "customer_id": "1",
            "article_id": "0000000001",
            "seen": True,
        }
    ]


def test_source_level_seen_filter_keeps_seen_reorder_and_filters_other_seen_sources() -> (
    None
):
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "1",
                "article_id": "0000000001",
                "is_seen": True,
                "allow_seen": False,
                "has_reorder_source": False,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "1",
                "article_id": "0000000002",
                "is_seen": True,
                "allow_seen": True,
                "has_reorder_source": True,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "1",
                "article_id": "0000000003",
                "is_seen": False,
                "allow_seen": False,
                "has_reorder_source": False,
            },
        ]
    )

    assert hasattr(filter_module, "filter_seen_items_by_source_policy")

    result = filter_module.filter_seen_items_by_source_policy(candidates)

    assert result["article_id"].tolist() == ["0000000002", "0000000003"]


def test_candidate_seen_flags_include_is_seen_allow_seen_and_has_reorder_source() -> (
    None
):
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "1",
                "article_id": "0000000001",
                "allow_seen": True,
                "has_reorder_source": True,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "1",
                "article_id": "0000000002",
                "allow_seen": False,
                "has_reorder_source": False,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "1",
                "article_id": "0000000003",
                "allow_seen": False,
                "has_reorder_source": False,
            },
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["1", "1"],
            "article_id": ["0000000001", "0000000002"],
            "week_id": [10, 9],
        }
    )

    result = build_candidate_seen_flags(candidates, transactions)

    assert result.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "strategy": "enhanced_default",
            "customer_id": "1",
            "article_id": "0000000001",
            "seen": True,
            "is_seen": True,
            "allow_seen": True,
            "has_reorder_source": True,
        },
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "strategy": "enhanced_default",
            "customer_id": "1",
            "article_id": "0000000002",
            "seen": True,
            "is_seen": True,
            "allow_seen": False,
            "has_reorder_source": False,
        },
    ]


def test_recommendable_pool_uses_cutoff_history_only() -> None:
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
    transactions = pd.DataFrame(
        {
            "article_id": ["a1", "a2"],
            "week_id": [10, 11],
        }
    )

    pool = build_recommendable_pool(transactions, windows)

    assert pool.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "article_id": "a1",
        }
    ]


def test_recommendable_pool_cache_round_trips_and_detects_stale_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_recommendable_pool_cache_paths(tmp_path, monkeypatch)
    input_path = tmp_path / "weekly_transactions.parquet"
    input_path.write_text("v1", encoding="utf-8")
    windows = pd.DataFrame(
        [
            {"split": "valid", "cutoff_week": 10, "label_week": 11},
            {"split": "test", "cutoff_week": 12, "label_week": 13},
        ]
    )
    transactions = pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002", "0000000003"],
            "week_id": [9, 11, 13],
        }
    )
    input_artifacts = {"weekly_transactions": str(input_path)}

    write_recommendable_pool_cache(
        transactions=transactions,
        windows=windows,
        input_artifacts=input_artifacts,
    )
    result = read_recommendable_pool_cache(windows)

    assert result.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "article_id": "0000000001",
        },
        {
            "split": "test",
            "cutoff_week": 12,
            "label_week": 13,
            "article_id": "0000000001",
        },
        {
            "split": "test",
            "cutoff_week": 12,
            "label_week": 13,
            "article_id": "0000000002",
        },
    ]
    assert recommendable_pool_cache_fresh(
        windows=windows,
        input_artifacts=input_artifacts,
    )

    input_path.write_text("v2", encoding="utf-8")

    assert not recommendable_pool_cache_fresh(
        windows=windows,
        input_artifacts=input_artifacts,
    )


def test_recommendable_pool_cache_non_object_metadata_is_stale(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_recommendable_pool_cache_paths(tmp_path, monkeypatch)
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
    input_path = tmp_path / "weekly_transactions.parquet"
    input_path.write_text("v1", encoding="utf-8")
    write_recommendable_pool_cache(
        transactions=pd.DataFrame({"article_id": ["0000000001"], "week_id": [10]}),
        windows=windows,
        input_artifacts={"weekly_transactions": str(input_path)},
    )
    metadata_path = (
        tmp_path
        / "features"
        / "recommendable_pool"
        / "strategy=all"
        / "split=valid"
        / "cutoff_week=10"
        / "metadata.json"
    )
    metadata_path.write_text("[]", encoding="utf-8")

    assert not recommendable_pool_cache_fresh(
        windows=windows,
        input_artifacts={"weekly_transactions": str(input_path)},
    )


def test_recommendable_pool_cache_manifest_merges_existing_entries(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_recommendable_pool_cache_paths(tmp_path, monkeypatch)
    manifest_path = tmp_path / "features" / "metadata.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"entries": {"strategy:default": {"feature_count": 5}}}),
        encoding="utf-8",
    )
    input_path = tmp_path / "weekly_transactions.parquet"
    input_path.write_text("transactions", encoding="utf-8")

    manifest = write_recommendable_pool_cache(
        transactions=pd.DataFrame({"article_id": ["0000000001"], "week_id": [10]}),
        windows=pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}]),
        input_artifacts={"weekly_transactions": str(input_path)},
    )

    assert "strategy:default" in manifest["entries"]
    assert "feature:recommendable_pool:strategy:all" in manifest["entries"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_feature_cache_manifest_merges_entries_without_overwriting(
    tmp_path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "features" / "metadata.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "entries": {
                    "strategy:default": {"feature_count": 2},
                    "strategy:similarity": {"feature_count": 1},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.FEATURE_CACHE_METADATA_PATH",
        manifest_path,
    )

    result = update_feature_cache_manifest(
        "strategy:default",
        {"feature_count": 6, "strategy": "default"},
    )

    assert result["entries"] == {
        "strategy:default": {"feature_count": 6, "strategy": "default"},
        "strategy:similarity": {"feature_count": 1},
    }
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored == result


def test_build_feature_cache_rejects_mixed_strategy_candidates() -> None:
    candidates = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "strategy": ["default", "similarity"],
            "customer_id": ["1", "1"],
            "article_id": ["0000000001", "0000000002"],
        }
    )

    with pytest.raises(ValueError, match="strategy"):
        build_and_write_feature_cache_for_strategy(
            strategy="default",
            candidates=candidates,
            transactions=_transactions(),
            article_attributes=_article_attributes(),
            user_profile=None,
            trend_predictions=None,
        )


def test_build_feature_cache_writes_partition_metadata_and_global_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    feature_root = tmp_path / "features"
    manifest_path = feature_root / "metadata.json"

    def partition_path(
        feature_name: str,
        *,
        strategy: str,
        split: str,
        cutoff_week: int,
    ):
        return (
            feature_root
            / feature_name
            / f"strategy={strategy}"
            / f"split={split}"
            / f"cutoff_week={int(cutoff_week)}"
            / "part.parquet"
        )

    def partition_metadata_path(
        feature_name: str,
        *,
        strategy: str,
        split: str,
        cutoff_week: int,
    ):
        return partition_path(
            feature_name,
            strategy=strategy,
            split=split,
            cutoff_week=cutoff_week,
        ).with_name("metadata.json")

    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.FEATURE_CACHE_METADATA_PATH",
        manifest_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.feature_cache_partition_path",
        partition_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache."
        "feature_cache_partition_metadata_path",
        partition_metadata_path,
    )

    manifest = build_and_write_feature_cache_for_strategy(
        strategy="default",
        candidates=_candidates(),
        transactions=_transactions(),
        article_attributes=_article_attributes(),
        user_profile=_user_profile(),
        trend_predictions=None,
        input_paths={"weekly_transactions": "transactions.parquet"},
    )

    entry = manifest["entries"]["strategy:default"]
    for key in (
        "input_artifacts",
        "input_fingerprints",
        "output_artifacts",
        "schema_version",
        "algorithm_version",
        "config",
        "row_counts",
    ):
        assert key in entry

    written_features = set(entry["manifest"]["partitions"])
    assert "recommendable_pool" not in written_features
    assert written_features == {
        "popularity_scores",
        "recent_scores",
        "similarity_scores",
        "trend_scores",
        "candidate_seen_flags",
    }
    assert entry["output_artifacts"]["feature_cache_metadata"] == str(manifest_path)

    for feature_name, partitions in entry["manifest"]["partitions"].items():
        assert len(partitions) == 1
        partition = partitions[0]
        parquet_path = partition_path(
            feature_name,
            strategy="default",
            split="valid",
            cutoff_week=10,
        )
        metadata_path = parquet_path.with_name("metadata.json")
        assert parquet_path.exists()
        assert metadata_path.exists()
        assert str(parquet_path) in entry["output_artifacts"].values()
        assert str(metadata_path) in entry["output_artifacts"].values()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["output_artifacts"] == {
            "partition": str(parquet_path),
            "partition_metadata": str(metadata_path),
        }
        assert partition["path"] == str(parquet_path)
        assert partition["metadata_path"] == str(metadata_path)
        assert partition["rows"] == len(pd.read_parquet(parquet_path))

    popularity = pd.read_parquet(
        partition_path(
            "popularity_scores",
            strategy="default",
            split="valid",
            cutoff_week=10,
        )
    )
    similarity = pd.read_parquet(
        partition_path(
            "similarity_scores",
            strategy="default",
            split="valid",
            cutoff_week=10,
        )
    )
    seen_flags = pd.read_parquet(
        partition_path(
            "candidate_seen_flags",
            strategy="default",
            split="valid",
            cutoff_week=10,
        )
    )

    assert popularity.columns.tolist() == [
        "split",
        "cutoff_week",
        "label_week",
        "strategy",
        "article_id",
        "pop_score",
    ]
    assert len(popularity) == 2
    assert similarity.columns.tolist() == [
        "split",
        "cutoff_week",
        "label_week",
        "strategy",
        "customer_id",
        "article_id",
        "sim_score",
    ]
    assert len(similarity) == len(_candidates())
    assert len(seen_flags) <= len(_candidates())


def test_feature_cache_partition_metadata_records_feature_specific_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    feature_root = tmp_path / "features"
    manifest_path = feature_root / "metadata.json"

    def partition_path(
        feature_name: str,
        *,
        strategy: str,
        split: str,
        cutoff_week: int,
    ):
        return (
            feature_root
            / feature_name
            / f"strategy={strategy}"
            / f"split={split}"
            / f"cutoff_week={int(cutoff_week)}"
            / "part.parquet"
        )

    def partition_metadata_path(
        feature_name: str,
        *,
        strategy: str,
        split: str,
        cutoff_week: int,
    ):
        return partition_path(
            feature_name,
            strategy=strategy,
            split=split,
            cutoff_week=cutoff_week,
        ).with_name("metadata.json")

    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.FEATURE_CACHE_METADATA_PATH",
        manifest_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.feature_cache_partition_path",
        partition_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache."
        "feature_cache_partition_metadata_path",
        partition_metadata_path,
    )

    build_and_write_feature_cache_for_strategy(
        strategy="similarity",
        candidates=_similarity_candidates(),
        transactions=_transactions(),
        article_attributes=_article_attributes(),
        user_profile=_user_profile(),
        trend_predictions=None,
        input_paths={
            "weekly_transactions": str(tmp_path / "weekly.parquet"),
            "article_attributes": str(tmp_path / "article_attributes.csv"),
            "trend_predictions": str(tmp_path / "predictions.csv"),
            "time_windows": str(tmp_path / "time_windows.parquet"),
            "target_users": str(tmp_path / "target_users.parquet"),
            "user_profile": str(tmp_path / "user_profile.parquet"),
            "candidate_items": str(tmp_path / "candidate_items.parquet"),
            "candidate_metadata": str(tmp_path / "metadata.json"),
        },
    )

    seen_metadata = json.loads(
        partition_metadata_path(
            "candidate_seen_flags",
            strategy="similarity",
            split="valid",
            cutoff_week=10,
        ).read_text(encoding="utf-8")
    )
    similarity_metadata = json.loads(
        partition_metadata_path(
            "similarity_scores",
            strategy="similarity",
            split="valid",
            cutoff_week=10,
        ).read_text(encoding="utf-8")
    )
    trend_metadata = json.loads(
        partition_metadata_path(
            "trend_scores",
            strategy="similarity",
            split="valid",
            cutoff_week=10,
        ).read_text(encoding="utf-8")
    )

    assert set(seen_metadata["input_artifacts"]) == {
        "weekly_transactions",
        "candidate_items",
        "candidate_metadata",
    }
    assert set(similarity_metadata["input_artifacts"]) == {
        "article_attributes",
        "user_profile",
        "candidate_items",
        "candidate_metadata",
    }
    assert set(trend_metadata["input_artifacts"]) == {
        "article_attributes",
        "trend_predictions",
        "candidate_items",
        "candidate_metadata",
    }


def test_build_feature_cache_accepts_empty_candidates_with_manifest_only(
    tmp_path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "features" / "metadata.json"
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.FEATURE_CACHE_METADATA_PATH",
        manifest_path,
    )
    candidates = pd.DataFrame(
        columns=[
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ]
    )

    manifest = build_and_write_feature_cache_for_strategy(
        strategy="default",
        candidates=candidates,
        transactions=_transactions(),
        article_attributes=_article_attributes(),
        user_profile=None,
        trend_predictions=None,
        input_paths={"candidate_items": "candidate_items.parquet"},
    )

    entry = manifest["entries"]["strategy:default"]
    assert entry["input_artifacts"] == {"candidate_items": "candidate_items.parquet"}
    assert "input_fingerprints" in entry
    assert entry["output_artifacts"] == {"feature_cache_metadata": str(manifest_path)}
    assert entry["schema_version"] == 1
    assert entry["algorithm_version"] == "recommendation-feature-cache-v1"
    assert entry["config"] == {"strategy": "default"}
    assert entry["row_counts"]["candidate_rows"] == 0
    assert entry["manifest"]["strategy"] == "default"
    assert entry["manifest"]["features"] == list(FEATURE_NAMES)
    assert entry["manifest"]["partitions"] == {}
    assert not any(manifest_path.parent.glob("**/*.parquet"))
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored == manifest


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["valid", "valid", "valid"],
            "cutoff_week": [10, 10, 10],
            "label_week": [11, 11, 11],
            "strategy": ["default", "default", "default"],
            "customer_id": ["1", "1", "2"],
            "article_id": ["0000000001", "0000000002", "0000000001"],
        }
    )


def _similarity_candidates() -> pd.DataFrame:
    candidates = _candidates().copy()
    candidates["strategy"] = "similarity"
    return candidates


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["1", "2", "3"],
            "article_id": ["0000000001", "0000000001", "0000000002"],
            "week_id": [10, 9, 8],
        }
    )


def _article_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002"],
            "attr_type": ["colour_group_name", "colour_group_name"],
            "attr_value": ["red", "blue"],
        }
    )


def _user_profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "customer_id": ["1"],
            "attr_type": ["colour_group_name"],
            "attr_value": ["red"],
            "preference_score": [1.0],
        }
    )


def _patch_recommendable_pool_cache_paths(tmp_path, monkeypatch) -> None:
    feature_root = tmp_path / "features"
    manifest_path = feature_root / "metadata.json"

    def partition_path(
        feature_name: str,
        *,
        strategy: str,
        split: str,
        cutoff_week: int,
    ):
        return (
            feature_root
            / feature_name
            / f"strategy={strategy}"
            / f"split={split}"
            / f"cutoff_week={int(cutoff_week)}"
            / "part.parquet"
        )

    def partition_metadata_path(
        feature_name: str,
        *,
        strategy: str,
        split: str,
        cutoff_week: int,
    ):
        return partition_path(
            feature_name,
            strategy=strategy,
            split=split,
            cutoff_week=cutoff_week,
        ).with_name("metadata.json")

    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.FEATURE_CACHE_METADATA_PATH",
        manifest_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.feature_cache_partition_path",
        partition_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache."
        "feature_cache_partition_metadata_path",
        partition_metadata_path,
    )
