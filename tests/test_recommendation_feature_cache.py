from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.recommendation.experiments import enhanced_runner
from fashion_trend.recommendation.features import cache as feature_cache
from fashion_trend.recommendation.features.cache import (
    FEATURE_CACHE_ALGORITHM_VERSION,
    FEATURE_CACHE_SCHEMA_VERSION,
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


def test_enhanced_feature_cache_writes_strategy_scoped_partitions(
    tmp_path,
    monkeypatch,
) -> None:
    partition_path, _metadata_path = _patch_feature_cache_paths(tmp_path, monkeypatch)

    manifest = build_and_write_feature_cache_for_strategy(
        strategy="enhanced_default",
        candidates=_enhanced_candidates(),
        transactions=_enhanced_transactions(),
        article_attributes=_enhanced_article_attributes(),
        user_profile=_enhanced_user_profile(),
        trend_predictions=_enhanced_trend_predictions(),
        customer_profile=_enhanced_customer_profile(),
        article_product_map=_enhanced_article_product_map(),
        input_paths=_enhanced_input_paths(tmp_path),
    )

    entry = manifest["entries"]["strategy:enhanced_default"]
    enhanced_features = {
        "reorder_scores",
        "variant_scores",
        "age_popularity_scores",
        "preference_popularity_scores",
        "source_rank_scores",
        "source_count_scores",
    }
    assert enhanced_features.issubset(set(entry["manifest"]["partitions"]))
    for feature_name in enhanced_features:
        enhanced_path = partition_path(
            feature_name,
            strategy="enhanced_default",
            split="valid",
            cutoff_week=10,
        )
        default_path = partition_path(
            feature_name,
            strategy="default",
            split="valid",
            cutoff_week=10,
        )
        assert enhanced_path.exists()
        assert not default_path.exists()


def test_enhanced_feature_cache_rejects_default_strategy_reuse() -> None:
    with pytest.raises(ValueError, match="strategy"):
        build_and_write_feature_cache_for_strategy(
            strategy="default",
            candidates=_enhanced_candidates(),
            transactions=_enhanced_transactions(),
            article_attributes=_enhanced_article_attributes(),
            user_profile=_enhanced_user_profile(),
            trend_predictions=None,
            customer_profile=_enhanced_customer_profile(),
            article_product_map=_enhanced_article_product_map(),
        )


def test_enhanced_feature_metadata_records_algorithm_and_strategy(
    tmp_path,
    monkeypatch,
) -> None:
    _partition_path, metadata_path_for = _patch_feature_cache_paths(
        tmp_path,
        monkeypatch,
    )
    input_paths = _enhanced_input_paths(tmp_path)

    build_and_write_feature_cache_for_strategy(
        strategy="enhanced_default",
        candidates=_enhanced_candidates(),
        transactions=_enhanced_transactions(),
        article_attributes=_enhanced_article_attributes(),
        user_profile=_enhanced_user_profile(),
        trend_predictions=_enhanced_trend_predictions(),
        customer_profile=_enhanced_customer_profile(),
        article_product_map=_enhanced_article_product_map(),
        input_paths=input_paths,
    )

    metadata_path = metadata_path_for(
        "variant_scores",
        strategy="enhanced_default",
        split="valid",
        cutoff_week=10,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == FEATURE_CACHE_SCHEMA_VERSION
    assert metadata["algorithm_version"] == FEATURE_CACHE_ALGORITHM_VERSION
    assert metadata["config"]["feature_name"] == "variant_scores"
    assert metadata["config"]["strategy"] == "enhanced_default"
    assert set(metadata["input_artifacts"]) == {
        "weekly_transactions",
        "article_product_map",
        "candidate_items",
        "candidate_metadata",
    }
    assert (
        metadata["input_artifacts"]["article_product_map"]
        == input_paths["article_product_map"]
    )

    source_rank_metadata = json.loads(
        metadata_path_for(
            "source_rank_scores",
            strategy="enhanced_default",
            split="valid",
            cutoff_week=10,
        ).read_text(encoding="utf-8")
    )
    assert set(source_rank_metadata["input_artifacts"]) == {
        "weekly_transactions",
        "article_attributes",
        "user_profile",
        "trend_predictions",
        "customer_profile",
        "article_product_map",
        "candidate_items",
        "candidate_metadata",
    }


def test_enhanced_feature_cache_rejects_changed_candidate_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    _partition_path, _metadata_path = _patch_feature_cache_paths(
        tmp_path,
        monkeypatch,
    )
    input_paths = _enhanced_input_paths(tmp_path)
    for key, path in input_paths.items():
        Path(path).write_text(key, encoding="utf-8")
    build_and_write_feature_cache_for_strategy(
        strategy="enhanced_default",
        candidates=_enhanced_candidates(),
        transactions=_enhanced_transactions(),
        article_attributes=_enhanced_article_attributes(),
        user_profile=_enhanced_user_profile(),
        trend_predictions=_enhanced_trend_predictions(),
        customer_profile=_enhanced_customer_profile(),
        article_product_map=_enhanced_article_product_map(),
        input_paths=input_paths,
    )

    Path(input_paths["candidate_metadata"]).write_text("changed", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="candidate_metadata.*--force-cache",
    ):
        feature_cache.assert_feature_cache_partitions_fresh(
            strategy="enhanced_default",
            candidates=_enhanced_candidates(),
            input_paths=input_paths,
        )


def test_enhanced_feature_cache_reuse_requires_global_manifest_entry(
    tmp_path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "features" / "metadata.json"
    monkeypatch.setattr(
        enhanced_runner,
        "FEATURE_CACHE_METADATA_PATH",
        manifest_path,
    )
    empty_candidates = pd.DataFrame(
        columns=["split", "cutoff_week", "label_week"],
    )

    assert (
        enhanced_runner.enhanced_feature_cache_partitions_exist(empty_candidates)
        is False
    )

    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"entries": {"strategy:default": {}}}),
        encoding="utf-8",
    )
    assert (
        enhanced_runner.enhanced_feature_cache_partitions_exist(empty_candidates)
        is False
    )

    manifest_path.write_text(
        json.dumps({"entries": {"strategy:enhanced_default": {}}}),
        encoding="utf-8",
    )
    assert (
        enhanced_runner.enhanced_feature_cache_partitions_exist(empty_candidates)
        is True
    )


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


def _enhanced_candidates() -> pd.DataFrame:
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


def _enhanced_candidate(
    article_id: str,
    candidate_sources: str,
    best_source_rank: int,
    *,
    has_reorder_source: bool = False,
) -> dict[str, object]:
    return {
        "split": "valid",
        "cutoff_week": 10,
        "label_week": 11,
        "strategy": "enhanced_default",
        "customer_id": "1",
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
            {"customer_id": "1", "article_id": "0000000001", "week_id": 10},
            {"customer_id": "1", "article_id": "0000000001", "week_id": 8},
            {"customer_id": "1", "article_id": "0000000002", "week_id": 9},
            {"customer_id": "2", "article_id": "0000000001", "week_id": 10},
            {"customer_id": "2", "article_id": "0000000002", "week_id": 10},
            {"customer_id": "2", "article_id": "0000000003", "week_id": 9},
            {"customer_id": "2", "article_id": "0000000003", "week_id": 8},
            {"customer_id": "2", "article_id": "0000000004", "week_id": 7},
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
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "red",
                "preference_score": 1.0,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "blue",
                "preference_score": 0.5,
            },
        ]
    )


def _enhanced_trend_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["valid"],
            "week_id": [10],
            "attr_type": ["colour_group_name"],
            "attr_value": ["red"],
            "pred_target_growth": [1.0],
        }
    )


def _enhanced_customer_profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["1", "2"],
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


def _enhanced_input_paths(tmp_path) -> dict[str, str]:
    return {
        "weekly_transactions": str(tmp_path / "weekly.parquet"),
        "article_attributes": str(tmp_path / "article_attributes.csv"),
        "user_profile": str(tmp_path / "user_profile.parquet"),
        "trend_predictions": str(tmp_path / "predictions.csv"),
        "customer_profile": str(tmp_path / "customer_profile.parquet"),
        "article_product_map": str(tmp_path / "article_product_map.parquet"),
        "candidate_items": str(tmp_path / "candidate_items.parquet"),
        "candidate_metadata": str(tmp_path / "candidate_metadata.json"),
    }


def _patch_feature_cache_paths(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        "fashion_trend.recommendation.paths.feature_cache_partition_path",
        partition_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.paths.feature_cache_partition_metadata_path",
        partition_metadata_path,
    )
    return partition_path, partition_metadata_path


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
