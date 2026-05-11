from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fashion_trend.recommendation import paths as recommendation_paths
from fashion_trend.recommendation.contracts import (
    CANDIDATE_ITEM_COLUMNS,
    RECOMMENDATION_ITEMS_COLUMNS,
    RECOMMENDATIONS_COLUMNS,
    USER_PROFILE_COLUMNS,
)
from fashion_trend.recommendation.evaluation.metrics import evaluate_recommendations
from fashion_trend.recommendation.evaluation.runner import (
    build_recommendable_pool_for_windows,
)
from fashion_trend.recommendation.experiments.runner import (
    candidate_strategy_for_method,
)
from fashion_trend.recommendation.freshness import build_artifact_metadata
from fashion_trend.recommendation.inputs import (
    build_evaluation_labels,
    build_target_users,
    build_user_profile,
)
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.outputs import (
    RecommendationResultChunkWriter,
    write_recommendation_result,
)
from fashion_trend.recommendation.readers import read_recommendation_items
from fashion_trend.recommendation.registry import get_recommendation_method
from fashion_trend.recommendation.retrieval.candidates import (
    build_candidate_items,
    build_source_frames_for_frames,
)
from fashion_trend.recommendation.runner import (
    filter_cached_seen_items,
    method_input_artifacts,
    run_recommendation_method_by_window,
)
from fashion_trend.recommendation.time_windows import build_recommendation_windows


def test_registry_lists_unknown_method_choices() -> None:
    with pytest.raises(ValueError, match="global_popularity.*pop_similarity"):
        get_recommendation_method("missing")


def test_global_popularity_method_contract() -> None:
    method = get_recommendation_method("global_popularity")

    assert method.name == "global_popularity"
    assert method.default_candidate_strategy is None
    assert method.required_features == ("pop_score",)
    assert method.default_weights == {"pop_score": 1.0}


def test_recent_popularity_method_contract() -> None:
    method = get_recommendation_method("recent_popularity")

    assert method.name == "recent_popularity"
    assert method.default_candidate_strategy is None
    assert method.required_features == ("recent_score",)


def test_pop_similarity_method_contract() -> None:
    method = get_recommendation_method("pop_similarity")

    assert method.default_candidate_strategy == "default"
    assert method.required_features == ("pop_score", "sim_score", "recent_score")
    assert method.default_weights == {
        "pop_score": 0.45,
        "sim_score": 0.45,
        "recent_score": 0.10,
    }


def test_pop_similarity_trend_uses_default_candidates_and_trend_score() -> None:
    method = get_recommendation_method("pop_similarity_trend")

    assert method.default_candidate_strategy == "default"
    assert method.required_features == (
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
    )
    assert method.default_weights == {
        "pop_score": 0.2,
        "sim_score": 0.2,
        "trend_score": 0.1,
        "recent_score": 0.5,
    }


def test_method_metadata_includes_candidate_and_feature_cache_artifacts(
    tmp_path,
) -> None:
    candidate_path = tmp_path / "candidate_items.parquet"
    candidate_metadata = tmp_path / "candidate_metadata.json"
    cache_metadata = tmp_path / "feature_metadata.json"
    partition = tmp_path / "features" / "part.parquet"

    artifacts = method_input_artifacts(
        base_input_paths={
            "recommendation_inputs": "data/processed/recommend/metadata.json"
        },
        candidate_items=str(candidate_path),
        candidate_metadata=str(candidate_metadata),
        feature_cache_metadata=str(cache_metadata),
        feature_partitions=[str(partition)],
    )

    assert artifacts == {
        "recommendation_inputs": "data/processed/recommend/metadata.json",
        "candidate_items": str(candidate_path),
        "candidate_metadata": str(candidate_metadata),
        "feature_cache_metadata": str(cache_metadata),
        "feature_partition_0000": str(partition),
    }


def test_filter_cached_seen_items_empty_seen_partition_returns_metadata_path(
    tmp_path,
    monkeypatch,
) -> None:
    partition_path = tmp_path / "candidate_seen_flags" / "part.parquet"
    metadata_path = tmp_path / "candidate_seen_flags" / "metadata.json"
    partition_path.parent.mkdir(parents=True)
    pd.DataFrame(
        columns=[
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
            "seen",
        ]
    ).to_parquet(partition_path, index=False)
    monkeypatch.setattr(
        "fashion_trend.recommendation.runner.feature_cache_partition_path",
        lambda feature_name, strategy, split, cutoff_week: partition_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.runner.feature_cache_partition_metadata_path",
        lambda feature_name, strategy, split, cutoff_week: metadata_path,
    )
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "default",
                "customer_id": "u1",
                "article_id": "a1",
            }
        ]
    )

    filtered, seen_partition, seen_metadata = filter_cached_seen_items(
        candidates,
        strategy="default",
        window={"split": "valid", "cutoff_week": 10, "label_week": 11},
    )

    assert filtered.equals(candidates)
    assert seen_partition == str(partition_path)
    assert seen_metadata == str(metadata_path)


def test_method_output_paths_use_parquet_items() -> None:
    paths = recommendation_paths.method_output_paths("pop_similarity")

    assert paths.recommendation_items.name == "recommendation_items.parquet"
    assert paths.recommendation_items_csv.name == "recommendation_items.csv"


def test_result_writer_writes_items_parquet_by_default(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "pop_similarity"
    monkeypatch.setattr(
        recommendation_paths,
        "method_output_paths",
        lambda method: recommendation_paths.RecommendationOutputPaths(
            output_dir=output_dir,
            recommendations=output_dir / "recommendations.csv",
            recommendation_items=output_dir / "recommendation_items.parquet",
            recommendation_items_csv=output_dir / "recommendation_items.csv",
            params=output_dir / "params.json",
            metadata=output_dir / "metadata.json",
            metrics=output_dir / "metrics.json",
        ),
    )
    from fashion_trend.recommendation.methods.base import RecommendationResult
    from fashion_trend.recommendation.outputs import write_recommendation_result

    result = RecommendationResult(
        recommendations=pd.DataFrame(columns=list(RECOMMENDATIONS_COLUMNS)),
        recommendation_items=pd.DataFrame(columns=list(RECOMMENDATION_ITEMS_COLUMNS)),
        params={"method": "pop_similarity"},
        metadata={"method": "pop_similarity"},
    )

    write_recommendation_result(result)

    assert (output_dir / "recommendations.csv").exists()
    assert (output_dir / "recommendation_items.parquet").exists()
    assert not (output_dir / "recommendation_items.csv").exists()


def test_chunk_writer_streams_multiple_item_chunks_to_readable_parquet(
    tmp_path,
    monkeypatch,
) -> None:
    output_dir = _patch_method_output_dir(monkeypatch, tmp_path, "pop_similarity")

    with RecommendationResultChunkWriter("pop_similarity") as writer:
        writer.write_chunk(
            _recommendation_result(
                [_recommendation_item_row(article_id="0000000001", rank=1)]
            )
        )
        writer.write_chunk(_recommendation_result([]))
        writer.write_chunk(
            _recommendation_result(
                [_recommendation_item_row(article_id="0000000002", rank=2)]
            )
        )
        writer.publish()

    items = read_recommendation_items(output_dir / "recommendation_items.parquet")

    assert items["article_id"].tolist() == ["0000000001", "0000000002"]
    assert not (output_dir / "recommendation_items.parquet.tmp").exists()


def test_chunk_writer_uses_stable_schema_across_numeric_dtypes(
    tmp_path,
    monkeypatch,
) -> None:
    output_dir = _patch_method_output_dir(monkeypatch, tmp_path, "pop_similarity")
    int_like_row = _recommendation_item_row(article_id="0000000001", rank=1)
    float_like_row = _recommendation_item_row(article_id="0000000002", rank=2)
    for column in ("score", "pop_score", "sim_score", "trend_score", "recent_score"):
        int_like_row[column] = 1
        float_like_row[column] = 1.5

    with RecommendationResultChunkWriter("pop_similarity") as writer:
        writer.write_chunk(_recommendation_result([int_like_row]))
        writer.write_chunk(_recommendation_result([float_like_row]))
        writer.publish()

    items = read_recommendation_items(output_dir / "recommendation_items.parquet")

    assert items["article_id"].tolist() == ["0000000001", "0000000002"]
    assert items["score"].tolist() == [1.0, 1.5]


def test_empty_items_written_by_result_writer_can_be_read_back(
    tmp_path,
    monkeypatch,
) -> None:
    output_dir = _patch_method_output_dir(monkeypatch, tmp_path, "pop_similarity")

    write_recommendation_result(_recommendation_result([]))

    items = read_recommendation_items(output_dir / "recommendation_items.parquet")

    assert items.empty
    assert tuple(items.columns) == RECOMMENDATION_ITEMS_COLUMNS
    _assert_recommendation_items_arrow_schema(
        output_dir / "recommendation_items.parquet"
    )


def test_empty_streaming_items_can_be_read_back(tmp_path, monkeypatch) -> None:
    output_dir = _patch_method_output_dir(monkeypatch, tmp_path, "pop_similarity")

    with RecommendationResultChunkWriter("pop_similarity") as writer:
        writer.write_chunk(_recommendation_result([]))
        writer.publish()

    items = read_recommendation_items(output_dir / "recommendation_items.parquet")

    assert items.empty
    assert tuple(items.columns) == RECOMMENDATION_ITEMS_COLUMNS
    _assert_recommendation_items_arrow_schema(
        output_dir / "recommendation_items.parquet"
    )


def test_chunk_writer_rejects_bad_item_columns_and_cleans_temp_files(
    tmp_path,
    monkeypatch,
) -> None:
    output_dir = _patch_method_output_dir(monkeypatch, tmp_path, "pop_similarity")
    bad_items = pd.DataFrame(
        [_recommendation_item_row()],
        columns=list(RECOMMENDATION_ITEMS_COLUMNS[:-1]),
    )

    with pytest.raises(ValueError, match="recommendation_items"):
        with RecommendationResultChunkWriter("pop_similarity") as writer:
            writer.write_chunk(
                RecommendationResult(
                    recommendations=pd.DataFrame(columns=list(RECOMMENDATIONS_COLUMNS)),
                    recommendation_items=bad_items,
                    params={"method": "pop_similarity"},
                    metadata={"method": "pop_similarity"},
                )
            )

    assert not (output_dir / "recommendations.csv.tmp").exists()
    assert not (output_dir / "recommendation_items.parquet.tmp").exists()


def _patch_method_output_dir(monkeypatch, tmp_path, method: str):
    output_dir = tmp_path / method
    monkeypatch.setattr(
        recommendation_paths,
        "method_output_paths",
        lambda requested_method: recommendation_paths.RecommendationOutputPaths(
            output_dir=output_dir,
            recommendations=output_dir / "recommendations.csv",
            recommendation_items=output_dir / "recommendation_items.parquet",
            recommendation_items_csv=output_dir / "recommendation_items.csv",
            params=output_dir / "params.json",
            metadata=output_dir / "metadata.json",
            metrics=output_dir / "metrics.json",
        ),
    )
    return output_dir


def _assert_recommendation_items_arrow_schema(path) -> None:
    schema = pq.read_schema(path)

    assert schema.field("rank").type == pa.int64()
    assert schema.field("score").type == pa.float64()
    assert schema.field("pop_score").type == pa.float64()
    assert schema.field("sim_score").type == pa.float64()
    assert schema.field("trend_score").type == pa.float64()
    assert schema.field("recent_score").type == pa.float64()


def _recommendation_result(
    item_rows: list[dict[str, object]],
    *,
    method: str = "pop_similarity",
) -> RecommendationResult:
    return RecommendationResult(
        recommendations=pd.DataFrame(columns=list(RECOMMENDATIONS_COLUMNS)),
        recommendation_items=pd.DataFrame(
            item_rows,
            columns=list(RECOMMENDATION_ITEMS_COLUMNS),
        ),
        params={"method": method},
        metadata={"method": method},
    )


def _recommendation_item_row(
    *,
    article_id: str = "0000000001",
    rank: int = 1,
    method: str = "pop_similarity",
) -> dict[str, object]:
    return {
        "customer_id": "0000001",
        "split": "valid",
        "cutoff_week": 104,
        "label_week": 105,
        "method": method,
        "article_id": article_id,
        "rank": rank,
        "score": 1.0,
        "pop_score": 0.4,
        "sim_score": 0.3,
        "trend_score": 0.2,
        "recent_score": 0.1,
        "candidate_sources": "pop",
    }


def sample_method_context(
    *,
    method_name: str = "global_popularity",
    exclude_seen: bool = True,
    user_profile: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
) -> RecommendationContext:
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
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
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4"],
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "week_id": [9, 9, 10, 10],
        }
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "attr_id": [101, 102, 101, 103],
            "attr_type": ["product_type_name"] * 4,
            "attr_value": ["Dress", "Shirt", "Dress", "Shoes"],
        }
    )
    if user_profile is None:
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
                    "purchase_count": 1,
                    "last_purchase_week": 9,
                }
            ],
            columns=list(USER_PROFILE_COLUMNS),
        )
    if candidates is None:
        candidates = pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "strategy": "default",
                    "customer_id": "u1",
                    "article_id": "0000000003",
                    "candidate_sources": "popularity|similarity",
                    "primary_source": "similarity",
                    "best_source_rank": 1,
                },
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "strategy": "default",
                    "customer_id": "u1",
                    "article_id": "0000000001",
                    "candidate_sources": "popularity",
                    "primary_source": "popularity",
                    "best_source_rank": 2,
                },
            ],
            columns=list(CANDIDATE_ITEM_COLUMNS),
        )
    return RecommendationContext(
        method=method_name,
        top_k=12,
        exclude_seen=exclude_seen,
        transactions=transactions,
        article_attributes=article_attributes,
        windows=windows,
        target_users=target_users,
        candidates=candidates,
        user_profile=user_profile,
        trend_predictions=None,
    )


def assert_method_result_shape(result, method_name: str) -> None:
    assert tuple(result.recommendations.columns) == RECOMMENDATIONS_COLUMNS
    assert tuple(result.recommendation_items.columns) == RECOMMENDATION_ITEMS_COLUMNS
    assert set(result.recommendations["method"]) == {method_name}
    assert set(result.recommendation_items["method"]) == {method_name}
    assert result.recommendation_items["rank"].tolist() == list(
        range(1, len(result.recommendation_items) + 1)
    )
    assert result.recommendation_items["rank"].between(1, 12).all()
    assert result.params["method"] == method_name


@pytest.mark.parametrize("method_name", ["global_popularity", "recent_popularity"])
def test_popularity_baselines_build_without_profile_or_candidates(
    method_name: str,
) -> None:
    method = get_recommendation_method(method_name)
    context = sample_method_context(
        method_name=method_name,
        user_profile=None,
        candidates=None,
    )

    result = method.build_recommendations(context)

    assert_method_result_shape(result, method_name)
    assert "0000000001" not in set(result.recommendation_items["article_id"])
    assert result.params["exclude_seen"] is True


@pytest.mark.parametrize("method_name", ["global_popularity", "recent_popularity"])
def test_popularity_baselines_backfill_after_excluding_seen(method_name: str) -> None:
    context = sample_method_context(method_name=method_name)
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4"],
            "article_id": ["0000000001", "0000000001", "0000000002", "0000000003"],
            "week_id": [10, 10, 10, 10],
        }
    )

    result = get_recommendation_method(method_name).build_recommendations(
        replace(context, top_k=2, transactions=transactions)
    )

    assert set(result.recommendation_items["article_id"]) == {
        "0000000002",
        "0000000003",
    }
    assert len(result.recommendation_items) == 2


def test_attribute_similarity_falls_back_when_profile_is_empty() -> None:
    method = get_recommendation_method("attribute_similarity")
    empty_profile = pd.DataFrame(columns=list(USER_PROFILE_COLUMNS))
    context = sample_method_context(
        method_name="attribute_similarity",
        user_profile=empty_profile,
    )

    result = method.build_recommendations(context)

    assert_method_result_shape(result, "attribute_similarity")
    assert result.metadata["fallback_user_count"] == 1
    assert result.params["weights"] == {"recent_score": 1.0}


def test_attribute_similarity_backfills_short_similarity_candidates() -> None:
    method = get_recommendation_method("attribute_similarity")
    context = sample_method_context(method_name="attribute_similarity")
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "similarity",
                "customer_id": "u1",
                "article_id": "0000000001",
                "candidate_sources": "similarity",
                "primary_source": "similarity",
                "best_source_rank": 1,
            }
        ],
        columns=list(CANDIDATE_ITEM_COLUMNS),
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4"],
            "article_id": ["0000000001", "0000000001", "0000000002", "0000000003"],
            "week_id": [10, 10, 10, 10],
        }
    )

    result = method.build_recommendations(
        replace(context, top_k=2, transactions=transactions, candidates=candidates)
    )

    assert set(result.recommendation_items["article_id"]) == {
        "0000000002",
        "0000000003",
    }
    assert len(result.recommendation_items) == 2


def test_pop_similarity_builds_without_trend_predictions() -> None:
    method = get_recommendation_method("pop_similarity")
    context = sample_method_context(method_name="pop_similarity")

    result = method.build_recommendations(context)

    assert_method_result_shape(result, "pop_similarity")
    assert "trend_score" in result.recommendation_items.columns
    assert result.recommendation_items["trend_score"].eq(0.0).all()


def test_pop_similarity_trend_method_builds_recommendations_with_trend_predictions() -> (
    None
):
    method = get_recommendation_method("pop_similarity_trend")
    context = sample_method_context(method_name="pop_similarity_trend")
    predictions = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "week_id": [10, 10],
            "attr_id": [101, 102],
            "attr_type": ["product_type_name", "product_type_name"],
            "attr_value": ["Dress", "Shirt"],
            "pred_target_growth": [2.0, 1.0],
        }
    )

    result = method.build_recommendations(
        replace(context, trend_predictions=predictions)
    )

    assert_method_result_shape(result, "pop_similarity_trend")
    assert result.recommendation_items["trend_score"].max() > 0.0


def test_pop_similarity_trend_method_requires_trend_predictions() -> None:
    method = get_recommendation_method("pop_similarity_trend")
    context = sample_method_context(method_name="pop_similarity_trend")

    with pytest.raises(FileNotFoundError, match="trend predictions"):
        method.build_recommendations(context)


def test_window_runner_writes_streamed_method_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(recommendation_paths, "OUTPUT_RECOMMENDATION_DIR", tmp_path)
    context = sample_method_context(method_name="global_popularity")

    result = run_recommendation_method_by_window(
        method_name="global_popularity",
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        windows=context.windows,
        target_users=context.target_users,
        exclude_seen=context.exclude_seen,
        input_paths={"weekly_transactions": "in_memory"},
    )
    output_paths = recommendation_paths.method_output_paths("global_popularity")
    params = json.loads(output_paths.params.read_text(encoding="utf-8"))
    metadata = json.loads(output_paths.metadata.read_text(encoding="utf-8"))

    assert output_paths.recommendations.exists()
    assert output_paths.recommendation_items.exists()
    assert output_paths.params.exists()
    assert output_paths.metadata.exists()
    assert tuple(result.recommendations.columns) == RECOMMENDATIONS_COLUMNS
    assert tuple(result.recommendation_items.columns) == RECOMMENDATION_ITEMS_COLUMNS
    assert result.metadata["window_count"] == 1
    assert result.metadata["recommendation_item_rows"] == len(
        result.recommendation_items
    )
    assert params["candidate_strategy"] is None
    assert params["score_features"] == ["pop_score"]
    assert metadata["generated_at"].endswith("Z")
    assert metadata["input_artifacts"] == {"weekly_transactions": "in_memory"}
    assert metadata["input_fingerprints"]["weekly_transactions"]["path"] == "in_memory"
    assert metadata["schema_version"] == 1
    assert metadata["algorithm_version"] == "recommendation-method-v1"
    assert metadata["config"] == {
        "method": "global_popularity",
        "top_k": 12,
        "candidate_strategy": None,
        "exclude_seen": True,
        "weights": {"pop_score": 1.0},
    }
    assert metadata["output_artifacts"] == {
        "recommendations": str(output_paths.recommendations),
        "recommendation_items": str(output_paths.recommendation_items),
        "params": str(output_paths.params),
        "metadata": str(output_paths.metadata),
    }
    assert metadata["window_config"] == {
        "window_count": 1,
        "splits": ["valid"],
        "min_cutoff_week": 10,
        "max_cutoff_week": 10,
        "min_label_week": 11,
        "max_label_week": 11,
    }


def test_window_runner_records_trend_score_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(recommendation_paths, "OUTPUT_RECOMMENDATION_DIR", tmp_path)
    context = sample_method_context(method_name="pop_similarity_trend")
    _write_pop_similarity_cache_partitions(
        tmp_path,
        monkeypatch,
        context.candidates,
        include_trend=True,
    )
    predictions = pd.DataFrame(
        {
            "split": ["valid"],
            "week_id": [10],
            "attr_id": [101],
            "attr_type": ["product_type_name"],
            "attr_value": ["Dress"],
            "pred_target_growth": [2.0],
        }
    )

    run_recommendation_method_by_window(
        method_name="pop_similarity_trend",
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        windows=context.windows,
        target_users=context.target_users,
        candidates=context.candidates,
        user_profile=context.user_profile,
        trend_predictions=predictions,
        input_paths={"trend_predictions": "outputs/models/lightgbm/predictions.csv"},
        trend_model_source="outputs/models/lightgbm/predictions.csv",
    )
    output_paths = recommendation_paths.method_output_paths("pop_similarity_trend")
    metadata = json.loads(output_paths.metadata.read_text(encoding="utf-8"))

    assert metadata["trend_score_config"]["stable_trend_model_source"] == (
        "outputs/models/lightgbm/predictions.csv"
    )
    assert "product_type_name" in metadata["trend_score_config"]["core_attr_types"]
    assert metadata["trend_score_config"]["attr_weights"]["product_type_name"] == 0.35


def test_cached_trend_method_requires_trend_predictions_in_input_chain(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(recommendation_paths, "OUTPUT_RECOMMENDATION_DIR", tmp_path)
    context = sample_method_context(method_name="pop_similarity_trend")
    _write_pop_similarity_cache_partitions(
        tmp_path,
        monkeypatch,
        context.candidates,
        include_trend=True,
    )

    with pytest.raises(FileNotFoundError, match="trend predictions"):
        run_recommendation_method_by_window(
            method_name="pop_similarity_trend",
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=context.windows,
            target_users=context.target_users,
            candidates=context.candidates,
            user_profile=context.user_profile,
            trend_predictions=None,
        )


def test_cached_trend_method_rejects_stale_trend_score_cache(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(recommendation_paths, "OUTPUT_RECOMMENDATION_DIR", tmp_path)
    context = sample_method_context(method_name="pop_similarity_trend")
    _write_pop_similarity_cache_partitions(
        tmp_path,
        monkeypatch,
        context.candidates,
        include_trend=True,
    )
    predictions = pd.DataFrame(
        {
            "split": ["valid"],
            "week_id": [10],
            "attr_id": [101],
            "attr_type": ["product_type_name"],
            "attr_value": ["Dress"],
            "pred_target_growth": [2.0],
        }
    )

    with pytest.raises(RuntimeError, match="trend_predictions"):
        run_recommendation_method_by_window(
            method_name="pop_similarity_trend",
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=context.windows,
            target_users=context.target_users,
            candidates=context.candidates,
            user_profile=context.user_profile,
            trend_predictions=predictions,
            input_paths={"trend_predictions": "outputs/models/lightgbm/new.csv"},
        )


def test_cached_method_rejects_feature_metadata_missing_required_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(recommendation_paths, "OUTPUT_RECOMMENDATION_DIR", tmp_path)
    context = sample_method_context(method_name="pop_similarity")
    _write_pop_similarity_cache_partitions(
        tmp_path,
        monkeypatch,
        context.candidates,
    )
    metadata_path = (
        tmp_path
        / "features"
        / "popularity_scores"
        / "strategy=default"
        / "split=valid"
        / "cutoff_week=10"
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_artifacts"] = {}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="popularity_scores.*candidate_items"):
        run_recommendation_method_by_window(
            method_name="pop_similarity",
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=context.windows,
            target_users=context.target_users,
            candidates=context.candidates,
            user_profile=context.user_profile,
            input_paths={
                "candidate_items": (
                    "data/processed/recommend/candidates/default.parquet"
                ),
            },
        )


def test_cached_method_rejects_invalid_feature_metadata_json(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(recommendation_paths, "OUTPUT_RECOMMENDATION_DIR", tmp_path)
    context = sample_method_context(method_name="pop_similarity")
    _write_pop_similarity_cache_partitions(
        tmp_path,
        monkeypatch,
        context.candidates,
    )
    metadata_path = (
        tmp_path
        / "features"
        / "popularity_scores"
        / "strategy=default"
        / "split=valid"
        / "cutoff_week=10"
        / "metadata.json"
    )
    metadata_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata is invalid"):
        run_recommendation_method_by_window(
            method_name="pop_similarity",
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=context.windows,
            target_users=context.target_users,
            candidates=context.candidates,
            user_profile=context.user_profile,
            input_paths={
                "candidate_items": (
                    "data/processed/recommend/candidates/default.parquet"
                ),
            },
        )


def test_window_runner_uses_feature_cache_and_records_used_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(recommendation_paths, "OUTPUT_RECOMMENDATION_DIR", tmp_path)
    context = sample_method_context(method_name="pop_similarity")
    cache_paths = _write_pop_similarity_cache_partitions(
        tmp_path,
        monkeypatch,
        context.candidates,
    )
    feature_metadata = tmp_path / "features" / "metadata.json"
    feature_metadata.parent.mkdir(parents=True, exist_ok=True)
    feature_metadata.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "fashion_trend.recommendation.methods.baselines.global_popularity"
        ".build_ranking_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached runner must not rebuild ranking features")
        ),
    )

    result = run_recommendation_method_by_window(
        method_name="pop_similarity",
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        windows=context.windows,
        target_users=context.target_users,
        candidates=context.candidates,
        user_profile=context.user_profile,
        input_paths={
            "candidate_items": "data/processed/recommend/candidates/default.parquet",
            "candidate_metadata": ("data/processed/recommend/candidates/metadata.json"),
            "feature_cache_metadata": str(feature_metadata),
        },
    )

    output_paths = recommendation_paths.method_output_paths("pop_similarity")
    metadata = json.loads(output_paths.metadata.read_text(encoding="utf-8"))

    assert tuple(result.recommendation_items.columns) == RECOMMENDATION_ITEMS_COLUMNS
    assert cache_paths["seen"] in metadata["used_feature_artifacts"]
    assert cache_paths["seen_metadata"] in metadata["used_feature_artifacts"]
    assert cache_paths["pop"] in metadata["used_feature_artifacts"]
    assert cache_paths["recent"] in metadata["used_feature_artifacts"]
    assert cache_paths["sim"] in metadata["used_feature_artifacts"]
    assert metadata["input_artifacts"]["feature_cache_metadata"] == str(
        feature_metadata
    )
    assert cache_paths["seen"] in metadata["input_artifacts"].values()
    assert metadata["window_summaries"][0]["used_feature_artifacts"] == (
        metadata["used_feature_artifacts"]
    )
    assert metadata["underfilled_user_count"] >= metadata["backfilled_user_count"]
    assert metadata["window_summaries"][0]["still_underfilled_user_count"] >= 0


def _write_pop_similarity_cache_partitions(
    tmp_path,
    monkeypatch,
    candidates: pd.DataFrame,
    *,
    include_trend: bool = False,
) -> dict[str, str]:
    assert candidates is not None
    base_dir = tmp_path / "features"

    def partition_path(feature_name, strategy, split, cutoff_week):
        return (
            base_dir
            / feature_name
            / f"strategy={strategy}"
            / f"split={split}"
            / f"cutoff_week={cutoff_week}"
            / "part.parquet"
        )

    def metadata_path(feature_name, strategy, split, cutoff_week):
        return partition_path(
            feature_name,
            strategy,
            split,
            cutoff_week,
        ).with_name("metadata.json")

    monkeypatch.setattr(
        "fashion_trend.recommendation.runner.feature_cache_partition_path",
        partition_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.runner.feature_cache_partition_metadata_path",
        metadata_path,
    )

    window = candidates.iloc[0]
    split = str(window["split"])
    cutoff_week = int(window["cutoff_week"])
    label_week = int(window["label_week"])
    strategy = str(window["strategy"])
    seen = candidates.iloc[[1]].loc[
        :,
        [
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
    ]
    _write_cache_partition(
        partition_path("candidate_seen_flags", strategy, split, cutoff_week),
        seen.assign(seen=True),
    )

    article_scores = candidates.loc[
        :,
        ["split", "cutoff_week", "label_week", "strategy", "article_id"],
    ].drop_duplicates()
    _write_cache_partition(
        partition_path("popularity_scores", strategy, split, cutoff_week),
        article_scores.assign(pop_score=[0.8, 0.2]),
    )
    _write_cache_partition(
        partition_path("recent_scores", strategy, split, cutoff_week),
        article_scores.assign(recent_score=[0.7, 0.1]),
    )
    _write_cache_partition(
        partition_path("similarity_scores", strategy, split, cutoff_week),
        candidates.loc[
            :,
            [
                "split",
                "cutoff_week",
                "label_week",
                "strategy",
                "customer_id",
                "article_id",
            ],
        ].assign(sim_score=[0.9, 0.4]),
    )
    feature_names = [
        "candidate_seen_flags",
        "popularity_scores",
        "recent_scores",
        "similarity_scores",
    ]
    if include_trend:
        _write_cache_partition(
            partition_path("trend_scores", strategy, split, cutoff_week),
            article_scores.assign(trend_score=[0.6, 0.3]),
        )
        feature_names.append("trend_scores")
    for feature_name in feature_names:
        partition_file = partition_path(feature_name, strategy, split, cutoff_week)
        metadata_file = metadata_path(feature_name, strategy, split, cutoff_week)
        metadata_file.write_text(
            json.dumps(
                build_artifact_metadata(
                    name=f"recommendation_feature_cache_{feature_name}",
                    input_artifacts=(
                        {
                            "candidate_items": (
                                "data/processed/recommend/candidates/default.parquet"
                            ),
                            "trend_predictions": (
                                "outputs/models/lightgbm/predictions.csv"
                            ),
                        }
                        if feature_name == "trend_scores"
                        else {
                            "candidate_items": (
                                "data/processed/recommend/candidates/default.parquet"
                            )
                        }
                    ),
                    output_artifacts={
                        "partition": str(partition_file),
                        "partition_metadata": str(metadata_file),
                    },
                    schema_version=1,
                    algorithm_version="recommendation-feature-cache-v1",
                    config={
                        "feature_name": feature_name,
                        "strategy": strategy,
                        "split": split,
                        "cutoff_week": cutoff_week,
                        "label_week": label_week,
                    },
                    row_counts={"rows": 1},
                )
            ),
            encoding="utf-8",
        )

    return {
        "seen": str(
            partition_path("candidate_seen_flags", strategy, split, cutoff_week)
        ),
        "seen_metadata": str(
            metadata_path("candidate_seen_flags", strategy, split, cutoff_week)
        ),
        "pop": str(partition_path("popularity_scores", strategy, split, cutoff_week)),
        "recent": str(partition_path("recent_scores", strategy, split, cutoff_week)),
        "sim": str(partition_path("similarity_scores", strategy, split, cutoff_week)),
    }


def _write_cache_partition(path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)


def make_small_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [
                "00000000000000000000000000000001",
                "00000000000000000000000000000001",
                "00000000000000000000000000000001",
                "00000000000000000000000000000001",
                "00000000000000000000000000000002",
                "00000000000000000000000000000002",
                "00000000000000000000000000000002",
                "00000000000000000000000000000003",
                "00000000000000000000000000000003",
            ],
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
                "0000000002",
                "0000000004",
                "0000000005",
                "0000000006",
                "0000000001",
            ],
            "week_id": [8, 10, 11, 13, 9, 11, 13, 10, 12],
        }
    )


def make_small_article_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
                "0000000005",
                "0000000006",
            ],
            "attr_id": [101, 102, 101, 103, 104, 102],
            "attr_type": ["product_type_name"] * 6,
            "attr_value": ["Dress", "Shirt", "Dress", "Skirt", "Pants", "Shirt"],
        }
    )


def make_small_trend_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["valid", "valid", "valid", "test", "test", "test"],
            "week_id": [10, 10, 10, 12, 12, 12],
            "attr_id": [101, 102, 103, 101, 102, 104],
            "attr_type": ["product_type_name"] * 6,
            "attr_value": ["Dress", "Shirt", "Skirt", "Dress", "Shirt", "Pants"],
            "pred_target_growth": [0.8, 0.3, 0.1, 0.2, 0.7, 0.9],
        }
    )


def build_candidates_for_registered_method(
    method_name: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    predictions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame | None:
    strategy = candidate_strategy_for_method(method_name)
    if strategy is None:
        return None
    return build_candidate_items(
        strategy=strategy,
        source_frames=build_source_frames_for_frames(
            strategy=strategy,
            transactions=transactions,
            article_attributes=article_attributes,
            trend_predictions=predictions,
            windows=windows,
            target_users=target_users,
            user_profile=profile,
        ),
    )


def test_recommendation_pipeline_small_fixture_runs_without_leakage() -> None:
    transactions = make_small_transactions()
    article_attributes = make_small_article_attributes()
    predictions = make_small_trend_predictions()

    windows = build_recommendation_windows(predictions)
    target_users = build_target_users(transactions, windows)
    labels = build_evaluation_labels(transactions, windows, target_users)
    profile = build_user_profile(
        transactions, article_attributes, windows, target_users
    )

    source_frames = build_source_frames_for_frames(
        strategy="default",
        transactions=transactions,
        article_attributes=article_attributes,
        trend_predictions=predictions,
        windows=windows,
        target_users=target_users,
        user_profile=profile,
    )
    candidates = build_candidate_items(strategy="default", source_frames=source_frames)
    result = get_recommendation_method("pop_similarity_trend").build_recommendations(
        RecommendationContext(
            method="pop_similarity_trend",
            top_k=12,
            exclude_seen=True,
            transactions=transactions,
            article_attributes=article_attributes,
            windows=windows,
            target_users=target_users,
            candidates=candidates,
            user_profile=profile,
            trend_predictions=predictions,
            weights={
                "pop_score": 0.35,
                "sim_score": 0.35,
                "trend_score": 0.25,
                "recent_score": 0.05,
            },
        )
    )
    metrics = evaluate_recommendations(
        result.recommendations,
        target_users,
        labels,
        build_recommendable_pool_for_windows(transactions, windows),
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["user_count"] > 0
    assert metrics["test"]["user_count"] > 0
    assert result.recommendations["prediction"].map(type).eq(str).all()
    assert result.recommendation_items["article_id"].map(type).eq(str).all()
    assert result.recommendation_items["customer_id"].map(type).eq(str).all()
    assert result.recommendation_items["rank"].max() <= 12
    assert not result.recommendation_items.duplicated(
        ["customer_id", "split", "cutoff_week", "label_week", "article_id"]
    ).any()
    merged_seen = result.recommendation_items.merge(
        transactions,
        on=["customer_id", "article_id"],
        how="inner",
    )
    assert (
        merged_seen["week_id"].astype(int) > merged_seen["cutoff_week"].astype(int)
    ).all()


@pytest.mark.parametrize(
    "method_name",
    [
        "global_popularity",
        "recent_popularity",
        "attribute_similarity",
        "pop_similarity",
        "pop_similarity_trend",
    ],
)
def test_each_registered_method_builds_recommendations_on_small_fixture(
    method_name: str,
) -> None:
    transactions = make_small_transactions()
    article_attributes = make_small_article_attributes()
    predictions = make_small_trend_predictions()
    windows = build_recommendation_windows(predictions)
    target_users = build_target_users(transactions, windows)
    profile = build_user_profile(
        transactions, article_attributes, windows, target_users
    )
    candidates = build_candidates_for_registered_method(
        method_name,
        transactions,
        article_attributes,
        predictions,
        windows,
        target_users,
        profile,
    )

    result = get_recommendation_method(method_name).build_recommendations(
        RecommendationContext(
            method=method_name,
            top_k=12,
            exclude_seen=True,
            transactions=transactions,
            article_attributes=article_attributes,
            windows=windows,
            target_users=target_users,
            candidates=candidates,
            user_profile=profile,
            trend_predictions=(
                predictions if method_name == "pop_similarity_trend" else None
            ),
            weights=None,
        )
    )

    assert set(result.recommendations["method"]) == {method_name}
    assert set(result.recommendation_items["method"]) == {method_name}
    assert result.recommendations["prediction"].map(type).eq(str).all()
    assert result.recommendation_items["article_id"].map(type).eq(str).all()
    assert result.recommendation_items["customer_id"].map(type).eq(str).all()
    assert result.recommendation_items["rank"].between(1, 12).all()
    assert not result.recommendation_items.duplicated(
        ["customer_id", "split", "cutoff_week", "label_week", "article_id"]
    ).any()
