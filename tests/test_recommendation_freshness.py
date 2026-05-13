from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.experiments import (
    enhanced_runner,
)
from fashion_trend.recommendation.experiments import runner as experiment_runner
from fashion_trend.recommendation.fingerprints import build_input_fingerprints
from fashion_trend.recommendation.freshness import (
    assert_fresh_metadata,
    build_artifact_metadata,
)
from fashion_trend.recommendation.retrieval.candidates import _enhanced_config


def test_build_artifact_metadata_records_versions_config_and_rows(tmp_path) -> None:
    input_path = tmp_path / "candidate_items.parquet"
    input_path.write_text("candidate rows", encoding="utf-8")
    input_artifacts = {"default_candidates": str(input_path)}
    output_artifacts = {"recommendations": str(tmp_path / "recommendations.csv")}
    config = {"method": "pop_similarity", "top_k": 12}
    row_counts = {"recommendation_rows": 2}

    metadata = build_artifact_metadata(
        name="recommendation_method",
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        schema_version=1,
        algorithm_version="recommendation-method-v1",
        config=config,
        row_counts=row_counts,
    )

    assert metadata["name"] == "recommendation_method"
    assert metadata["schema_version"] == 1
    assert metadata["algorithm_version"] == "recommendation-method-v1"
    assert metadata["config"] == config
    assert metadata["row_counts"] == row_counts
    assert metadata["input_artifacts"] == input_artifacts
    assert metadata["input_fingerprints"] == build_input_fingerprints(input_artifacts)
    assert metadata["output_artifacts"] == output_artifacts


def test_assert_fresh_metadata_rejects_changed_candidate_fingerprint(tmp_path) -> None:
    input_path = tmp_path / "candidate_items.parquet"
    input_path.write_text("old candidate rows", encoding="utf-8")
    input_artifacts = {"default_candidates": str(input_path)}
    output_artifacts = {"recommendations": str(tmp_path / "recommendations.csv")}
    config = {"method": "pop_similarity", "top_k": 12}
    metadata = build_artifact_metadata(
        name="recommendation_method",
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        schema_version=1,
        algorithm_version="recommendation-method-v1",
        config=config,
        row_counts={"recommendation_rows": 1},
    )
    metadata_path = tmp_path / "metadata.json"
    write_json_atomic(metadata, metadata_path)
    input_path.write_text("new candidate rows", encoding="utf-8")

    with pytest.raises(RuntimeError, match="input_fingerprints changed"):
        assert_fresh_metadata(
            metadata_path=metadata_path,
            expected_input_artifacts=input_artifacts,
            expected_output_artifacts=output_artifacts,
            expected_schema_version=1,
            expected_algorithm_version="recommendation-method-v1",
            expected_config=config,
            stale_message=lambda reason: f"stale: {reason}",
        )

    stored = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert stored["input_fingerprints"] != build_input_fingerprints(input_artifacts)


def test_assert_fresh_metadata_rejects_non_object_metadata(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata is invalid"):
        assert_fresh_metadata(
            metadata_path=metadata_path,
            expected_input_artifacts={},
            expected_output_artifacts={},
            expected_schema_version=1,
            expected_algorithm_version="v1",
            expected_config={},
            stale_message=lambda reason: f"stale: {reason}",
        )


def test_enhanced_candidate_freshness_requires_customer_profile_and_product_map(
    tmp_path,
    monkeypatch,
) -> None:
    candidate_path = (
        tmp_path / "candidates" / "enhanced_default" / "candidate_items.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("candidate rows", encoding="utf-8")
    required_inputs = {
        "weekly_transactions": str(tmp_path / "weekly.parquet"),
        "article_attributes": str(tmp_path / "article_attributes.csv"),
        "trend_predictions": str(tmp_path / "predictions.csv"),
        "time_windows": str(tmp_path / "time_windows.parquet"),
        "target_users": str(tmp_path / "target_users.parquet"),
        "user_profile": str(tmp_path / "user_profile.parquet"),
        "customer_profile": str(tmp_path / "customer_profile.parquet"),
        "article_product_map": str(tmp_path / "article_product_map.parquet"),
    }
    for path in required_inputs.values():
        Path(path).write_text("input", encoding="utf-8")
    stale_inputs = {
        key: value
        for key, value in required_inputs.items()
        if key not in {"customer_profile", "article_product_map"}
    }
    metadata = build_artifact_metadata(
        name="recommendation_candidates",
        input_artifacts=stale_inputs,
        output_artifacts={"candidate_items": str(candidate_path)},
        schema_version=1,
        algorithm_version="recommendation-candidates-v1",
        config={
            "strategy": "enhanced_default",
            "candidates_per_source": 12,
            **_enhanced_config("enhanced_default"),
        },
        row_counts={"candidate_rows": 1},
    )
    write_json_atomic(metadata, candidate_path.with_name("metadata.json"))
    monkeypatch.setattr(
        experiment_runner,
        "candidate_items_path",
        lambda strategy: candidate_path,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "article_product_map.*customer_profile.*--force-candidates|"
            "customer_profile.*article_product_map.*--force-candidates"
        ),
    ):
        experiment_runner._validate_candidate_items_fresh(
            "enhanced_default",
            required_inputs,
        )


def test_enhanced_method_output_metadata_includes_enhanced_feature_partitions(
    tmp_path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "outputs" / "enhanced_pop_similarity_trend"
    output_dir.mkdir(parents=True)
    recommendations_path = output_dir / "recommendations.csv"
    items_path = output_dir / "recommendation_items.parquet"
    params_path = output_dir / "params.json"
    metadata_path = output_dir / "metadata.json"
    recommendations_path.write_text("recommendations", encoding="utf-8")
    pd.DataFrame({"customer_id": ["0000001"]}).to_parquet(items_path, index=False)
    params_path.write_text("{}", encoding="utf-8")
    base_inputs = {
        "recommendation_inputs": str(tmp_path / "recommend_inputs.json"),
        "weekly_transactions": str(tmp_path / "weekly.parquet"),
        "article_attributes": str(tmp_path / "article_attributes.csv"),
        "trend_predictions": str(tmp_path / "predictions.csv"),
        "time_windows": str(tmp_path / "time_windows.parquet"),
        "target_users": str(tmp_path / "target_users.parquet"),
        "user_profile": str(tmp_path / "user_profile.parquet"),
        "customer_profile": str(tmp_path / "customer_profile.parquet"),
        "article_product_map": str(tmp_path / "article_product_map.parquet"),
        "candidate_items": str(tmp_path / "candidate_items.parquet"),
        "candidate_metadata": str(tmp_path / "candidate_metadata.json"),
        "feature_cache_metadata": str(tmp_path / "feature_metadata.json"),
    }
    for path in base_inputs.values():
        Path(path).write_text("input", encoding="utf-8")
    metadata = build_artifact_metadata(
        name="recommendation_method:enhanced_pop_similarity_trend",
        input_artifacts=base_inputs,
        output_artifacts={
            "recommendations": str(recommendations_path),
            "recommendation_items": str(items_path),
            "params": str(params_path),
            "metadata": str(metadata_path),
        },
        schema_version=1,
        algorithm_version="recommendation-method-v1",
        config={
            "method": "enhanced_pop_similarity_trend",
            "top_k": 12,
            "candidate_strategy": "enhanced_default",
            "exclude_seen": True,
            "weights": {
                "pop_score": 0.2,
                "recent_score": 0.2,
                "sim_score": 0.1,
                "trend_score": 0.1,
                "reorder_score": 0.1,
                "variant_score": 0.1,
                "age_pop_score": 0.1,
                "preference_pop_score": 0.05,
                "source_rank_score": 0.025,
                "source_count_score": 0.025,
            },
        },
        row_counts={"candidate_rows": 1},
    )
    metadata["window_summaries"] = [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "candidate_rows": 1,
        }
    ]
    write_json_atomic(metadata, metadata_path)
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: type(
            "OutputPaths",
            (),
            {
                "recommendations": recommendations_path,
                "recommendation_items": items_path,
                "params": params_path,
                "metadata": metadata_path,
            },
        )(),
    )

    with pytest.raises(RuntimeError, match="feature partitions"):
        experiment_runner._validate_method_output_fresh(
            "enhanced_pop_similarity_trend",
            base_inputs,
        )


def test_recommendation_enhanced_payload_records_candidate_and_cache_fingerprints(
    tmp_path,
) -> None:
    artifacts = {
        "enhanced_default_candidates": str(tmp_path / "candidate_items.parquet"),
        "enhanced_default_candidate_metadata": str(
            tmp_path / "candidate_metadata.json"
        ),
        "feature_cache_metadata": str(tmp_path / "feature_metadata.json"),
        "feature_partition_0000": str(tmp_path / "part.parquet"),
        "feature_partition_metadata_0000": str(tmp_path / "metadata.json"),
    }
    for key, path in artifacts.items():
        Path(path).write_text(key, encoding="utf-8")

    payload = enhanced_runner.build_enhanced_experiment_payload(
        comparison_payloads=[],
        search_results=[],
        best_weights={},
        enhanced_metrics={"valid": {}, "test": {}},
        freshness_artifacts=artifacts,
    )

    assert payload["freshness_artifacts"] == artifacts
    assert payload["freshness_fingerprints"] == build_input_fingerprints(artifacts)
