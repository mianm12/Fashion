from __future__ import annotations

import json

import pytest

from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.fingerprints import build_input_fingerprints
from fashion_trend.recommendation.freshness import (
    assert_fresh_metadata,
    build_artifact_metadata,
)


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
