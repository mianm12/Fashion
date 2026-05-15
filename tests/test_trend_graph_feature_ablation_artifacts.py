from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.trend_graph_feature_ablation import artifact_io, contracts, paths
from fashion_trend.foundation.paths import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS


def test_contract_constants_keep_stable_order() -> None:
    assert contracts.SCHEMA_VERSION == "trend_graph_feature_ablation.v1"
    assert contracts.ABLATION_EXPERIMENT_ID == "trend_graph_feature_ablation"
    assert contracts.ABLATION_VARIANTS == (
        "no_graph",
        "current_coarse_graph",
        "full_enhanced",
        "wo_hierarchy_context",
        "wo_sibling_competition",
    )
    assert contracts.FEATURE_GROUP_NAMES == (
        "base_numeric_non_graph",
        "categorical",
        "coarse_graph",
        "hierarchy_context",
        "sibling_competition",
        "light_structure",
    )
    assert contracts.TARGET_COLUMNS == (
        "target_growth",
        "target_log_heat_t1",
        "target_rank_in_type_t1",
    )
    assert contracts.ALL_SAMPLE_KEY_COLUMNS == ("week_id", "attr_id")
    assert contracts.SPLIT_SAMPLE_KEY_COLUMNS == ("split", "week_id", "attr_id")
    assert contracts.PREDICTION_COLUMNS == TREND_MODEL_PREDICTION_COLUMNS
    assert contracts.SUMMARY_COLUMNS == (
        "variant",
        "feature_count",
        "best_iteration",
        "training_elapsed_seconds",
        "valid_ndcg_at_10",
        "valid_spearman",
        "valid_precision_at_10",
        "valid_recall_at_10",
        "test_ndcg_at_10",
        "test_spearman",
        "test_precision_at_10",
        "test_recall_at_10",
    )
    assert contracts.RUN_ARTIFACT_FILENAMES == (
        "predictions.csv",
        "metrics.json",
        "feature_importance.csv",
        "metadata.json",
        "params.json",
        "model.txt",
    )


def test_paths_stay_under_experiment_root() -> None:
    output_paths = (
        paths.enhanced_sample_path("all"),
        paths.enhanced_sample_path("train"),
        paths.run_artifact_path("full_enhanced", "metrics.json"),
        paths.staging_run_dir("full_enhanced"),
        paths.FEATURE_GROUPS_PATH,
        paths.FEATURE_SCHEMA_PATH,
        paths.ROW_ALIGNMENT_CHECK_PATH,
        paths.INPUT_HASHES_PATH,
        paths.METRICS_SUMMARY_CSV_PATH,
        paths.METRICS_SUMMARY_MD_PATH,
        paths.EXPERIMENT_DOC_PATH,
        paths.MANIFEST_PATH,
    )

    assert paths.enhanced_sample_path("all") == (
        paths.FEATURES_DIR / "enhanced_samples_all.parquet"
    )
    assert paths.enhanced_sample_path("train") == (
        paths.FEATURES_DIR / "enhanced_samples_train.parquet"
    )
    assert paths.enhanced_sample_path("valid") == (
        paths.FEATURES_DIR / "enhanced_samples_valid.parquet"
    )
    assert paths.enhanced_sample_path("test") == (
        paths.FEATURES_DIR / "enhanced_samples_test.parquet"
    )

    for output_path in output_paths:
        assert output_path.resolve(strict=False).is_relative_to(
            paths.EXPERIMENT_ROOT.resolve(strict=False)
        )


def test_default_write_guard_accepts_experiment_root_output() -> None:
    guarded_path = artifact_io.assert_experiment_write_path(
        paths.METRICS_SUMMARY_CSV_PATH
    )

    assert guarded_path == paths.METRICS_SUMMARY_CSV_PATH.resolve(strict=False)


def test_write_guard_accepts_injected_tmp_root(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    output_path = root / "runs" / "full_enhanced" / "metrics.json"

    guarded_path = artifact_io.assert_experiment_write_path(output_path, root=root)

    assert guarded_path == output_path.resolve(strict=False)


def test_write_guard_rejects_path_outside_injected_tmp_root(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    outside_path = tmp_path / "outside" / "metrics.json"

    with pytest.raises(ValueError, match="不在允许根目录内"):
        artifact_io.assert_experiment_write_path(outside_path, root=root)


@pytest.mark.parametrize(
    "forbidden_path",
    (
        OUTPUT_DIR / "models" / "lightgbm" / "predictions.csv",
        OUTPUT_DIR / "metrics" / "lightgbm" / "trend_metrics.json",
        OUTPUT_DIR / "reports" / "manifest.json",
        OUTPUT_DIR / "defense_app" / "fashion_demo.sqlite",
        PROJECT_ROOT / "apps" / "defense_app" / "local.sqlite",
        DATA_DIR / "processed" / "features" / "trend_model_samples.parquet",
    ),
)
def test_default_write_guard_rejects_stable_outputs(forbidden_path: Path) -> None:
    with pytest.raises(ValueError):
        artifact_io.assert_experiment_write_path(forbidden_path)


def test_write_guard_rejects_forbidden_path_even_with_forbidden_root() -> None:
    forbidden_root = OUTPUT_DIR / "models" / "lightgbm"
    forbidden_path = forbidden_root / "predictions.csv"

    with pytest.raises(ValueError, match="禁止写入稳定产物路径"):
        artifact_io.assert_experiment_write_path(forbidden_path, root=forbidden_root)


def test_write_guard_rejects_forbidden_path_with_regular_tmp_root(
    tmp_path: Path,
) -> None:
    forbidden_path = OUTPUT_DIR / "models" / "lightgbm" / "predictions.csv"

    with pytest.raises(ValueError, match="禁止写入稳定产物路径"):
        artifact_io.assert_experiment_write_path(forbidden_path, root=tmp_path)


def test_existing_hash_entry_records_file_metadata(tmp_path: Path) -> None:
    content = b"trend graph feature ablation\n"
    file_path = tmp_path / "input.csv"
    file_path.write_bytes(content)

    entry = artifact_io.build_input_hash_entry(file_path, row_count=12)

    assert entry["path"] == str(file_path)
    assert entry["exists"] is True
    assert entry["hash"] == hashlib.sha256(content).hexdigest()
    assert entry["size"] == len(content)
    assert entry["mtime"] == file_path.stat().st_mtime
    assert entry["row_count"] == 12


def test_optional_missing_hash_entry_records_absence(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.parquet"

    entry = artifact_io.build_input_hash_entry(
        missing_path,
        required=False,
        row_count=None,
    )

    assert entry == {
        "path": str(missing_path),
        "exists": False,
        "hash": None,
        "size": None,
        "mtime": None,
        "row_count": None,
    }


def test_required_missing_hash_entry_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        artifact_io.build_input_hash_entry(missing_path)


def test_digest_json_payload_is_stable_for_key_order() -> None:
    left = artifact_io.digest_json_payload({"b": 2, "a": {"d": 4, "c": 3}})
    right = artifact_io.digest_json_payload({"a": {"c": 3, "d": 4}, "b": 2})

    assert left == right
