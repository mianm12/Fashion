from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.trend_graph_feature_ablation.runner as ablation_runner
import experiments.trend_graph_feature_ablation.train_runs as train_runs
from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_EXPERIMENT_ID,
    ABLATION_VARIANTS,
    RUN_ARTIFACT_FILENAMES,
    SCHEMA_VERSION,
    SUMMARY_COLUMNS,
)
from experiments.trend_graph_feature_ablation.evaluate import (
    evaluate_variant_predictions,
)
from experiments.trend_graph_feature_ablation.summarize import (
    build_metrics_summary_frame,
    render_metrics_summary_markdown,
)
from experiments.trend_graph_feature_ablation.write_docs import render_experiment_doc
from fashion_trend.foundation.paths import OUTPUT_DIR
from fashion_trend.trend.models.supervised.lightgbm import _build_lightgbm_predictions
from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
from fashion_trend.trend.splits import build_trend_model_split_frames
from tests.trend_samples import sample_trend_model_samples_for_split


def test_run_single_variant_writes_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    monkeypatch.setattr(
        train_runs,
        "STABLE_LIGHTGBM_PARAMS_PATH",
        tmp_path / "missing-stable-params.json",
    )
    output_dir = tmp_path / "runs" / "no_graph"

    metadata = train_runs.run_single_variant(
        "no_graph",
        _split_frames(),
        output_dir=output_dir,
        input_hashes={"enhanced_samples_train": {"hash": "abc"}},
        experiment_root=tmp_path,
    )

    assert set(_artifact_names(output_dir)) == {
        "predictions.csv",
        "feature_importance.csv",
        "metadata.json",
        "params.json",
        "model.txt",
    }
    assert metadata["variant"] == "no_graph"
    assert metadata["feature_mask_digest"]
    assert metadata["best_iteration"] == 7
    assert metadata["training_elapsed_seconds"] >= 0
    assert metadata["attr_type_categories"] == ["colour_group_name"]
    assert metadata["output_dir"] == str(output_dir.resolve(strict=False))

    predictions = pd.read_csv(
        output_dir / "predictions.csv",
        dtype={"attr_id": str},
    )
    assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)

    params = json.loads((output_dir / "params.json").read_text(encoding="utf-8"))
    assert params["lightgbm_params"]["objective"] == "regression_l1"
    assert params["early_stopping"] == {"stopping_rounds": 30}
    assert params["param_source"]["default"] == "builtin"
    assert params["best_iteration"] == 7
    assert params["feature_mask"] == metadata["feature_mask"]

    importance = pd.read_csv(output_dir / "feature_importance.csv")
    assert {
        "feature",
        "split_importance",
        "gain_importance",
        "normalized_gain_importance",
    }.issubset(importance.columns)
    assert (output_dir / "model.txt").read_text(encoding="utf-8") == (
        "fake lightgbm model"
    )


def test_run_single_variant_rejects_unknown_valid_attr_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    split_frames = _split_frames()
    split_frames["valid"] = split_frames["valid"].copy()
    split_frames["valid"].loc[
        split_frames["valid"].index[0],
        "attr_type",
    ] = "unknown_type"

    with pytest.raises(ValueError, match="unknown attr_type"):
        train_runs.run_single_variant(
            "no_graph",
            split_frames,
            output_dir=tmp_path / "runs" / "no_graph",
            input_hashes={},
            experiment_root=tmp_path,
        )


def test_run_single_variant_rejects_unknown_test_attr_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    split_frames = _split_frames()
    split_frames["test"] = split_frames["test"].copy()
    split_frames["test"].loc[
        split_frames["test"].index[0],
        "attr_type",
    ] = "unknown_type"

    with pytest.raises(ValueError, match="unknown attr_type"):
        train_runs.run_single_variant(
            "current_coarse_graph",
            split_frames,
            output_dir=tmp_path / "runs" / "current_coarse_graph",
            input_hashes={},
            experiment_root=tmp_path,
        )


def test_run_single_variant_rejects_forbidden_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    forbidden_dir = OUTPUT_DIR / "models" / "lightgbm" / "runs" / "x"

    with pytest.raises(ValueError, match="禁止写入稳定产物路径"):
        train_runs.run_single_variant(
            "no_graph",
            _split_frames(),
            output_dir=forbidden_dir,
            input_hashes={},
            experiment_root=OUTPUT_DIR / "models" / "lightgbm",
        )

    assert not (forbidden_dir / "metadata.json").exists()


def test_run_single_variant_rejects_non_finite_numeric_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    split_frames = _split_frames()
    split_frames["train"] = split_frames["train"].copy()
    split_frames["train"].loc[
        split_frames["train"].index[0],
        "growth_lag_1",
    ] = np.inf

    with pytest.raises(ValueError, match="非有限值: growth_lag_1"):
        train_runs.run_single_variant(
            "no_graph",
            split_frames,
            output_dir=tmp_path / "runs" / "no_graph",
            input_hashes={},
            experiment_root=tmp_path,
        )


def test_evaluate_variant_predictions_writes_metrics_with_recall(
    tmp_path: Path,
) -> None:
    predictions = _build_standard_predictions()
    output_path = tmp_path / "runs" / "no_graph" / "metrics.json"
    prediction_path = tmp_path / "runs" / "no_graph" / "predictions.csv"

    payload = evaluate_variant_predictions(
        "no_graph",
        predictions,
        prediction_path=prediction_path,
        output_path=output_path,
        experiment_root=tmp_path,
    )

    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == payload
    assert payload["run_id"] == "no_graph"
    assert "10" in payload["overall"]["valid"]["recall_at_k"]


def test_evaluate_variant_predictions_rejects_forbidden_output_path(
    tmp_path: Path,
) -> None:
    predictions = _build_standard_predictions()
    forbidden_path = OUTPUT_DIR / "metrics" / "lightgbm" / "runs" / "x" / "metrics.json"

    with pytest.raises(ValueError, match="禁止写入稳定产物路径"):
        evaluate_variant_predictions(
            "no_graph",
            predictions,
            prediction_path=tmp_path / "predictions.csv",
            output_path=forbidden_path,
            experiment_root=tmp_path,
        )

    assert not forbidden_path.exists()


def test_build_metrics_summary_frame_reads_recall_and_preserves_contract_order() -> (
    None
):
    metrics_payloads = {
        variant: _metrics_payload(
            valid_recall=0.10 + index,
            test_recall=0.20 + index,
        )
        for index, variant in enumerate(ABLATION_VARIANTS)
    }
    metadata_payloads = {
        variant: _metadata_payload(index)
        for index, variant in enumerate(ABLATION_VARIANTS)
    }

    summary = build_metrics_summary_frame(metrics_payloads, metadata_payloads)

    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert tuple(summary["variant"]) == ABLATION_VARIANTS
    assert summary["valid_recall_at_10"].tolist() == [
        0.10 + index for index, _variant in enumerate(ABLATION_VARIANTS)
    ]
    assert summary["feature_count"].tolist() == [
        3 + index for index, _variant in enumerate(ABLATION_VARIANTS)
    ]


def test_build_metrics_summary_frame_rejects_missing_variant_or_recall() -> None:
    metrics_payloads = {
        variant: _metrics_payload(valid_recall=0.1, test_recall=0.2)
        for variant in ABLATION_VARIANTS
    }
    metadata_payloads = {
        variant: _metadata_payload(index)
        for index, variant in enumerate(ABLATION_VARIANTS)
    }

    missing_metrics = dict(metrics_payloads)
    missing_metrics.pop("no_graph")
    with pytest.raises(ValueError, match="缺少 metrics payload"):
        build_metrics_summary_frame(missing_metrics, metadata_payloads)

    missing_recall = dict(metrics_payloads)
    missing_recall["no_graph"] = _metrics_payload(valid_recall=0.1, test_recall=0.2)
    del missing_recall["no_graph"]["overall"]["valid"]["recall_at_k"]
    with pytest.raises(ValueError, match="缺少指标"):
        build_metrics_summary_frame(missing_recall, metadata_payloads)


def test_build_metrics_summary_frame_validates_metadata_scalars() -> None:
    metrics_payloads, metadata_payloads = _summary_payloads()
    metadata_payloads["no_graph"]["best_iteration"] = None
    metadata_payloads["current_coarse_graph"]["best_iteration"] = 7.0

    summary = build_metrics_summary_frame(metrics_payloads, metadata_payloads)

    assert (
        summary.loc[summary["variant"] == "no_graph", "best_iteration"].item() is None
    )
    assert (
        summary.loc[
            summary["variant"] == "current_coarse_graph",
            "best_iteration",
        ].item()
        == 7
    )

    bad_best_iteration = dict(metadata_payloads)
    bad_best_iteration["no_graph"] = dict(metadata_payloads["no_graph"])
    bad_best_iteration["no_graph"]["best_iteration"] = "7"
    with pytest.raises(ValueError, match="no_graph.*best_iteration"):
        build_metrics_summary_frame(metrics_payloads, bad_best_iteration)

    bad_elapsed = dict(metadata_payloads)
    bad_elapsed["no_graph"] = dict(metadata_payloads["no_graph"])
    bad_elapsed["no_graph"]["training_elapsed_seconds"] = math.inf
    with pytest.raises(ValueError, match="no_graph.*training_elapsed_seconds"):
        build_metrics_summary_frame(metrics_payloads, bad_elapsed)


def test_build_metrics_summary_frame_validates_metric_values_and_keys() -> None:
    metrics_payloads, metadata_payloads = _summary_payloads()

    string_spearman = _summary_payloads()[0]
    string_spearman["no_graph"]["overall"]["valid"]["spearman"] = "0.5"
    with pytest.raises(ValueError, match="no_graph.*valid.*spearman"):
        build_metrics_summary_frame(string_spearman, metadata_payloads)

    nan_recall = _summary_payloads()[0]
    nan_recall["no_graph"]["overall"]["valid"]["recall_at_k"]["10"] = np.nan
    with pytest.raises(ValueError, match="no_graph.*valid.*recall_at_k.*10"):
        build_metrics_summary_frame(nan_recall, metadata_payloads)

    missing_nested_key = _summary_payloads()[0]
    del missing_nested_key["no_graph"]["overall"]["valid"]["recall_at_k"]["10"]
    with pytest.raises(ValueError, match="no_graph.*valid.*recall_at_k.*10"):
        build_metrics_summary_frame(missing_nested_key, metadata_payloads)

    missing_valid_split = _summary_payloads()[0]
    del missing_valid_split["no_graph"]["overall"]["valid"]
    with pytest.raises(ValueError, match="no_graph.*valid"):
        build_metrics_summary_frame(missing_valid_split, metadata_payloads)

    missing_test_split = _summary_payloads()[0]
    del missing_test_split["no_graph"]["overall"]["test"]
    with pytest.raises(ValueError, match="no_graph.*test"):
        build_metrics_summary_frame(missing_test_split, metadata_payloads)

    none_metrics = _summary_payloads()[0]
    none_metrics["no_graph"]["overall"]["valid"]["spearman"] = None
    none_metrics["no_graph"]["overall"]["valid"]["recall_at_k"]["10"] = None
    summary = build_metrics_summary_frame(none_metrics, metadata_payloads)
    row = summary.loc[summary["variant"] == "no_graph"].iloc[0]
    assert row["valid_spearman"] is None
    assert row["valid_recall_at_10"] is None


def test_render_metrics_summary_markdown_contains_recall_and_trailing_newline() -> None:
    summary = pd.DataFrame(
        [{column: "value" if column == "variant" else 1 for column in SUMMARY_COLUMNS}],
        columns=SUMMARY_COLUMNS,
    )

    markdown = render_metrics_summary_markdown(summary)

    assert "valid_recall_at_10" in markdown
    assert markdown.endswith("\n")


def test_render_metrics_summary_markdown_falls_back_without_tabulate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = pd.DataFrame(
        [
            {
                column: "no_graph" if column == "variant" else 1
                for column in SUMMARY_COLUMNS
            }
        ],
        columns=SUMMARY_COLUMNS,
    )

    def raise_import_error(self, *args, **kwargs):
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", raise_import_error)

    markdown = render_metrics_summary_markdown(summary)

    assert "valid_recall_at_10" in markdown
    assert "no_graph" in markdown
    assert markdown.endswith("\n")


def test_render_experiment_doc_contains_non_stable_boundaries() -> None:
    document = render_experiment_doc(
        "| variant | valid_ndcg_at_10 |\n|---|---:|\n| no_graph | 0.1 |\n",
        ["uv", "run", "python", "src/19_run_trend_graph_feature_ablation.py"],
    )

    assert "非 stable 独立实验" in document
    assert "不覆盖 `outputs/models/lightgbm/`" in document
    assert "不写 `outputs/reports/manifest.json`" in document
    assert "不改变 defense app 数据源" in document
    assert "reports experimental" in document
    assert "src/19_run_trend_graph_feature_ablation.py" in document


def test_run_trend_graph_feature_ablation_publishes_expected_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root = tmp_path / ABLATION_EXPERIMENT_ID
    monkeypatch.setattr(
        ablation_runner,
        "_load_input_frames",
        _fake_runner_inputs,
    )
    monkeypatch.setattr(
        ablation_runner,
        "_build_input_hashes",
        _fake_input_hashes,
    )
    monkeypatch.setattr(
        ablation_runner,
        "build_enhanced_sample_frames",
        _fake_enhanced_sample_frames,
    )
    monkeypatch.setattr(
        ablation_runner,
        "run_single_variant",
        _fake_run_single_variant,
    )
    monkeypatch.setattr(
        ablation_runner,
        "evaluate_variant_predictions",
        _fake_evaluate_variant_predictions,
    )

    payload = ablation_runner.run_trend_graph_feature_ablation(
        command=["uv", "run", "python", "src/19_run_trend_graph_feature_ablation.py"],
        experiment_root=experiment_root,
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["experiment_id"] == ABLATION_EXPERIMENT_ID
    assert payload["metrics_summary_path"] == str(
        experiment_root / "metrics_summary.md"
    )
    assert not (experiment_root / ".staging").exists()

    expected_paths = [
        experiment_root / "features" / f"enhanced_samples_{split}.parquet"
        for split in ("all", "train", "valid", "test")
    ]
    expected_paths.extend(
        [
            experiment_root / "features" / "feature_groups.json",
            experiment_root / "features" / "feature_schema.json",
            experiment_root / "features" / "row_alignment_check.json",
            experiment_root / "features" / "input_hashes.json",
            experiment_root / "metrics_summary.csv",
            experiment_root / "metrics_summary.md",
            experiment_root / "experiment.md",
            experiment_root / "manifest.json",
        ]
    )
    for variant in ABLATION_VARIANTS:
        expected_paths.extend(
            experiment_root / "runs" / variant / filename
            for filename in RUN_ARTIFACT_FILENAMES
        )
    missing = [path for path in expected_paths if not path.exists()]
    assert missing == []

    input_hashes = json.loads(
        (experiment_root / "features" / "input_hashes.json").read_text(encoding="utf-8")
    )
    assert input_hashes["stable_lightgbm_params"]["exists"] is False
    assert input_hashes["stable_lightgbm_params"]["required"] is False

    feature_schema = json.loads(
        (experiment_root / "features" / "feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert feature_schema["schema_version"] == SCHEMA_VERSION
    assert isinstance(feature_schema["features"], list)

    manifest = json.loads(
        (experiment_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["experiment_id"] == ABLATION_EXPERIMENT_ID
    assert manifest["command"] == [
        "uv",
        "run",
        "python",
        "src/19_run_trend_graph_feature_ablation.py",
    ]
    assert manifest["variants"] == list(ABLATION_VARIANTS)
    assert manifest["warnings"] == []
    assert manifest["row_alignment"] == {"passed": True}
    assert manifest["metrics_summary_path"] == str(
        experiment_root / "metrics_summary.md"
    )

    rendered_doc = (experiment_root / "experiment.md").read_text(encoding="utf-8")
    assert "不覆盖 `outputs/models/lightgbm/`" in rendered_doc
    assert "不写 `outputs/reports/manifest.json`" in rendered_doc


def test_run_trend_graph_feature_ablation_does_not_publish_on_stage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root = tmp_path / ABLATION_EXPERIMENT_ID
    monkeypatch.setattr(
        ablation_runner,
        "_load_input_frames",
        _fake_runner_inputs,
    )
    monkeypatch.setattr(
        ablation_runner,
        "_build_input_hashes",
        _fake_input_hashes,
    )

    def fail_build(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        ablation_runner,
        "build_enhanced_sample_frames",
        fail_build,
    )

    with pytest.raises(RuntimeError, match="boom"):
        ablation_runner.run_trend_graph_feature_ablation(
            command="test command",
            experiment_root=experiment_root,
        )

    assert not (experiment_root / "manifest.json").exists()
    assert not (experiment_root / "metrics_summary.md").exists()


def test_trend_graph_feature_ablation_cli_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_ablation_cli_module()
    observed: dict[str, object] = {}

    def fake_success(*, command):
        observed["command"] = command
        return {
            "schema_version": SCHEMA_VERSION,
            "metrics_summary_path": "outputs/experiments/x/metrics_summary.md",
        }

    monkeypatch.setattr(cli, "run_trend_graph_feature_ablation", fake_success)

    assert cli.main([]) == 0
    assert observed["command"] == [
        "uv",
        "run",
        "python",
        "src/19_run_trend_graph_feature_ablation.py",
    ]

    def fake_failure(*, command):
        raise RuntimeError("failed")

    monkeypatch.setattr(cli, "run_trend_graph_feature_ablation", fake_failure)

    assert cli.main([]) == 1


def test_trend_graph_feature_ablation_manifest_stays_under_experiment_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root = tmp_path / ABLATION_EXPERIMENT_ID
    monkeypatch.setattr(
        ablation_runner,
        "_load_input_frames",
        _fake_runner_inputs,
    )
    monkeypatch.setattr(
        ablation_runner,
        "_build_input_hashes",
        _fake_input_hashes,
    )
    monkeypatch.setattr(
        ablation_runner,
        "build_enhanced_sample_frames",
        _fake_enhanced_sample_frames,
    )
    monkeypatch.setattr(
        ablation_runner,
        "run_single_variant",
        _fake_run_single_variant,
    )
    monkeypatch.setattr(
        ablation_runner,
        "evaluate_variant_predictions",
        _fake_evaluate_variant_predictions,
    )

    ablation_runner.run_trend_graph_feature_ablation(
        command="test command",
        experiment_root=experiment_root,
    )

    manifest = json.loads(
        (experiment_root / "manifest.json").read_text(encoding="utf-8")
    )
    artifact_paths = [
        Path(value)
        for value in _flatten_manifest_values(manifest["artifacts"])
        if isinstance(value, str)
    ]
    assert artifact_paths
    assert all(
        path.resolve(strict=False).is_relative_to(experiment_root.resolve(strict=False))
        for path in artifact_paths
    )
    forbidden_fragments = (
        "outputs/models/lightgbm",
        "outputs/reports",
        "outputs/defense_app",
        "apps/defense_app",
        "data/processed/features",
    )
    assert not any(
        fragment in str(path)
        for path in artifact_paths
        for fragment in forbidden_fragments
    )


def _split_frames() -> dict[str, pd.DataFrame]:
    return build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )


def _build_standard_predictions() -> pd.DataFrame:
    split_frames = _split_frames()
    predictions = pd.concat(
        [
            _build_lightgbm_predictions(
                split_frame,
                split_frame["target_growth"].astype(float),
            )
            for split_frame in split_frames.values()
        ],
        ignore_index=True,
    )
    return predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]


def _metrics_payload(*, valid_recall: float, test_recall: float) -> dict[str, object]:
    def split_metrics(recall: float) -> dict[str, object]:
        return {
            "ndcg_at_k": {"10": 0.5},
            "spearman": 0.6,
            "precision_at_k": {"10": 0.7},
            "recall_at_k": {"10": recall},
        }

    return {
        "overall": {
            "valid": split_metrics(valid_recall),
            "test": split_metrics(test_recall),
        }
    }


def _metadata_payload(index: int) -> dict[str, object]:
    return {
        "feature_mask": {
            "numeric_features": [
                f"num_{index}_{offset}" for offset in range(index + 2)
            ],
            "categorical_features": [f"cat_{index}"],
        },
        "best_iteration": index + 1,
        "training_elapsed_seconds": float(index) + 0.5,
    }


def _summary_payloads() -> (
    tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]
):
    metrics_payloads = {
        variant: _metrics_payload(valid_recall=0.1, test_recall=0.2)
        for variant in ABLATION_VARIANTS
    }
    metadata_payloads = {
        variant: _metadata_payload(index)
        for index, variant in enumerate(ABLATION_VARIANTS)
    }
    return metrics_payloads, metadata_payloads


def _artifact_names(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.iterdir() if path.is_file())


def _fake_runner_inputs() -> ablation_runner.TrendGraphAblationInputs:
    samples_all = pd.DataFrame(
        {
            "week_id": [1, 2, 3],
            "attr_id": ["a", "b", "c"],
            "target_growth": [0.1, 0.2, 0.3],
            "target_log_heat_t1": [1.0, 1.1, 1.2],
            "target_rank_in_type_t1": [1, 2, 3],
        }
    )
    split_samples = {
        split: samples_all.iloc[[index]].copy().assign(split=split)
        for index, split in enumerate(("train", "valid", "test"))
    }
    return ablation_runner.TrendGraphAblationInputs(
        samples_all=samples_all,
        split_samples=split_samples,
        hierarchy_edges=pd.DataFrame(
            {
                "parent_attr_id": ["a"],
                "child_attr_id": ["b"],
                "edge_weight": [1.0],
            }
        ),
    )


def _fake_input_hashes(
    inputs: ablation_runner.TrendGraphAblationInputs,
) -> dict[str, dict[str, object]]:
    return {
        "trend_model_samples": {"exists": True, "hash": "all", "row_count": 3},
        "trend_model_samples_train": {"exists": True, "hash": "train", "row_count": 1},
        "trend_model_samples_valid": {"exists": True, "hash": "valid", "row_count": 1},
        "trend_model_samples_test": {"exists": True, "hash": "test", "row_count": 1},
        "graph_edges_attribute_hierarchy": {
            "exists": True,
            "hash": "edges",
            "row_count": 1,
        },
        "graph_nodes_attribute": {"exists": True, "hash": "nodes", "row_count": None},
        "stable_lightgbm_params": {
            "exists": False,
            "hash": None,
            "row_count": None,
            "required": False,
        },
    }


def _fake_enhanced_sample_frames(
    samples_all: pd.DataFrame,
    split_samples: dict[str, pd.DataFrame],
    hierarchy_edges: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "all": samples_all.copy().assign(kg_dummy=1.0),
        **{
            split: frame.copy().assign(kg_dummy=1.0)
            for split, frame in split_samples.items()
        },
    }


def _fake_run_single_variant(
    variant: str,
    split_frames,
    *,
    output_dir: Path,
    input_hashes,
    experiment_root: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = ABLATION_VARIANTS.index(variant)
    pd.DataFrame(
        {
            "week_id": [1],
            "attr_type": ["colour_group_name"],
            "attr_id": ["a"],
            "target_growth": [0.1],
            "prediction": [0.1 + index],
            "split": ["valid"],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["growth_lag_1"],
            "split_importance": [1],
            "gain_importance": [1.0],
            "normalized_gain_importance": [1.0],
        }
    ).to_csv(output_dir / "feature_importance.csv", index=False)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "feature_mask": {
            "numeric_features": ["growth_lag_1", f"feature_{index}"],
            "categorical_features": ["attr_type"],
        },
        "best_iteration": index + 1,
        "training_elapsed_seconds": float(index) + 0.5,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "params.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "variant": variant}),
        encoding="utf-8",
    )
    (output_dir / "model.txt").write_text("fake model", encoding="utf-8")
    return metadata


def _fake_evaluate_variant_predictions(
    variant: str,
    predictions: pd.DataFrame,
    *,
    prediction_path: Path,
    output_path: Path,
    experiment_root: Path,
) -> dict[str, object]:
    index = ABLATION_VARIANTS.index(variant)
    payload = _metrics_payload(valid_recall=0.1 + index, test_recall=0.2 + index)
    payload["overall"]["valid"]["ndcg_at_k"]["10"] = 0.3 + index
    payload["overall"]["test"]["ndcg_at_k"]["10"] = 0.4 + index
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _flatten_manifest_values(value: object) -> list[object]:
    if isinstance(value, dict):
        flattened: list[object] = []
        for nested in value.values():
            flattened.extend(_flatten_manifest_values(nested))
        return flattened
    if isinstance(value, list):
        flattened = []
        for nested in value:
            flattened.extend(_flatten_manifest_values(nested))
        return flattened
    return [value]


def _load_ablation_cli_module():
    module_name = "_trend_graph_feature_ablation_cli_for_tests"
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "19_run_trend_graph_feature_ablation.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_lightgbm_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fit(
        train_features,
        train_target,
        valid_features,
        valid_target,
        *,
        config,
    ):
        assert len(train_features) == len(train_target)
        assert len(valid_features) == len(valid_target)
        assert config.lightgbm_params["objective"] == "regression_l1"
        assert str(train_features["attr_type"].dtype) == "category"
        assert str(valid_features["attr_type"].dtype) == "category"
        return _FakeLightGBMModel(train_features.columns.tolist())

    monkeypatch.setattr(train_runs, "_fit_lightgbm_model", fake_fit)


class _FakeBooster:
    def __init__(self, feature_names: list[str]) -> None:
        self._feature_names = list(feature_names)

    def feature_name(self) -> list[str]:
        return list(self._feature_names)

    def feature_importance(self, importance_type: str) -> list[float] | list[int]:
        if importance_type == "split":
            return [1 for _ in self._feature_names]
        if importance_type == "gain":
            return [float(index + 1) for index, _ in enumerate(self._feature_names)]
        raise AssertionError(f"unexpected importance_type={importance_type}")

    def model_to_string(self) -> str:
        return "fake lightgbm model"


class _FakeLightGBMModel:
    best_iteration_ = 7

    def __init__(self, feature_names: list[str]) -> None:
        self.booster_ = _FakeBooster(feature_names)

    def predict(self, features: pd.DataFrame, num_iteration: int | None = None):
        assert num_iteration == 7
        return features["growth_lag_1"].astype(float).to_numpy()
