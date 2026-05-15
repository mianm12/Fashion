from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.trend_graph_feature_ablation import paths as ablation_paths
from experiments.trend_graph_feature_ablation.artifact_io import (
    assert_experiment_write_path,
    build_input_hash_entry,
)
from experiments.trend_graph_feature_ablation.build_features import (
    build_enhanced_sample_frames,
    build_row_alignment_check,
)
from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_EXPERIMENT_ID,
    ABLATION_VARIANTS,
    RUN_ARTIFACT_FILENAMES,
    SCHEMA_VERSION,
)
from experiments.trend_graph_feature_ablation.evaluate import (
    evaluate_variant_predictions,
)
from experiments.trend_graph_feature_ablation.feature_groups import (
    build_feature_groups_payload,
)
from experiments.trend_graph_feature_ablation.summarize import (
    build_metrics_summary_frame,
    render_metrics_summary_markdown,
)
from experiments.trend_graph_feature_ablation.train_runs import (
    STABLE_LIGHTGBM_PARAMS_PATH,
    run_single_variant,
)
from experiments.trend_graph_feature_ablation.write_docs import (
    render_experiment_doc,
)
from fashion_trend.catalog.paths import (
    GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH,
    GRAPH_NODES_ATTRIBUTE_PATH,
)
from fashion_trend.catalog.readers import read_attribute_hierarchy_edges
from fashion_trend.foundation.io import (
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from fashion_trend.trend.features.samples import validate_trend_model_samples
from fashion_trend.trend.paths import (
    TREND_MODEL_SAMPLES_PATH,
    TREND_MODEL_SAMPLES_TEST_PATH,
    TREND_MODEL_SAMPLES_TRAIN_PATH,
    TREND_MODEL_SAMPLES_VALID_PATH,
)
from fashion_trend.trend.splits import read_trend_model_split

EXPERIMENT_ROOT = ablation_paths.EXPERIMENT_ROOT

_SPLIT_PATHS: Mapping[str, Path] = {
    "train": TREND_MODEL_SAMPLES_TRAIN_PATH,
    "valid": TREND_MODEL_SAMPLES_VALID_PATH,
    "test": TREND_MODEL_SAMPLES_TEST_PATH,
}


@dataclass(frozen=True)
class TrendGraphAblationInputs:
    samples_all: pd.DataFrame
    split_samples: dict[str, pd.DataFrame]
    hierarchy_edges: pd.DataFrame


@dataclass(frozen=True)
class ExperimentOutputPaths:
    root: Path

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def metrics_summary_csv(self) -> Path:
        return self.root / "metrics_summary.csv"

    @property
    def metrics_summary_md(self) -> Path:
        return self.root / "metrics_summary.md"

    @property
    def experiment_doc(self) -> Path:
        return self.root / "experiment.md"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def enhanced_sample(self, split: str) -> Path:
        return self.features_dir / f"enhanced_samples_{split}.parquet"

    def feature_groups(self) -> Path:
        return self.features_dir / "feature_groups.json"

    def feature_schema(self) -> Path:
        return self.features_dir / "feature_schema.json"

    def row_alignment_check(self) -> Path:
        return self.features_dir / "row_alignment_check.json"

    def input_hashes(self) -> Path:
        return self.features_dir / "input_hashes.json"

    def run_dir(self, variant: str) -> Path:
        return self.runs_dir / variant

    def run_artifact(self, variant: str, filename: str) -> Path:
        return self.run_dir(variant) / filename


def run_trend_graph_feature_ablation(
    *,
    command: str | Sequence[str] | None = None,
    experiment_root: Path | None = None,
) -> dict[str, object]:
    """编排趋势图特征消融独立实验，并只发布到实验根目录。"""

    root = assert_experiment_write_path(
        Path(experiment_root or EXPERIMENT_ROOT),
        root=experiment_root or EXPERIMENT_ROOT,
    )
    final_paths = ExperimentOutputPaths(root=root)
    staging_root = root / ".staging" / uuid.uuid4().hex
    staging_paths = ExperimentOutputPaths(root=staging_root)
    assert_experiment_write_path(staging_root, root=root)

    try:
        inputs = _load_input_frames()
        input_hashes = _build_input_hashes(inputs)
        enhanced_frames = build_enhanced_sample_frames(
            inputs.samples_all,
            inputs.split_samples,
            inputs.hierarchy_edges,
        )
        row_alignment = build_row_alignment_check(
            inputs.samples_all,
            inputs.split_samples,
            enhanced_frames,
        )
        if not bool(row_alignment["passed"]):
            raise RuntimeError(f"趋势图特征消融行对齐校验失败: {row_alignment}")

        feature_groups = build_feature_groups_payload()
        _write_feature_artifacts(
            staging_paths,
            root=root,
            enhanced_frames=enhanced_frames,
            feature_groups=feature_groups,
            input_hashes=input_hashes,
            row_alignment=row_alignment,
        )
        metrics_payloads, metadata_payloads = _run_variants(
            staging_paths,
            root=root,
            split_frames={
                split_name: enhanced_frames[split_name]
                for split_name in ("train", "valid", "test")
            },
            input_hashes=input_hashes,
        )
        summary_frame = build_metrics_summary_frame(metrics_payloads, metadata_payloads)
        summary_markdown = render_metrics_summary_markdown(summary_frame)
        _write_summary_artifacts(
            staging_paths,
            root=root,
            summary_frame=summary_frame,
            summary_markdown=summary_markdown,
            command=command,
        )
        manifest = _build_manifest(
            final_paths,
            command=command,
            input_hashes=input_hashes,
            row_alignment=row_alignment,
        )
        _write_json_guarded(manifest, staging_paths.manifest, root=root)

        published_paths = _publish_staged_outputs(
            staging_paths,
            final_paths,
            root=root,
        )
    except Exception:
        _remove_staging_root(staging_root)
        raise

    _remove_staging_root(staging_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": ABLATION_EXPERIMENT_ID,
        "experiment_root": str(root),
        "metrics_summary_path": str(final_paths.metrics_summary_md),
        "manifest_path": str(final_paths.manifest),
        "published_paths": [str(path) for path in published_paths],
        "variants": list(ABLATION_VARIANTS),
        "row_alignment": {"passed": bool(row_alignment["passed"])},
    }


def _load_input_frames() -> TrendGraphAblationInputs:
    samples_all = pd.read_parquet(TREND_MODEL_SAMPLES_PATH)
    validate_trend_model_samples(samples_all)
    _validate_sample_identifier_columns(
        samples_all, source_name="趋势图消融 all samples"
    )
    split_samples = {
        split_name: read_trend_model_split(path)
        for split_name, path in _SPLIT_PATHS.items()
    }
    for split_name, split_frame in split_samples.items():
        _validate_sample_identifier_columns(
            split_frame,
            source_name=f"趋势图消融 {split_name} samples",
        )
    hierarchy_edges = read_attribute_hierarchy_edges(
        GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH
    )
    return TrendGraphAblationInputs(
        samples_all=samples_all,
        split_samples=split_samples,
        hierarchy_edges=hierarchy_edges,
    )


def _build_input_hashes(
    inputs: TrendGraphAblationInputs,
) -> dict[str, dict[str, object]]:
    return {
        "trend_model_samples": build_input_hash_entry(
            TREND_MODEL_SAMPLES_PATH,
            row_count=len(inputs.samples_all),
        ),
        "trend_model_samples_train": build_input_hash_entry(
            TREND_MODEL_SAMPLES_TRAIN_PATH,
            row_count=len(inputs.split_samples["train"]),
        ),
        "trend_model_samples_valid": build_input_hash_entry(
            TREND_MODEL_SAMPLES_VALID_PATH,
            row_count=len(inputs.split_samples["valid"]),
        ),
        "trend_model_samples_test": build_input_hash_entry(
            TREND_MODEL_SAMPLES_TEST_PATH,
            row_count=len(inputs.split_samples["test"]),
        ),
        "graph_edges_attribute_hierarchy": build_input_hash_entry(
            GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH,
            row_count=len(inputs.hierarchy_edges),
        ),
        "graph_nodes_attribute": build_input_hash_entry(
            GRAPH_NODES_ATTRIBUTE_PATH,
            row_count=None,
        ),
        "stable_lightgbm_params": _optional_input_hash_entry(
            STABLE_LIGHTGBM_PARAMS_PATH
        ),
    }


def _optional_input_hash_entry(path: Path) -> dict[str, object]:
    entry = build_input_hash_entry(path, required=False, row_count=None)
    entry["required"] = False
    return entry


def _write_feature_artifacts(
    paths: ExperimentOutputPaths,
    *,
    root: Path,
    enhanced_frames: Mapping[str, pd.DataFrame],
    feature_groups: Mapping[str, object],
    input_hashes: Mapping[str, Mapping[str, object]],
    row_alignment: Mapping[str, object],
) -> None:
    for split_name in ("all", "train", "valid", "test"):
        _write_parquet_guarded(
            enhanced_frames[split_name],
            paths.enhanced_sample(split_name),
            root=root,
        )
    _write_json_guarded(dict(feature_groups), paths.feature_groups(), root=root)
    _write_json_guarded(
        {
            "schema_version": SCHEMA_VERSION,
            "features": list(feature_groups["feature_schema"]),
        },
        paths.feature_schema(),
        root=root,
    )
    _write_json_guarded(dict(row_alignment), paths.row_alignment_check(), root=root)
    _write_json_guarded(
        {key: dict(value) for key, value in input_hashes.items()},
        paths.input_hashes(),
        root=root,
    )


def _run_variants(
    paths: ExperimentOutputPaths,
    *,
    root: Path,
    split_frames: Mapping[str, pd.DataFrame],
    input_hashes: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    metrics_payloads: dict[str, dict[str, object]] = {}
    metadata_payloads: dict[str, dict[str, object]] = {}
    for variant in ABLATION_VARIANTS:
        output_dir = paths.run_dir(variant)
        metadata = run_single_variant(
            variant,
            split_frames,
            output_dir=output_dir,
            input_hashes=input_hashes,
            experiment_root=root,
        )
        predictions = pd.read_csv(
            paths.run_artifact(variant, "predictions.csv"),
            dtype={"attr_id": str},
        )
        metrics = evaluate_variant_predictions(
            variant,
            predictions,
            prediction_path=paths.run_artifact(variant, "predictions.csv"),
            output_path=paths.run_artifact(variant, "metrics.json"),
            experiment_root=root,
        )
        metadata_payloads[variant] = dict(metadata)
        metrics_payloads[variant] = dict(metrics)
    return metrics_payloads, metadata_payloads


def _write_summary_artifacts(
    paths: ExperimentOutputPaths,
    *,
    root: Path,
    summary_frame: pd.DataFrame,
    summary_markdown: str,
    command: str | Sequence[str] | None,
) -> None:
    _write_csv_guarded(summary_frame, paths.metrics_summary_csv, root=root)
    _write_text_guarded(summary_markdown, paths.metrics_summary_md, root=root)
    _write_text_guarded(
        render_experiment_doc(summary_markdown, command),
        paths.experiment_doc,
        root=root,
    )


def _build_manifest(
    paths: ExperimentOutputPaths,
    *,
    command: str | Sequence[str] | None,
    input_hashes: Mapping[str, Mapping[str, object]],
    row_alignment: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": ABLATION_EXPERIMENT_ID,
        "command": _command_payload(command),
        "input_hashes": {key: dict(value) for key, value in input_hashes.items()},
        "variants": list(ABLATION_VARIANTS),
        "artifacts": _artifact_paths_payload(paths),
        "metrics_summary_path": str(paths.metrics_summary_md),
        "warnings": [],
        "row_alignment": {"passed": bool(row_alignment["passed"])},
    }


def _artifact_paths_payload(paths: ExperimentOutputPaths) -> dict[str, object]:
    return {
        "features": {
            "enhanced_samples_all": str(paths.enhanced_sample("all")),
            "enhanced_samples_train": str(paths.enhanced_sample("train")),
            "enhanced_samples_valid": str(paths.enhanced_sample("valid")),
            "enhanced_samples_test": str(paths.enhanced_sample("test")),
            "feature_groups": str(paths.feature_groups()),
            "feature_schema": str(paths.feature_schema()),
            "row_alignment_check": str(paths.row_alignment_check()),
            "input_hashes": str(paths.input_hashes()),
        },
        "runs": {
            variant: {
                filename: str(paths.run_artifact(variant, filename))
                for filename in RUN_ARTIFACT_FILENAMES
            }
            for variant in ABLATION_VARIANTS
        },
        "metrics_summary_csv": str(paths.metrics_summary_csv),
        "metrics_summary_md": str(paths.metrics_summary_md),
        "experiment_doc": str(paths.experiment_doc),
        "manifest": str(paths.manifest),
    }


def _publish_staged_outputs(
    staging_paths: ExperimentOutputPaths,
    final_paths: ExperimentOutputPaths,
    *,
    root: Path,
) -> list[Path]:
    publish_pairs = _preflight_publish_outputs(
        staging_paths,
        final_paths,
        root=root,
    )
    published: list[Path] = []
    for source, destination in publish_pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        published.append(destination)
    return published


def _remove_staging_root(staging_root: Path) -> None:
    shutil.rmtree(staging_root, ignore_errors=True)
    try:
        staging_root.parent.rmdir()
    except OSError:
        pass


def _expected_relative_paths() -> list[Path]:
    paths: list[Path] = [
        Path("features") / f"enhanced_samples_{split}.parquet"
        for split in ("all", "train", "valid", "test")
    ]
    paths.extend(
        [
            Path("features") / "feature_groups.json",
            Path("features") / "feature_schema.json",
            Path("features") / "row_alignment_check.json",
            Path("features") / "input_hashes.json",
        ]
    )
    for variant in ABLATION_VARIANTS:
        paths.extend(
            Path("runs") / variant / filename for filename in RUN_ARTIFACT_FILENAMES
        )
    paths.extend(
        [
            Path("metrics_summary.csv"),
            Path("metrics_summary.md"),
            Path("experiment.md"),
            Path("manifest.json"),
        ]
    )
    return paths


def _preflight_publish_outputs(
    staging_paths: ExperimentOutputPaths,
    final_paths: ExperimentOutputPaths,
    *,
    root: Path,
) -> list[tuple[Path, Path]]:
    publish_pairs: list[tuple[Path, Path]] = []
    missing_sources: list[Path] = []
    for relative_path in _expected_relative_paths():
        source = staging_paths.root / relative_path
        destination = final_paths.root / relative_path
        _assert_publish_paths(source, destination, root=root)
        _assert_publish_auxiliary_paths(destination, root=root)
        if not source.exists():
            missing_sources.append(source)
        publish_pairs.append((source, destination))
    if missing_sources:
        examples = ", ".join(str(path) for path in missing_sources[:3])
        raise RuntimeError("趋势图特征消融 staging 产物缺失，取消发布: " f"{examples}")
    return publish_pairs


def _assert_publish_paths(source: Path, destination: Path, *, root: Path) -> None:
    assert_experiment_write_path(source, root=root)
    assert_experiment_write_path(destination, root=root)


def _assert_publish_auxiliary_paths(destination: Path, *, root: Path) -> None:
    assert_experiment_write_path(destination.parent, root=root)
    assert_experiment_write_path(
        destination.with_suffix(destination.suffix + ".tmp"),
        root=root,
    )
    assert_experiment_write_path(
        destination.with_suffix(destination.suffix + ".backup"),
        root=root,
    )


def _write_parquet_guarded(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    root: Path,
) -> None:
    _assert_atomic_write_paths(path, root=root)
    write_parquet_atomic(dataframe, path)


def _write_csv_guarded(dataframe: pd.DataFrame, path: Path, *, root: Path) -> None:
    _assert_atomic_write_paths(path, root=root)
    write_csv_atomic(dataframe, path)


def _write_json_guarded(
    payload: Mapping[str, object],
    path: Path,
    *,
    root: Path,
) -> None:
    _assert_atomic_write_paths(path, root=root)
    write_json_atomic(dict(payload), path)


def _write_text_guarded(text: str, path: Path, *, root: Path) -> None:
    _assert_atomic_write_paths(path, root=root)
    write_text_atomic(text, path)


def _assert_atomic_write_paths(path: Path, *, root: Path) -> None:
    assert_experiment_write_path(path, root=root)
    assert_experiment_write_path(path.with_suffix(path.suffix + ".tmp"), root=root)


def _command_payload(command: str | Sequence[str] | None) -> str | list[str]:
    if command is None:
        return "uv run python src/19_run_trend_graph_feature_ablation.py"
    if isinstance(command, str):
        return command
    return [str(part) for part in command]


def _validate_sample_identifier_columns(
    samples: pd.DataFrame,
    *,
    source_name: str,
) -> None:
    for column in ("attr_id", "attr_type", "attr_value"):
        if samples[column].isna().any():
            raise ValueError(f"{source_name} {column} 存在空值")
        if not samples[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{source_name} {column} 必须保持字符串语义")
