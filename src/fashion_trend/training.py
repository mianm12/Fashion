from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Mapping

import pandas as pd

from fashion_trend.foundation.io import (
    remove_file_if_exists,
    write_binary_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from fashion_trend.foundation.paths import OUTPUT_MODELS_DIR, PATH
from fashion_trend.models.base import (
    KNOWN_MODEL_TYPES,
    TrendArtifact,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.models.registry import get_trend_model_trainer
from fashion_trend.trend.predictions import validate_trend_model_predictions
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_VALUES
from fashion_trend.trend.splits import read_trend_model_split


def default_trend_model_input_paths() -> dict[str, Path]:
    return {
        "train": PATH["features_trend_model_samples_train"],
        "valid": PATH["features_trend_model_samples_valid"],
        "test": PATH["features_trend_model_samples_test"],
    }


def derive_trend_model_output_paths(
    model_name: str,
    output_root: Path = OUTPUT_MODELS_DIR,
) -> dict[str, Path]:
    output_dir = output_root / model_name
    return {
        "output_dir": output_dir,
        "predictions": output_dir / "predictions.csv",
        "params": output_dir / "params.json",
        "metadata": output_dir / "metadata.json",
    }


def read_trend_model_split_frames(
    input_paths: Mapping[str, Path],
) -> dict[str, pd.DataFrame]:
    missing_splits = set(TREND_MODEL_SPLIT_VALUES) - set(input_paths)
    if missing_splits:
        raise ValueError(f"趋势模型输入路径缺少 split: {sorted(missing_splits)}")
    return {
        split_name: read_trend_model_split(input_paths[split_name])
        for split_name in TREND_MODEL_SPLIT_VALUES
    }


def validate_trend_train_result(
    result: TrendTrainResult,
    context: TrendTrainContext,
) -> None:
    if result.model_name != context.model_name:
        raise ValueError(
            "趋势模型训练结果 model_name 与请求不一致: "
            f"result={result.model_name}, context={context.model_name}"
        )
    if result.model_type not in KNOWN_MODEL_TYPES:
        raise ValueError(f"趋势模型训练结果存在未知 model_type: {result.model_type}")
    if not isinstance(result.params, dict):
        raise ValueError("趋势模型训练结果 params 必须是字典。")

    split_samples = pd.concat(
        [context.split_frames[split_name] for split_name in context.split_order],
        ignore_index=True,
    )
    validate_trend_model_predictions(result.predictions, split_samples)
    _validate_integer_week_ids(result.predictions["week_id"], "趋势模型预测表")
    prediction_model_names = set(result.predictions["model_name"])
    if prediction_model_names != {result.model_name}:
        raise ValueError(
            "趋势模型训练结果预测 model_name 与训练结果不一致: "
            f"{sorted(prediction_model_names)}"
        )
    _validate_artifacts(result.artifacts)


def build_trend_train_metadata(
    result: TrendTrainResult,
    context: TrendTrainContext,
    output_paths: Mapping[str, Path],
) -> dict[str, object]:
    split_metadata: dict[str, dict[str, object]] = {}
    for split_name in context.split_order:
        split_predictions = result.predictions[
            result.predictions["split"] == split_name
        ]
        if split_predictions.empty:
            raise ValueError(f"趋势模型 metadata 缺少 {split_name} split。")
        split_week_ids = _validate_integer_week_ids(
            split_predictions["week_id"],
            f"趋势模型 metadata {split_name} split",
        )
        split_metadata[split_name] = {
            "rows": int(len(split_predictions)),
            "weeks": int(split_week_ids.nunique()),
            "attributes": int(split_predictions["attr_id"].nunique()),
            "week_min": int(split_week_ids.min()),
            "week_max": int(split_week_ids.max()),
        }

    week_ids = _validate_integer_week_ids(
        result.predictions["week_id"], "趋势模型 metadata"
    )
    core_metadata: dict[str, object] = {
        "model_name": result.model_name,
        "model_type": result.model_type,
        "input_paths": {
            split_name: str(context.input_paths[split_name])
            for split_name in context.split_order
        },
        "output_dir": str(output_paths["output_dir"]),
        "prediction_path": str(output_paths["predictions"]),
        "params_path": str(output_paths["params"]),
        "rows": int(len(result.predictions)),
        "weeks": int(week_ids.nunique()),
        "attributes": int(result.predictions["attr_id"].nunique()),
        "splits": split_metadata,
        "extra_artifacts": [
            {"path": artifact.relative_path, "kind": artifact.kind}
            for artifact in result.artifacts
        ],
    }

    overlapping_keys = sorted(set(core_metadata) & set(result.metadata))
    if overlapping_keys:
        raise ValueError(
            "趋势模型 metadata 不能覆盖 runner 核心字段: " + ", ".join(overlapping_keys)
        )
    return {**core_metadata, **result.metadata}


def write_trend_model_outputs(
    result: TrendTrainResult,
    metadata: dict[str, object],
    output_paths: Mapping[str, Path],
) -> None:
    _validate_output_payloads(result, metadata)
    output_dir = output_paths["output_dir"]
    output_items = _build_output_items(result, metadata, output_paths)
    _validate_output_destinations(output_items, output_dir)

    staging_dir = output_dir / f".tmp-trend-model-{uuid.uuid4().hex}"
    published_paths: list[tuple[Path, Path | None]] = []
    try:
        for final_path, payload in output_items:
            staging_path = staging_dir / final_path.relative_to(output_dir)
            _write_output_payload(payload, staging_path)

        for final_path, _payload in output_items:
            staging_path = staging_dir / final_path.relative_to(output_dir)
            backup_path = None
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                backup_path = final_path.with_name(
                    f".{final_path.name}.bak-{uuid.uuid4().hex}"
                )
                final_path.replace(backup_path)
            published_paths.append((final_path, backup_path))
            staging_path.replace(final_path)
    except Exception:
        _rollback_published_outputs(published_paths)
        raise
    else:
        _remove_backup_outputs(published_paths)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def run_trend_model_training(
    model_name: str,
    input_paths: Mapping[str, Path] | None = None,
    output_root: Path = OUTPUT_MODELS_DIR,
) -> dict[str, object]:
    trainer = get_trend_model_trainer(model_name)
    if input_paths is None:
        resolved_input_paths = default_trend_model_input_paths()
    else:
        resolved_input_paths = dict(input_paths)
    split_frames = read_trend_model_split_frames(resolved_input_paths)
    output_paths = derive_trend_model_output_paths(model_name, output_root)
    context = TrendTrainContext(
        model_name=model_name,
        split_frames=split_frames,
        input_paths=resolved_input_paths,
        output_dir=output_paths["output_dir"],
    )
    result = trainer.train(context)
    validate_trend_train_result(result, context)
    metadata = build_trend_train_metadata(result, context, output_paths)
    write_trend_model_outputs(result, metadata, output_paths)
    return metadata


def _validate_artifacts(artifacts: tuple[TrendArtifact, ...]) -> None:
    for artifact in artifacts:
        raw_path_parts = artifact.relative_path.split("/")
        artifact_path = Path(artifact.relative_path)
        if (
            not artifact.relative_path
            or artifact_path.is_absolute()
            or "." in raw_path_parts
            or ".." in raw_path_parts
        ):
            raise ValueError(f"趋势模型 artifact 路径不安全: {artifact.relative_path}")


def _validate_integer_week_ids(week_ids: pd.Series, source_name: str) -> pd.Series:
    try:
        numeric_week_ids = pd.to_numeric(week_ids, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} week_id 必须为整数。") from exc
    if numeric_week_ids.isna().any() or not (numeric_week_ids % 1 == 0).all():
        raise ValueError(f"{source_name} week_id 必须为整数。")
    return numeric_week_ids.astype("int64")


def _validate_output_payloads(
    result: TrendTrainResult,
    metadata: dict[str, object],
) -> None:
    _validate_json_payload(result.params, "params")
    _validate_json_payload(metadata, "metadata")
    _validate_artifacts(result.artifacts)
    for artifact in result.artifacts:
        if isinstance(artifact.payload, dict):
            _validate_json_payload(
                artifact.payload, f"artifact: {artifact.relative_path}"
            )
            continue
        if isinstance(artifact.payload, (pd.DataFrame, bytes)):
            continue
        raise ValueError("不支持的趋势模型 artifact payload: " + artifact.relative_path)


def _validate_json_payload(payload: dict[str, object], source_name: str) -> None:
    try:
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"趋势模型 {source_name} 不能序列化为 JSON。") from exc


def _build_output_items(
    result: TrendTrainResult,
    metadata: dict[str, object],
    output_paths: Mapping[str, Path],
) -> list[tuple[Path, pd.DataFrame | dict[str, object] | bytes]]:
    output_items: list[tuple[Path, pd.DataFrame | dict[str, object] | bytes]] = [
        (output_paths["predictions"], result.predictions),
        (output_paths["params"], result.params),
    ]
    output_items.extend(
        (output_paths["output_dir"] / artifact.relative_path, artifact.payload)
        for artifact in result.artifacts
    )
    output_items.append((output_paths["metadata"], metadata))
    return output_items


def _validate_output_destinations(
    output_items: list[tuple[Path, pd.DataFrame | dict[str, object] | bytes]],
    output_dir: Path,
) -> None:
    seen_paths: set[Path] = set()
    for final_path, _payload in output_items:
        try:
            final_path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError(f"趋势模型输出路径不在输出目录内: {final_path}") from exc
        if final_path in seen_paths:
            raise ValueError(f"趋势模型输出路径重复: {final_path}")
        seen_paths.add(final_path)
        if final_path.exists() and final_path.is_dir():
            raise IsADirectoryError(f"趋势模型输出路径是目录: {final_path}")
        _validate_output_parent_dirs(final_path.parent, output_dir)


def _validate_output_parent_dirs(parent_path: Path, output_dir: Path) -> None:
    try:
        relative_parent = parent_path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(f"趋势模型输出父目录不在输出目录内: {parent_path}") from exc
    current_path = output_dir
    if current_path.exists() and not current_path.is_dir():
        raise NotADirectoryError(f"趋势模型输出目录不是目录: {current_path}")
    for path_part in relative_parent.parts:
        current_path = current_path / path_part
        if current_path.exists() and not current_path.is_dir():
            raise NotADirectoryError(f"趋势模型输出父路径不是目录: {current_path}")


def _write_output_payload(
    payload: pd.DataFrame | dict[str, object] | bytes,
    output_path: Path,
) -> None:
    if isinstance(payload, pd.DataFrame):
        write_csv_atomic(payload, output_path)
        return
    if isinstance(payload, dict):
        write_json_atomic(payload, output_path)
        return
    if isinstance(payload, bytes):
        write_binary_atomic(payload, output_path)
        return
    raise ValueError("不支持的趋势模型输出 payload。")


def _rollback_published_outputs(
    published_paths: list[tuple[Path, Path | None]],
) -> None:
    for final_path, backup_path in reversed(published_paths):
        if backup_path is None:
            remove_file_if_exists(final_path)
            continue
        remove_file_if_exists(final_path)
        backup_path.replace(final_path)


def _remove_backup_outputs(
    published_paths: list[tuple[Path, Path | None]],
) -> None:
    for _final_path, backup_path in published_paths:
        if backup_path is not None:
            remove_file_if_exists(backup_path)
