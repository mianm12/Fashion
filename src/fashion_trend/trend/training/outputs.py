from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Mapping

import pandas as pd

from fashion_trend.foundation.artifacts import (
    validate_output_parent_dirs,
    validate_safe_path_segment,
)
from fashion_trend.foundation.io import (
    remove_file_if_exists,
    write_binary_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from fashion_trend.trend.models.base import (
    KNOWN_MODEL_TYPES,
    TrendArtifact,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.predictions import validate_trend_model_predictions


def derive_trend_model_output_paths(
    model_name: str,
    output_root: Path = OUTPUT_MODELS_DIR,
) -> dict[str, Path]:
    validate_safe_path_segment(model_name, "model_name")
    output_dir = output_root / model_name
    validate_output_parent_dirs(output_dir, output_root)
    return {
        "output_dir": output_dir,
        "predictions": output_dir / "predictions.csv",
        "params": output_dir / "params.json",
        "metadata": output_dir / "metadata.json",
    }


def validate_trend_train_result(
    result: TrendTrainResult,
    context: TrendTrainContext,
) -> None:
    """校验训练器返回的 TrendTrainResult 能进入标准输出流程。

    校验范围包括请求模型名、已知模型类型、参数载荷类型、预测表契约、整数
    week_id、预测表中的 model_name 一致性，以及附加产物路径安全。
    """

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
    """构建训练输出 metadata.json 的内存载荷。

    metadata 记录输入路径、标准输出路径、整体行数和属性数，以及每个 split 的
    行数、周数、属性数和周编号范围。训练器只能追加非核心字段，不能覆盖 runner
    负责维护的核心 metadata 键。
    """

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
    """写出趋势模型标准产物和附加产物。

    写入前先校验 params、metadata 和 artifact payload 是否可写，并构造包含
    predictions、params、artifact 与 metadata 的输出项。实际发布采用暂存目录
    加备份替换的两阶段流程：全部载荷先写到暂存目录，再逐个替换目标文件；若
    替换过程中失败，则恢复已替换文件并清理暂存目录。
    """

    _validate_output_payloads(result, metadata)
    output_dir = output_paths["output_dir"]
    output_items = _build_output_items(result, metadata, output_paths)
    _validate_output_destinations(output_items, output_dir)

    staging_dir = output_dir / f".tmp-trend-model-{uuid.uuid4().hex}"
    published_paths: list[tuple[Path, Path | None]] = []
    try:
        # 先写入暂存目录，避免部分载荷写入失败时污染已有目标产物。
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
        # 替换阶段失败时，按已发布路径倒序恢复调用方可见的旧产物。
        _rollback_published_outputs(published_paths)
        raise
    else:
        _remove_backup_outputs(published_paths)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _validate_artifacts(artifacts: tuple[TrendArtifact, ...]) -> None:
    """校验附加产物使用安全的输出目录相对路径。"""

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
    """校验 week_id 可无损转换为整数周编号。"""

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
    """校验输出载荷的 JSON 可序列化能力和 artifact payload 类型。"""

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
    """按当前训练产物 JSON 写出选项校验载荷可序列化。"""

    try:
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"趋势模型 {source_name} 不能序列化为 JSON。") from exc


def _build_output_items(
    result: TrendTrainResult,
    metadata: dict[str, object],
    output_paths: Mapping[str, Path],
) -> list[tuple[Path, pd.DataFrame | dict[str, object] | bytes]]:
    """构造待写入暂存目录并最终发布到目标路径的输出项。"""

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
    """校验目标路径位于输出目录内，且没有重复路径或目录冲突。"""

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
        validate_output_parent_dirs(final_path.parent, output_dir)


def _write_output_payload(
    payload: pd.DataFrame | dict[str, object] | bytes,
    output_path: Path,
) -> None:
    """按载荷类型把单个输出项写入暂存路径。"""

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
    """发布失败时移除新产物，并用备份恢复已替换的旧产物。"""

    for final_path, backup_path in reversed(published_paths):
        if backup_path is None:
            remove_file_if_exists(final_path)
            continue
        remove_file_if_exists(final_path)
        backup_path.replace(final_path)


def _remove_backup_outputs(
    published_paths: list[tuple[Path, Path | None]],
) -> None:
    """发布成功后清理目标文件替换过程中留下的备份产物。"""

    for _final_path, backup_path in published_paths:
        if backup_path is not None:
            remove_file_if_exists(backup_path)
