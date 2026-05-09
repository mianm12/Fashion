from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

import fashion_trend.trend.training.run_artifacts as run_artifacts
from fashion_trend.trend.models.base import TrendTrainContext
from fashion_trend.trend.models.registry import get_trend_model_trainer
from fashion_trend.trend.paths import (
    OUTPUT_MODELS_DIR,
    TREND_MODEL_SAMPLES_TEST_PATH,
    TREND_MODEL_SAMPLES_TRAIN_PATH,
    TREND_MODEL_SAMPLES_VALID_PATH,
)
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_VALUES
from fashion_trend.trend.splits import read_trend_model_split
from fashion_trend.trend.training.outputs import (
    build_trend_train_metadata,
    derive_trend_model_output_paths,
    validate_trend_train_result,
    write_trend_model_outputs,
)


def default_trend_model_input_paths() -> dict[str, Path]:
    """返回趋势模型 train/valid/test 的默认输入样本路径。"""

    return {
        "train": TREND_MODEL_SAMPLES_TRAIN_PATH,
        "valid": TREND_MODEL_SAMPLES_VALID_PATH,
        "test": TREND_MODEL_SAMPLES_TEST_PATH,
    }


def read_trend_model_split_frames(
    input_paths: Mapping[str, Path],
) -> dict[str, pd.DataFrame]:
    """按标准 split 顺序读取趋势模型训练、验证和测试样本。"""

    missing_splits = set(TREND_MODEL_SPLIT_VALUES) - set(input_paths)
    if missing_splits:
        raise ValueError(f"趋势模型输入路径缺少 split: {sorted(missing_splits)}")
    return {
        split_name: read_trend_model_split(input_paths[split_name])
        for split_name in TREND_MODEL_SPLIT_VALUES
    }


def run_trend_model_training(
    model_name: str,
    input_paths: Mapping[str, Path] | None = None,
    output_root: Path = OUTPUT_MODELS_DIR,
    *,
    run_id: str | None = None,
    trainer_options: Mapping[str, object] | None = None,
    promote: bool | None = None,
) -> dict[str, object]:
    """运行单个趋势模型的通用训练流程。

    流程依次解析默认或调用方传入的 split 输入路径，读取 train/valid/test
    样本，通过模型注册表查找训练器，并从模型名派生输出目录和标准文件路径。
    runner 随后构建 `TrendTrainContext` 调用训练器，校验返回的
    `TrendTrainResult`，生成 metadata 载荷，最后统一写出 predictions、params、
    metadata 和训练器附加产物。
    """

    trainer = get_trend_model_trainer(model_name)
    is_lightgbm = model_name == "lightgbm"
    if not is_lightgbm and (
        run_id is not None or trainer_options or promote is not None
    ):
        raise ValueError("run、参数和 promotion 选项只支持 --model lightgbm。")
    if input_paths is None:
        resolved_input_paths = default_trend_model_input_paths()
    else:
        resolved_input_paths = dict(input_paths)
    if is_lightgbm:
        return _run_lightgbm_training(
            trainer=trainer,
            model_name=model_name,
            input_paths=resolved_input_paths,
            output_root=output_root,
            run_id=run_id,
            trainer_options=trainer_options or {},
            promote=promote,
        )
    return _run_standard_trend_model_training(
        trainer=trainer,
        model_name=model_name,
        input_paths=resolved_input_paths,
        output_root=output_root,
    )


def _run_standard_trend_model_training(
    *,
    trainer,
    model_name: str,
    input_paths: Mapping[str, Path],
    output_root: Path,
) -> dict[str, object]:
    split_frames = read_trend_model_split_frames(input_paths)
    output_paths = derive_trend_model_output_paths(model_name, output_root)
    context = TrendTrainContext(
        model_name=model_name,
        split_frames=split_frames,
        input_paths=input_paths,
        output_dir=output_paths["output_dir"],
    )
    result = trainer.train(context)
    validate_trend_train_result(result, context)
    metadata = build_trend_train_metadata(result, context, output_paths)
    write_trend_model_outputs(result, metadata, output_paths)
    return metadata


def _run_lightgbm_training(
    *,
    trainer,
    model_name: str,
    input_paths: Mapping[str, Path],
    output_root: Path,
    run_id: str | None,
    trainer_options: Mapping[str, object],
    promote: bool | None,
) -> dict[str, object]:
    split_frames = read_trend_model_split_frames(input_paths)
    stable_paths = derive_trend_model_output_paths(model_name, output_root)
    run_root = stable_paths["output_dir"] / "runs"
    explicit_run_id = run_id is not None
    resolved_run_id = run_id or run_artifacts.generate_lightgbm_run_id(run_root)
    run_paths = derive_trend_model_output_paths(
        model_name,
        output_root,
        run_id=resolved_run_id,
    )
    if run_paths["output_dir"].exists():
        raise FileExistsError(f"LightGBM run_id 已存在: {resolved_run_id}")
    promotion_requested = _resolve_lightgbm_promotion_default(
        explicit_run_id=explicit_run_id,
        trainer_options=trainer_options,
        promote=promote,
    )
    context = TrendTrainContext(
        model_name=model_name,
        split_frames=split_frames,
        input_paths=input_paths,
        output_dir=run_paths["output_dir"],
        trainer_options=trainer_options,
    )
    result = trainer.train(context)
    validate_trend_train_result(result, context)
    metadata = build_trend_train_metadata(
        result,
        context,
        run_paths,
        run_id=resolved_run_id,
        run_dir=run_paths["output_dir"],
        stable_output_dir=run_paths["stable_output_dir"],
        promotion_requested=promotion_requested,
    )
    write_trend_model_outputs(result, metadata, run_paths)
    run_artifacts.upsert_lightgbm_run_index(
        run_paths["index"],
        run_artifacts.build_lightgbm_run_summary(
            run_id=resolved_run_id,
            metadata=metadata,
            promotion_status="not_requested",
        ),
    )
    if promotion_requested:
        try:
            run_artifacts.publish_lightgbm_run_to_stable(
                result=result,
                run_metadata=metadata,
                run_context=context,
                stable_paths=stable_paths,
            )
        except Exception as exc:
            run_artifacts.record_lightgbm_promotion_failure(
                index_path=run_paths["index"],
                summary=run_artifacts.build_lightgbm_run_summary(
                    run_id=resolved_run_id,
                    metadata=metadata,
                    promotion_status="failed",
                    promotion_error=str(exc),
                ),
                run_dir=run_paths["output_dir"],
                stable_dir=stable_paths["output_dir"],
                promotion_error=exc,
            )
            raise

        try:
            run_artifacts.upsert_lightgbm_run_index(
                run_paths["index"],
                run_artifacts.build_lightgbm_run_summary(
                    run_id=resolved_run_id,
                    metadata=metadata,
                    promotion_status="succeeded",
                ),
            )
        except Exception as exc:
            run_artifacts.record_lightgbm_index_update_failure(
                run_dir=run_paths["output_dir"],
                stable_dir=stable_paths["output_dir"],
                attempted_status="succeeded",
                index_error=exc,
            )
            raise
    return metadata


def _resolve_lightgbm_promotion_default(
    *,
    explicit_run_id: bool,
    trainer_options: Mapping[str, object],
    promote: bool | None,
) -> bool:
    if promote is not None:
        return bool(promote)
    has_custom_config = bool(trainer_options)
    return not explicit_run_id and not has_custom_config
