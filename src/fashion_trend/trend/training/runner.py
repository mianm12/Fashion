from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

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
) -> dict[str, object]:
    """运行单个趋势模型的通用训练流程。

    流程依次解析默认或调用方传入的 split 输入路径，读取 train/valid/test
    样本，通过模型注册表查找训练器，并从模型名派生输出目录和标准文件路径。
    runner 随后构建 `TrendTrainContext` 调用训练器，校验返回的
    `TrendTrainResult`，生成 metadata 载荷，最后统一写出 predictions、params、
    metadata 和训练器附加产物。
    """

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
