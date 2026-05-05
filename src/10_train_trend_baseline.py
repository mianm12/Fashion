from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from fashion_trend import log
from fashion_trend.config import PATH
from fashion_trend.models.baseline_last_week import (
    LAST_WEEK_MODEL_NAME,
    LAST_WEEK_PARAMS,
    predict_last_week,
)
from fashion_trend.trend import (
    TREND_MODEL_SPLIT_VALUES,
    read_trend_model_split,
    validate_trend_baseline_predictions,
    write_json,
    write_trend_csv,
)

LOG_SOURCE = "trend-baseline-train"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练趋势 baseline 模型并写出预测结果。"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[LAST_WEEK_MODEL_NAME],
        help="需要训练的 baseline 模型名称。",
    )
    return parser.parse_args(argv)


def build_prediction_metadata(predictions: pd.DataFrame) -> dict[str, object]:
    split_metadata: dict[str, dict[str, object]] = {}
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_predictions = predictions[predictions["split"] == split_name]
        split_metadata[split_name] = {
            "rows": int(len(split_predictions)),
            "weeks": int(split_predictions["week_id"].nunique()),
            "attributes": int(split_predictions["attr_id"].nunique()),
            "week_min": int(split_predictions["week_id"].min()),
            "week_max": int(split_predictions["week_id"].max()),
        }

    return {
        "model_name": LAST_WEEK_MODEL_NAME,
        "input_paths": {
            "train": str(PATH["features_trend_model_samples_train"]),
            "valid": str(PATH["features_trend_model_samples_valid"]),
            "test": str(PATH["features_trend_model_samples_test"]),
        },
        "prediction_path": str(PATH["output_model_last_week_predictions"]),
        "total_rows": int(len(predictions)),
        "total_weeks": int(predictions["week_id"].nunique()),
        "total_attributes": int(predictions["attr_id"].nunique()),
        "splits": split_metadata,
    }


def train_trend_baseline(model_name: str) -> dict[str, object]:
    if model_name != LAST_WEEK_MODEL_NAME:
        raise ValueError(f"不支持的趋势 baseline 模型: {model_name}")

    input_paths = {
        "train": PATH["features_trend_model_samples_train"],
        "valid": PATH["features_trend_model_samples_valid"],
        "test": PATH["features_trend_model_samples_test"],
    }
    split_frames = []
    for split_name in TREND_MODEL_SPLIT_VALUES:
        input_path = input_paths[split_name]
        log.info(f"输入 {split_name} 样本 split: {input_path}", source=LOG_SOURCE)
        split_frames.append(read_trend_model_split(input_path))

    split_samples = pd.concat(split_frames, ignore_index=True)
    predictions = predict_last_week(split_samples)
    validate_trend_baseline_predictions(predictions, split_samples)

    write_trend_csv(predictions, PATH["output_model_last_week_predictions"])
    write_json(dict(LAST_WEEK_PARAMS), PATH["output_model_last_week_params"])
    metadata = build_prediction_metadata(predictions)
    write_json(metadata, PATH["output_model_last_week_metadata"])
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1

    try:
        metadata = train_trend_baseline(args.model)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"模型名称: {metadata['model_name']}", source=LOG_SOURCE)
    log.info(f"预测行数: {metadata['total_rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖样本周数: {metadata['total_weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {metadata['total_attributes']:,}", source=LOG_SOURCE)
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_stats = metadata["splits"][split_name]
        log.info(
            f"{split_name} 预测: rows={split_stats['rows']:,}, "
            f"weeks={split_stats['weeks']:,}, "
            f"attributes={split_stats['attributes']:,}, "
            f"week_range={split_stats['week_min']}..{split_stats['week_max']}",
            source=LOG_SOURCE,
        )
    log.info(
        f"预测输出文件: {PATH['output_model_last_week_predictions']}",
        source=LOG_SOURCE,
    )
    log.info(
        f"参数输出文件: {PATH['output_model_last_week_params']}",
        source=LOG_SOURCE,
    )
    log.info(
        f"元数据输出文件: {PATH['output_model_last_week_metadata']}",
        source=LOG_SOURCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
