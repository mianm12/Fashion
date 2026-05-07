from __future__ import annotations

import argparse
from typing import Sequence

from fashion_trend.foundation import logging as log
from fashion_trend.models.registry import UnknownTrendModelError
from fashion_trend.training import run_trend_model_training
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_VALUES

LOG_SOURCE = "trend-model-train"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析趋势模型训练入口参数。"""
    parser = argparse.ArgumentParser(description="训练趋势预测模型并写出预测结果。")
    parser.add_argument(
        "--model",
        required=True,
        help="需要训练的趋势模型名称。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """运行趋势模型训练 CLI，保留 argparse 用法错误码并记录运行摘要。"""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        metadata = run_trend_model_training(args.model)
    except UnknownTrendModelError as exc:
        log.error(str(exc), source=LOG_SOURCE)
        return 1
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"模型名称: {metadata['model_name']}", source=LOG_SOURCE)
    log.info(f"模型类型: {metadata['model_type']}", source=LOG_SOURCE)
    log.info(f"预测行数: {metadata['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖样本周数: {metadata['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {metadata['attributes']:,}", source=LOG_SOURCE)
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_stats = metadata["splits"][split_name]
        log.info(
            f"{split_name} 预测: rows={split_stats['rows']:,}, "
            f"weeks={split_stats['weeks']:,}, "
            f"attributes={split_stats['attributes']:,}, "
            f"week_range={split_stats['week_min']}..{split_stats['week_max']}",
            source=LOG_SOURCE,
        )
    log.info(f"输出目录: {metadata['output_dir']}", source=LOG_SOURCE)
    log.info(f"预测输出文件: {metadata['prediction_path']}", source=LOG_SOURCE)
    log.info(f"参数输出文件: {metadata['params_path']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
