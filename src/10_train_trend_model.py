from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fashion_trend.foundation import logging as log
from fashion_trend.trend.models.registry import UnknownTrendModelError
from fashion_trend.trend.models.supervised.lightgbm import LIGHTGBM_MODEL_NAME
from fashion_trend.trend.models.supervised.lightgbm_config import (
    resolve_lightgbm_config,
)
from fashion_trend.trend.paths import (
    OUTPUT_METRICS_DIR,
    OUTPUT_MODELS_DIR,
    TREND_MODEL_SAMPLES_TEST_PATH,
    TREND_MODEL_SAMPLES_TRAIN_PATH,
    TREND_MODEL_SAMPLES_VALID_PATH,
)
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_VALUES
from fashion_trend.trend.training import (
    derive_trend_model_output_paths,
    run_artifacts,
    run_trend_model_training,
)

LOG_SOURCE = "trend-model-train"
TREND_MODEL_SAMPLE_SPLIT_PATHS = {
    "train": TREND_MODEL_SAMPLES_TRAIN_PATH,
    "valid": TREND_MODEL_SAMPLES_VALID_PATH,
    "test": TREND_MODEL_SAMPLES_TEST_PATH,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析趋势模型训练入口参数。"""
    parser = argparse.ArgumentParser(description="训练趋势预测模型并写出预测结果。")
    parser.add_argument(
        "--model",
        required=True,
        help="需要训练的趋势模型名称。",
    )
    parser.add_argument("--run-id", help="LightGBM run id。")
    parser.add_argument("--params", type=Path, help="LightGBM 参数 JSON 文件。")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="LightGBM 参数覆盖，格式为 key=value。",
    )
    promotion_group = parser.add_mutually_exclusive_group()
    promotion_group.add_argument(
        "--promote",
        action="store_true",
        help="训练后发布到 stable。",
    )
    promotion_group.add_argument(
        "--no-promote",
        action="store_true",
        help="训练后不发布 stable。",
    )
    promotion_group.add_argument("--promote-run", help="发布一个已评估 LightGBM run。")
    args = parser.parse_args(argv)
    lightgbm_only_used = any(
        [
            args.run_id,
            args.params,
            args.param,
            args.promote,
            args.no_promote,
            args.promote_run,
        ]
    )
    if args.model != LIGHTGBM_MODEL_NAME and lightgbm_only_used:
        parser.error("run、参数和 promotion 选项只支持 --model lightgbm。")
    if args.promote_run and (args.run_id or args.params or args.param):
        parser.error("--promote-run 不能与 --run-id、--params 或 --param 组合。")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """运行趋势模型训练 CLI，记录运行摘要并返回稳定退出码。

    稳定输出位置为 outputs/models/<model>/；argparse 用法错误返回 2，
    领域处理错误返回 1。
    """
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        log.info(f"模型名称参数: {args.model}", source=LOG_SOURCE)
        if args.promote_run:
            metadata = run_artifacts.promote_existing_lightgbm_run(
                args.promote_run,
                model_output_root=OUTPUT_MODELS_DIR,
                metrics_output_root=OUTPUT_METRICS_DIR,
            )
        else:
            for split_name in TREND_MODEL_SPLIT_VALUES:
                log.info(
                    f"输入 {split_name} 样本: "
                    f"{TREND_MODEL_SAMPLE_SPLIT_PATHS[split_name]}",
                    source=LOG_SOURCE,
                )
            if _should_defer_lightgbm_run_output_log(args):
                log.info(
                    "业务阶段: split 样本 -> "
                    "outputs/models/lightgbm/runs/<auto-run-id>/predictions.csv"
                    "（run_id 生成后确定，最终路径以 metadata 为准）",
                    source=LOG_SOURCE,
                )
                log.info(
                    "输出目录: outputs/models/lightgbm/runs/<auto-run-id>"
                    "（run_id 生成后确定，最终路径以 metadata 为准）",
                    source=LOG_SOURCE,
                )
            else:
                output_paths = _derive_training_log_paths(
                    args.model,
                    run_id=args.run_id,
                )
                log.info(
                    "业务阶段: split 样本 -> "
                    f"{_format_path_for_log(output_paths['predictions'])}",
                    source=LOG_SOURCE,
                )
                log.info(
                    f"输出目录: {_format_path_for_log(output_paths['output_dir'])}",
                    source=LOG_SOURCE,
                )
            trainer_options = None
            if args.model == LIGHTGBM_MODEL_NAME and (args.params or args.param):
                trainer_options = {
                    "lightgbm_config": resolve_lightgbm_config(
                        params_path=args.params,
                        cli_params=args.param,
                    )
                }
            promote = None
            if args.promote:
                promote = True
            elif args.no_promote:
                promote = False
            metadata = run_trend_model_training(
                args.model,
                run_id=args.run_id,
                trainer_options=trainer_options,
                promote=promote,
            )
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


def _derive_training_log_paths(
    model_name: str,
    *,
    run_id: str | None,
) -> dict[str, Path]:
    if model_name == LIGHTGBM_MODEL_NAME and run_id:
        return derive_trend_model_output_paths(
            model_name,
            OUTPUT_MODELS_DIR,
            run_id=run_id,
        )
    return derive_trend_model_output_paths(model_name, OUTPUT_MODELS_DIR)


def _should_defer_lightgbm_run_output_log(args: argparse.Namespace) -> bool:
    return (
        args.model == LIGHTGBM_MODEL_NAME
        and args.run_id is None
        and not args.promote
        and (args.params is not None or bool(args.param) or args.no_promote)
    )


def _format_path_for_log(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
