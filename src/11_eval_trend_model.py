from __future__ import annotations

import argparse
from typing import Sequence

from fashion_trend.foundation import logging as log
from fashion_trend.trend.evaluation import run_trend_model_evaluation

LOG_SOURCE = "trend-model-eval"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析趋势模型评价入口参数。"""
    parser = argparse.ArgumentParser(description="评价趋势预测模型并写出指标。")
    parser.add_argument(
        "--model",
        required=True,
        help="需要评价的趋势模型名称。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """运行趋势模型评价 CLI，并返回稳定退出码。"""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        metrics = run_trend_model_evaluation(args.model)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"模型名称: {metrics['model_name']}", source=LOG_SOURCE)
    log.info(
        "评价 split: " + ", ".join(str(split) for split in metrics["evaluated_splits"]),
        source=LOG_SOURCE,
    )
    for split_name in metrics["evaluated_splits"]:
        split_metrics = metrics["overall"][split_name]
        split_groups = metrics["groups"][split_name]
        log.info(
            f"{split_name} 评价: "
            f"groups={split_groups['ranking_groups']:,}, "
            f"mae={_format_metric(split_metrics['mae'])}, "
            f"rmse={_format_metric(split_metrics['rmse'])}, "
            f"spearman={_format_metric(split_metrics['spearman'])}, "
            f"precision@10={_format_metric(split_metrics['precision_at_k']['10'])}, "
            f"recall@10={_format_metric(split_metrics['recall_at_k']['10'])}, "
            f"ndcg@10={_format_metric(split_metrics['ndcg_at_k']['10'])}",
            source=LOG_SOURCE,
        )
    log.info(f"评价输出文件: {metrics['output_path']}", source=LOG_SOURCE)
    return 0


def _format_metric(value: object) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
