from __future__ import annotations

from pathlib import Path

from fashion_trend.trend.evaluation.payloads import (
    build_trend_metrics_payload,
    derive_trend_metric_output_paths,
    read_trend_model_predictions,
    write_trend_metrics,
)
from fashion_trend.trend.paths import OUTPUT_METRICS_DIR, OUTPUT_MODELS_DIR


def run_trend_model_evaluation(
    model_name: str,
    model_output_root: Path = OUTPUT_MODELS_DIR,
    metrics_output_root: Path = OUTPUT_METRICS_DIR,
) -> dict[str, object]:
    """运行单个趋势模型的评价，并写出 trend_metrics.json。

    runner 先根据模型名推导 `predictions.csv` 输入路径和 `trend_metrics.json`
    输出路径，再读取标准预测表、构建评价载荷并写出指标文件。评价指标只聚合
    valid/test 切分；写出前由 payload 层执行严格 JSON 校验。
    """

    output_paths = derive_trend_metric_output_paths(
        model_name,
        model_output_root=model_output_root,
        metrics_output_root=metrics_output_root,
    )
    predictions = read_trend_model_predictions(output_paths["predictions"])
    payload = build_trend_metrics_payload(
        predictions,
        model_name=model_name,
        prediction_path=output_paths["predictions"],
        output_path=output_paths["metrics"],
    )
    write_trend_metrics(payload, output_paths["metrics"])
    return payload
