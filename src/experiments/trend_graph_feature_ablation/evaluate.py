from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.trend_graph_feature_ablation.artifact_io import (
    assert_experiment_write_path,
)
from experiments.trend_graph_feature_ablation.contracts import ABLATION_VARIANTS
from experiments.trend_graph_feature_ablation.paths import EXPERIMENT_ROOT
from fashion_trend.trend.evaluation.payloads import (
    build_trend_metrics_payload,
    write_trend_metrics,
)
from fashion_trend.trend.models.supervised.lightgbm import LIGHTGBM_MODEL_NAME


def evaluate_variant_predictions(
    variant: str,
    predictions: pd.DataFrame,
    *,
    prediction_path: Path,
    output_path: Path,
    experiment_root: Path | None = None,
) -> dict[str, object]:
    """评价单个图特征消融 variant 的预测，并写出 run-scoped metrics.json。"""

    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"未知趋势图特征消融 variant: {variant}")

    root = experiment_root or EXPERIMENT_ROOT
    metrics_path = assert_experiment_write_path(Path(output_path), root=root)
    tmp_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    assert_experiment_write_path(tmp_path, root=root)

    payload = build_trend_metrics_payload(
        predictions,
        LIGHTGBM_MODEL_NAME,
        Path(prediction_path),
        metrics_path,
        run_id=variant,
    )
    write_trend_metrics(payload, metrics_path)
    return payload
