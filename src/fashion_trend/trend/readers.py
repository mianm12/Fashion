from __future__ import annotations

import json
from pathlib import Path

from fashion_trend.trend.evaluation import read_trend_model_predictions
from fashion_trend.trend.heat.article_sales import read_article_week_sales
from fashion_trend.trend.heat.attribute_heat import read_attribute_week_heat
from fashion_trend.trend.labels.targets import read_attribute_week_target
from fashion_trend.trend.schema import TREND_METRICS_PAYLOAD_REQUIRED_KEYS
from fashion_trend.trend.splits import read_trend_model_split


def read_trend_metrics(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"趋势评价指标文件不存在: {metrics_path}")
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取趋势评价指标文件: {metrics_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"趋势评价指标文件必须是 JSON object: {metrics_path}")
    missing_keys = sorted(set(TREND_METRICS_PAYLOAD_REQUIRED_KEYS) - set(payload))
    if missing_keys:
        raise ValueError(
            "趋势评价指标文件缺少必要字段: "
            + ", ".join(missing_keys)
            + f"。文件: {metrics_path}"
        )
    return payload
