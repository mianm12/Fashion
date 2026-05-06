from __future__ import annotations

from fashion_trend.models.base import TrendModelTrainer
from fashion_trend.models.last_week import LAST_WEEK_MODEL_NAME, LastWeekTrainer


class UnknownTrendModelError(ValueError):
    """Raised when a requested trend model is not registered."""


TREND_MODEL_REGISTRY: dict[str, TrendModelTrainer] = {
    LAST_WEEK_MODEL_NAME: LastWeekTrainer(),
}


def list_trend_model_names() -> tuple[str, ...]:
    return tuple(sorted(TREND_MODEL_REGISTRY))


def get_trend_model_trainer(model_name: str) -> TrendModelTrainer:
    try:
        return TREND_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(list_trend_model_names())
        raise UnknownTrendModelError(
            f"不支持的趋势模型: {model_name}。可用模型: {available}"
        ) from exc
