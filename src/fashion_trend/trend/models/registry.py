from __future__ import annotations

from fashion_trend.trend.models.base import TrendModelTrainer
from fashion_trend.trend.models.baselines.last_week import (
    LAST_WEEK_MODEL_NAME,
    LastWeekTrainer,
)
from fashion_trend.trend.models.baselines.moving_average import (
    MOVING_AVERAGE_MODEL_NAME,
    MovingAverageTrainer,
)
from fashion_trend.trend.models.baselines.previous_growth import (
    PREVIOUS_GROWTH_MODEL_NAME,
    PreviousGrowthTrainer,
)
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_MODEL_NAME,
    LightGBMTrendTrainer,
)


class UnknownTrendModelError(ValueError):
    """请求的趋势模型未注册时抛出的错误。"""


TREND_MODEL_REGISTRY: dict[str, TrendModelTrainer] = {
    LAST_WEEK_MODEL_NAME: LastWeekTrainer(),
    LIGHTGBM_MODEL_NAME: LightGBMTrendTrainer(),
    MOVING_AVERAGE_MODEL_NAME: MovingAverageTrainer(),
    PREVIOUS_GROWTH_MODEL_NAME: PreviousGrowthTrainer(),
}


def list_trend_model_names() -> tuple[str, ...]:
    """返回当前注册表中可用的趋势模型名。"""

    return tuple(sorted(TREND_MODEL_REGISTRY))


def get_trend_model_trainer(model_name: str) -> TrendModelTrainer:
    """按模型名取得趋势模型训练器。"""

    try:
        return TREND_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(list_trend_model_names())
        raise UnknownTrendModelError(
            f"不支持的趋势模型: {model_name}。可用模型: {available}"
        ) from exc
