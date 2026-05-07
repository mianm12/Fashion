from __future__ import annotations

import importlib
from pathlib import Path


def test_trend_entrypoint_is_package_facade() -> None:
    trend = importlib.import_module("fashion_trend.trend")

    assert Path(trend.__file__).name == "__init__.py"
    assert hasattr(trend, "build_article_week_sales_frame")
    assert hasattr(trend, "TREND_MODEL_PREDICTION_COLUMNS")
