from __future__ import annotations

from typing import Any

from fashion_trend.recommendation.contracts import (
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
)

_ENHANCED_WEIGHT_GRID = (
    (14, 30, 10, 8, 16, 8, 4, 4, 3, 3),
    (16, 28, 10, 8, 16, 8, 4, 4, 3, 3),
    (12, 32, 10, 8, 16, 8, 4, 4, 3, 3),
    (14, 26, 12, 10, 16, 8, 4, 4, 3, 3),
    (14, 28, 8, 12, 16, 8, 4, 4, 3, 3),
    (14, 26, 10, 8, 20, 8, 4, 4, 3, 3),
    (14, 28, 10, 8, 16, 10, 4, 4, 3, 3),
    (14, 28, 10, 8, 16, 8, 6, 4, 3, 3),
    (14, 28, 10, 8, 16, 8, 4, 6, 3, 3),
    (14, 28, 10, 8, 16, 8, 4, 4, 4, 4),
    (18, 26, 10, 8, 14, 8, 4, 4, 4, 4),
    (12, 34, 8, 8, 14, 8, 4, 4, 4, 4),
    (12, 28, 12, 12, 14, 8, 4, 4, 3, 3),
    (10, 30, 10, 10, 20, 8, 4, 4, 2, 2),
    (16, 24, 12, 8, 18, 10, 4, 4, 2, 2),
    (12, 30, 8, 8, 18, 10, 6, 4, 2, 2),
    (12, 30, 8, 8, 18, 10, 4, 6, 2, 2),
    (16, 30, 8, 6, 16, 8, 4, 4, 4, 4),
    (10, 28, 12, 10, 18, 8, 4, 4, 3, 3),
    (14, 24, 12, 10, 18, 8, 4, 4, 3, 3),
    (12, 26, 10, 10, 18, 10, 4, 4, 3, 3),
    (14, 30, 8, 8, 18, 8, 4, 4, 3, 3),
    (16, 26, 8, 8, 18, 10, 4, 4, 3, 3),
    (12, 32, 8, 8, 18, 8, 4, 4, 3, 3),
)


def iter_enhanced_weight_grid() -> list[dict[str, float]]:
    return [
        {
            feature: value / 100.0
            for feature, value in zip(
                ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
                row,
                strict=True,
            )
        }
        for row in _ENHANCED_WEIGHT_GRID
    ]


def select_best_enhanced_weights(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        raise ValueError("enhanced grid search results are empty")

    best = min(
        results,
        key=lambda item: (
            -float(dict(item["valid_metrics"])["map_at_12"]),
            -float(dict(item["valid_metrics"])["ndcg_at_12"]),
            int(item["grid_index"]),
        ),
    )
    return {
        feature: float(dict(best["weights"])[feature])
        for feature in ENHANCED_RECOMMENDATION_SCORE_COLUMNS
    }
