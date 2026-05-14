from __future__ import annotations

from typing import Any

from fashion_trend.recommendation.contracts import (
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
)

_ENHANCED_WEIGHT_GRID = (
    (14, 30, 10, 8, 16, 8, 4, 4, 3, 3),
    (16, 28, 10, 8, 16, 8, 4, 4, 3, 3),
    (12, 32, 10, 8, 16, 8, 4, 4, 3, 3),
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


def select_best_enhanced_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("enhanced grid search results are empty")

    return min(
        results,
        key=lambda item: (
            -float(dict(item["valid_metrics"])["map_at_12"]),
            -float(dict(item["valid_metrics"])["ndcg_at_12"]),
            int(item["grid_index"]),
        ),
    )


def select_best_enhanced_weights(results: list[dict[str, Any]]) -> dict[str, float]:
    best = select_best_enhanced_result(results)
    return {
        feature: float(dict(best["weights"])[feature])
        for feature in ENHANCED_RECOMMENDATION_SCORE_COLUMNS
    }
