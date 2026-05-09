from __future__ import annotations

from typing import Any

POP_VALUES = (0.2, 0.3, 0.4)
SIM_VALUES = (0.2, 0.3, 0.4)
TREND_VALUES = (0.1, 0.2, 0.3)
RECENT_VALUES = (0.0, 0.05, 0.1)


def iter_weight_grid() -> list[dict[str, float]]:
    grid: list[dict[str, float]] = []
    for pop_score in POP_VALUES:
        for sim_score in SIM_VALUES:
            for trend_score in TREND_VALUES:
                for recent_score in RECENT_VALUES:
                    weights = {
                        "pop_score": pop_score,
                        "sim_score": sim_score,
                        "trend_score": trend_score,
                        "recent_score": recent_score,
                    }
                    if abs(sum(weights.values()) - 1.0) <= 1e-9:
                        grid.append(weights)
    return grid


def select_best_weights(
    results: list[dict[str, Any]],
    metric_name: str = "map_at_12",
) -> dict[str, float]:
    if not results:
        raise ValueError("grid search results are empty")

    best = min(
        results,
        key=lambda item: (
            -float(dict(item["valid_metrics"])[metric_name]),
            int(item["grid_index"]),
        ),
    )
    return dict(best["weights"])
