from __future__ import annotations

from typing import Any

WEIGHT_GRID = (
    (0.4, 0.2, 0.3, 0.1),
    (0.4, 0.3, 0.2, 0.1),
    (0.3, 0.3, 0.3, 0.1),
    (0.2, 0.4, 0.3, 0.1),
    (0.4, 0.1, 0.0, 0.5),
    (0.3, 0.1, 0.2, 0.4),
    (0.3, 0.0, 0.2, 0.5),
    (0.2, 0.1, 0.2, 0.5),
    (0.2, 0.0, 0.3, 0.5),
    (0.4, 0.0, 0.1, 0.5),
    (0.2, 0.2, 0.1, 0.5),
    (0.3, 0.2, 0.1, 0.4),
    (0.4, 0.2, 0.0, 0.4),
    (0.4, 0.1, 0.1, 0.4),
    (0.3, 0.2, 0.2, 0.3),
    (0.3, 0.1, 0.3, 0.3),
    (0.4, 0.1, 0.2, 0.3),
    (0.4, 0.2, 0.1, 0.3),
    (0.2, 0.3, 0.2, 0.3),
    (0.2, 0.2, 0.3, 0.3),
    (0.3, 0.3, 0.0, 0.4),
    (0.4, 0.3, 0.0, 0.3),
    (0.2, 0.3, 0.0, 0.5),
    (0.2, 0.2, 0.4, 0.2),
    (0.3, 0.0, 0.4, 0.3),
)


def iter_weight_grid() -> list[dict[str, float]]:
    return [
        {
            "pop_score": pop_score,
            "sim_score": sim_score,
            "trend_score": trend_score,
            "recent_score": recent_score,
        }
        for pop_score, sim_score, trend_score, recent_score in WEIGHT_GRID
    ]


def select_best_weights(
    results: list[dict[str, Any]],
    metric_name: str = "ndcg_at_12",
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
