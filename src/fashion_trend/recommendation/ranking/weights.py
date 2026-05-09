from __future__ import annotations

import math
from collections.abc import Sequence


def validate_score_weights(
    weights: dict[str, float],
    required_features: Sequence[str],
) -> dict[str, float]:
    """Validate score weights without normalizing them."""
    required = set(required_features)
    actual = set(weights)
    if actual != required:
        raise ValueError(
            "score weights keys mismatch: "
            f"expected={sorted(required)}, actual={sorted(actual)}"
        )

    result: dict[str, float] = {}
    for feature, value in weights.items():
        numeric_value = float(value)
        if numeric_value < 0 or not math.isfinite(numeric_value):
            raise ValueError(f"invalid weight for {feature}: {value}")
        result[feature] = numeric_value

    total = sum(result.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"score weights sum must be 1.0, got {total}")
    return dict(result)
