from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.recommendation.time_windows import (
    build_recommendation_windows,
    validate_recommendation_windows,
)


def test_build_recommendation_windows_uses_cutoff_week() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["valid", "valid", "test"],
            "week_id": [104, 105, 106],
            "attr_type": ["product_type_name"] * 3,
            "attr_id": [1, 2, 3],
            "attr_value": ["A", "B", "C"],
            "pred_target_growth": [0.1, 0.2, 0.3],
            "pred_share_t1": [0.3, 0.3, 0.4],
        }
    )

    windows = build_recommendation_windows(predictions)

    assert windows.to_dict("records") == [
        {"split": "valid", "cutoff_week": 104, "label_week": 105},
        {"split": "valid", "cutoff_week": 105, "label_week": 106},
        {"split": "test", "cutoff_week": 106, "label_week": 107},
    ]


def test_build_recommendation_windows_requires_valid_and_test() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["valid"],
            "week_id": [104],
            "attr_type": ["product_type_name"],
            "attr_id": [1],
            "attr_value": ["A"],
            "pred_target_growth": [0.1],
            "pred_share_t1": [1.0],
        }
    )

    with pytest.raises(ValueError, match="test"):
        build_recommendation_windows(predictions)


def test_validate_recommendation_windows_rejects_duplicate_windows() -> None:
    windows = pd.DataFrame(
        [
            {"split": "valid", "cutoff_week": 104, "label_week": 105},
            {"split": "valid", "cutoff_week": 104, "label_week": 105},
            {"split": "test", "cutoff_week": 106, "label_week": 107},
        ]
    )

    with pytest.raises(ValueError, match="重复"):
        validate_recommendation_windows(windows)
